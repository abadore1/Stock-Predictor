"""
Model evaluation and backtesting module
Includes trading simulation and performance analysis
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
import os
from datetime import datetime

import config
from data_handler import StockDataHandler
from dataset import create_data_splits, create_data_loaders, get_sequence_for_prediction
from model import StockPredictionTransformer


class TradingSimulator:
    """
    Backtesting simulator for trading strategies
    """
    def __init__(self, initial_capital: float = config.BACKTEST_INITIAL_CAPITAL,
                 transaction_cost: float = config.TRANSACTION_COST,
                 signal_threshold: float = config.SIGNAL_THRESHOLD):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.signal_threshold = signal_threshold
        
        self.reset()
    
    def reset(self):
        """Reset simulator state"""
        self.capital = self.initial_capital
        self.position = 0  # Number of shares held
        self.cash = self.initial_capital
        self.trades = []
        self.portfolio_values = []
        
    def generate_signal(self, predicted_price: float, current_price: float) -> str:
        """
        Generate trading signal based on prediction
        """
        price_change_pct = (predicted_price - current_price) / current_price
        
        if price_change_pct > self.signal_threshold:
            return "BUY"
        elif price_change_pct < -self.signal_threshold:
            return "SELL"
        else:
            return "HOLD"
    
    def execute_trade(self, signal: str, current_price: float, timestamp: pd.Timestamp):
        """
        Execute trading action based on signal
        """
        if signal == "BUY" and self.position == 0:
            # Buy with all available cash
            shares_to_buy = int(self.cash / (current_price * (1 + self.transaction_cost)))
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                self.cash -= cost
                self.position += shares_to_buy
                
                self.trades.append({
                    'timestamp': timestamp,
                    'action': 'BUY',
                    'shares': shares_to_buy,
                    'price': current_price,
                    'cost': cost
                })
        
        elif signal == "SELL" and self.position > 0:
            # Sell all shares
            proceeds = self.position * current_price * (1 - self.transaction_cost)
            self.cash += proceeds
            
            self.trades.append({
                'timestamp': timestamp,
                'action': 'SELL',
                'shares': self.position,
                'price': current_price,
                'proceeds': proceeds
            })
            
            self.position = 0
        
        # Calculate current portfolio value
        portfolio_value = self.cash + (self.position * current_price)
        self.portfolio_values.append({
            'timestamp': timestamp,
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'position': self.position,
            'stock_price': current_price
        })
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate trading performance metrics
        """
        if not self.portfolio_values:
            return {}
        
        portfolio_df = pd.DataFrame(self.portfolio_values)
        
        # Calculate returns
        portfolio_returns = portfolio_df['portfolio_value'].pct_change().dropna()
        
        # Total return
        total_return = (portfolio_df['portfolio_value'].iloc[-1] / self.initial_capital) - 1
        
        # Sharpe ratio (assuming 252 trading days, 288 5-min periods per day)
        annual_return = (1 + total_return) ** (252 * 288 / len(portfolio_df)) - 1
        volatility = portfolio_returns.std() * np.sqrt(252 * 288)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        rolling_max = portfolio_df['portfolio_value'].expanding().max()
        drawdown = (portfolio_df['portfolio_value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Win rate
        trades_df = pd.DataFrame(self.trades)
        if len(trades_df) > 0:
            buy_trades = trades_df[trades_df['action'] == 'BUY']
            sell_trades = trades_df[trades_df['action'] == 'SELL']
            
            if len(buy_trades) > 0 and len(sell_trades) > 0:
                profit_trades = 0
                for i in range(min(len(buy_trades), len(sell_trades))):
                    if sell_trades.iloc[i]['price'] > buy_trades.iloc[i]['price']:
                        profit_trades += 1
                win_rate = profit_trades / min(len(buy_trades), len(sell_trades))
            else:
                win_rate = 0
        else:
            win_rate = 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'win_rate': win_rate,
            'num_trades': len(self.trades),
            'final_portfolio_value': portfolio_df['portfolio_value'].iloc[-1]
        }


class ModelEvaluator:
    """
    Comprehensive model evaluation including backtesting
    """
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.data_handler = StockDataHandler()
        self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """
        Load trained model
        """
        # Load data to get input dimension
        try:
            data = self.data_handler.load_data()
        except FileNotFoundError:
            print("No processed data found. Please run data preparation first.")
            return
        
        # Get input dimension from feature columns
        target_features = config.FEATURE_COLUMNS
        market_features = [f'Market_{col}' for col in config.FEATURE_COLUMNS]
        input_dim = len(target_features + market_features)
        
        # Create model architecture
        self.model = StockPredictionTransformer(input_dim)
        
        # Load trained weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded model from {model_path}")
    
    def predict_batch(self, sequences: torch.Tensor) -> np.ndarray:
        """
        Make predictions on a batch of sequences
        """
        self.model.eval()
        with torch.no_grad():
            sequences = sequences.to(self.device)
            predictions = self.model(sequences)
            return predictions.cpu().numpy().flatten()
    
    def run_backtest(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        Run complete backtesting simulation
        """
        print("Starting backtesting simulation...")
        
        simulator = TradingSimulator()
        predictions = []
        actuals = []
        
        # Need at least sequence_length + 1 data points
        min_required = config.SEQUENCE_LENGTH + 1
        if len(test_data) < min_required:
            raise ValueError(f"Need at least {min_required} data points for backtesting")
        
        # Simulate trading day by day
        for i in range(config.SEQUENCE_LENGTH, len(test_data)):
            # Get sequence for prediction
            sequence_data = test_data.iloc[i-config.SEQUENCE_LENGTH:i]
            current_timestamp = test_data.index[i]
            current_price = test_data.iloc[i]['Close']
            
            try:
                # Get prediction sequence
                pred_sequence = get_sequence_for_prediction(
                    pd.concat([sequence_data], ignore_index=False),
                    config.SEQUENCE_LENGTH
                )
                
                # Make prediction
                predicted_price = self.predict_batch(pred_sequence)[0]
                predictions.append(predicted_price)
                actuals.append(current_price)
                
                # Generate trading signal
                signal = simulator.generate_signal(predicted_price, current_price)
                
                # Execute trade
                simulator.execute_trade(signal, current_price, current_timestamp)
                
            except Exception as e:
                print(f"Error at timestamp {current_timestamp}: {e}")
                continue
        
        # Get performance metrics
        trading_metrics = simulator.get_performance_metrics()
        
        # Calculate prediction accuracy metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        prediction_metrics = {
            'mse': np.mean((predictions - actuals) ** 2),
            'rmse': np.sqrt(np.mean((predictions - actuals) ** 2)),
            'mae': np.mean(np.abs(predictions - actuals)),
            'directional_accuracy': np.mean((np.diff(predictions) > 0) == (np.diff(actuals) > 0))
        }
        
        # Combine metrics
        all_metrics = {**prediction_metrics, **trading_metrics}
        
        print("Backtesting completed!")
        print(f"Total Return: {trading_metrics.get('total_return', 0):.2%}")
        print(f"Sharpe Ratio: {trading_metrics.get('sharpe_ratio', 0):.4f}")
        print(f"Max Drawdown: {trading_metrics.get('max_drawdown', 0):.2%}")
        print(f"Win Rate: {trading_metrics.get('win_rate', 0):.2%}")
        print(f"Number of Trades: {trading_metrics.get('num_trades', 0)}")
        
        return all_metrics, simulator.portfolio_values, simulator.trades


def main():
    """
    Main evaluation pipeline
    """
    # Load best model
    model_path = os.path.join(config.MODEL_SAVE_PATH, "best_model.pth")
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train the model first.")
        return
    
    # Create evaluator
    evaluator = ModelEvaluator(model_path)
    
    # Load data
    data = evaluator.data_handler.load_data()
    train_data, val_data, test_data = create_data_splits(data)
    
    # Run backtesting on test data
    metrics, portfolio_history, trades = evaluator.run_backtest(test_data)
    
    # Save results
    results = {
        'evaluation_timestamp': datetime.now().isoformat(),
        'metrics': {k: float(v) for k, v in metrics.items()},
        'portfolio_history': portfolio_history,
        'trades': trades
    }
    
    results_path = os.path.join(config.RESULTS_SAVE_PATH, "backtest_results.json")
    with open(results_path, 'w') as f:
        # Handle datetime serialization
        def convert_timestamps(obj):
            if isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_timestamps(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_timestamps(item) for item in obj]
            return obj
        
        results_serializable = convert_timestamps(results)
        json.dump(results_serializable, f, indent=2, default=str)
    
    print(f"Saved backtest results to {results_path}")
    
    # Create portfolio value DataFrame for easy analysis
    portfolio_df = pd.DataFrame(portfolio_history)
    if not portfolio_df.empty:
        portfolio_csv_path = os.path.join(config.RESULTS_SAVE_PATH, "portfolio_history.csv")
        portfolio_df.to_csv(portfolio_csv_path, index=False)
        print(f"Saved portfolio history to {portfolio_csv_path}")


if __name__ == "__main__":
    main()