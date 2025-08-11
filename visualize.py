"""
Visualization module for stock prediction AI
Provides comprehensive charts and analysis tools
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import config


# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ModelVisualizer:
    """
    Comprehensive visualization for model results
    """
    def __init__(self):
        self.results_path = config.RESULTS_SAVE_PATH
        
    def plot_training_history(self, history_file: str = "training_history.json"):
        """
        Plot training and validation loss curves
        """
        history_path = os.path.join(self.results_path, history_file)
        
        if not os.path.exists(history_path):
            print(f"Training history file not found: {history_path}")
            return
        
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(history['train_losses']) + 1)
        
        # Loss curves
        ax1.plot(epochs, history['train_losses'], label='Train Loss', marker='o', markersize=3)
        ax1.plot(epochs, history['val_losses'], label='Validation Loss', marker='s', markersize=3)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # RMSE over time
        rmse_values = [metrics['rmse'] for metrics in history['metrics_history']]
        ax2.plot(epochs, rmse_values, label='Validation RMSE', color='orange', marker='d', markersize=3)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('RMSE')
        ax2.set_title('Root Mean Square Error')
        ax2.grid(True, alpha=0.3)
        
        # Directional accuracy
        dir_acc = [metrics['directional_accuracy'] for metrics in history['metrics_history']]
        ax3.plot(epochs, dir_acc, label='Directional Accuracy', color='green', marker='^', markersize=3)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Directional Accuracy (Trading Signal Quality)')
        ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random (50%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # R-squared
        r2_values = [metrics['r2'] for metrics in history['metrics_history']]
        ax4.plot(epochs, r2_values, label='R²', color='purple', marker='*', markersize=4)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('R² Score')
        ax4.set_title('Model Fit Quality (R²)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, "training_history.png"), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_backtest_results(self, portfolio_file: str = "portfolio_history.csv"):
        """
        Plot backtesting results with portfolio performance
        """
        portfolio_path = os.path.join(self.results_path, portfolio_file)
        
        if not os.path.exists(portfolio_path):
            print(f"Portfolio history file not found: {portfolio_path}")
            return
        
        portfolio_df = pd.read_csv(portfolio_path)
        portfolio_df['timestamp'] = pd.to_datetime(portfolio_df['timestamp'])
        
        # Create interactive plot with plotly
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Portfolio Value', 'Stock Price', 'Position Size'),
            vertical_spacing=0.08,
            row_heights=[0.4, 0.4, 0.2]
        )
        
        # Portfolio value
        portfolio_return = (portfolio_df['portfolio_value'] / config.BACKTEST_INITIAL_CAPITAL - 1) * 100
        fig.add_trace(
            go.Scatter(x=portfolio_df['timestamp'], y=portfolio_return,
                      name='Portfolio Return (%)', line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        # Stock price
        fig.add_trace(
            go.Scatter(x=portfolio_df['timestamp'], y=portfolio_df['stock_price'],
                      name='Stock Price (¥)', line=dict(color='black', width=1)),
            row=2, col=1
        )
        
        # Position size
        fig.add_trace(
            go.Scatter(x=portfolio_df['timestamp'], y=portfolio_df['position'],
                      name='Position Size', fill='tonexty', line=dict(color='green', width=1)),
            row=3, col=1
        )
        
        fig.update_layout(
            title=f"Backtesting Results - {config.TARGET_SYMBOL}",
            height=800,
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Price (¥)", row=2, col=1)
        fig.update_yaxes(title_text="Shares", row=3, col=1)
        
        # Save and show
        fig.write_html(os.path.join(self.results_path, "backtest_results.html"))
        fig.show()
    
    def plot_prediction_accuracy(self, backtest_file: str = "backtest_results.json"):
        """
        Plot prediction vs actual prices
        """
        backtest_path = os.path.join(self.results_path, backtest_file)
        
        if not os.path.exists(backtest_path):
            print(f"Backtest results file not found: {backtest_path}")
            return
        
        with open(backtest_path, 'r') as f:
            results = json.load(f)
        
        # Extract portfolio history
        portfolio_history = results.get('portfolio_history', [])
        if not portfolio_history:
            print("No portfolio history found in backtest results")
            return
        
        # Create DataFrame
        df = pd.DataFrame(portfolio_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # For this visualization, we'll create a simple accuracy plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot stock price over time
        ax1.plot(df['timestamp'], df['stock_price'], label='Actual Price', linewidth=1.5, alpha=0.8)
        ax1.set_ylabel('Price (¥)')
        ax1.set_title(f'{config.TARGET_SYMBOL} - Stock Price During Backtest Period')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot portfolio value
        portfolio_return = (df['portfolio_value'] / config.BACKTEST_INITIAL_CAPITAL - 1) * 100
        ax2.plot(df['timestamp'], portfolio_return, label='Portfolio Return', color='green', linewidth=2)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Breakeven')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Return (%)')
        ax2.set_title('Portfolio Performance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, "prediction_accuracy.png"), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_trading_signals(self, backtest_file: str = "backtest_results.json"):
        """
        Plot trading signals and entry/exit points
        """
        backtest_path = os.path.join(self.results_path, backtest_file)
        
        if not os.path.exists(backtest_path):
            print(f"Backtest results file not found: {backtest_path}")
            return
        
        with open(backtest_path, 'r') as f:
            results = json.load(f)
        
        trades = results.get('trades', [])
        portfolio_history = results.get('portfolio_history', [])
        
        if not trades or not portfolio_history:
            print("No trading data found")
            return
        
        # Create DataFrames
        trades_df = pd.DataFrame(trades)
        portfolio_df = pd.DataFrame(portfolio_history)
        
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        portfolio_df['timestamp'] = pd.to_datetime(portfolio_df['timestamp'])
        
        # Create interactive plot
        fig = go.Figure()
        
        # Stock price line
        fig.add_trace(go.Scatter(
            x=portfolio_df['timestamp'],
            y=portfolio_df['stock_price'],
            mode='lines',
            name='Stock Price',
            line=dict(color='black', width=1)
        ))
        
        # Buy signals
        buy_trades = trades_df[trades_df['action'] == 'BUY']
        if not buy_trades.empty:
            fig.add_trace(go.Scatter(
                x=buy_trades['timestamp'],
                y=buy_trades['price'],
                mode='markers',
                name='Buy Signal',
                marker=dict(color='green', size=10, symbol='triangle-up')
            ))
        
        # Sell signals
        sell_trades = trades_df[trades_df['action'] == 'SELL']
        if not sell_trades.empty:
            fig.add_trace(go.Scatter(
                x=sell_trades['timestamp'],
                y=sell_trades['price'],
                mode='markers',
                name='Sell Signal',
                marker=dict(color='red', size=10, symbol='triangle-down')
            ))
        
        fig.update_layout(
            title=f"Trading Signals - {config.TARGET_SYMBOL}",
            xaxis_title="Date",
            yaxis_title="Price (¥)",
            hovermode='x unified',
            height=600
        )
        
        # Save and show
        fig.write_html(os.path.join(self.results_path, "trading_signals.html"))
        fig.show()
    
    def create_dashboard(self):
        """
        Create comprehensive dashboard with all visualizations
        """
        print("Creating comprehensive dashboard...")
        
        # Create all plots
        self.plot_training_history()
        self.plot_backtest_results()
        self.plot_prediction_accuracy()
        self.plot_trading_signals()
        
        # Load and display summary metrics
        self.display_summary_metrics()
        
        print("Dashboard created successfully!")
        print(f"Results saved to: {self.results_path}")
    
    def display_summary_metrics(self):
        """
        Display summary of all metrics
        """
        # Load test results
        test_results_path = os.path.join(self.results_path, "test_results.json")
        backtest_results_path = os.path.join(self.results_path, "backtest_results.json")
        
        print("\n" + "="*60)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*60)
        
        # Model accuracy metrics
        if os.path.exists(test_results_path):
            with open(test_results_path, 'r') as f:
                test_results = json.load(f)
            
            test_metrics = test_results.get('test_metrics', {})
            print("\nPrediction Accuracy:")
            print(f"  RMSE: {test_metrics.get('rmse', 0):.6f}")
            print(f"  MAE: {test_metrics.get('mae', 0):.6f}")
            print(f"  Directional Accuracy: {test_metrics.get('directional_accuracy', 0):.2%}")
            print(f"  R² Score: {test_metrics.get('r2', 0):.4f}")
        
        # Trading performance metrics
        if os.path.exists(backtest_results_path):
            with open(backtest_results_path, 'r') as f:
                backtest_results = json.load(f)
            
            trading_metrics = backtest_results.get('metrics', {})
            print("\nTrading Performance:")
            print(f"  Total Return: {trading_metrics.get('total_return', 0):.2%}")
            print(f"  Annual Return: {trading_metrics.get('annual_return', 0):.2%}")
            print(f"  Sharpe Ratio: {trading_metrics.get('sharpe_ratio', 0):.4f}")
            print(f"  Max Drawdown: {trading_metrics.get('max_drawdown', 0):.2%}")
            print(f"  Win Rate: {trading_metrics.get('win_rate', 0):.2%}")
            print(f"  Number of Trades: {trading_metrics.get('num_trades', 0)}")
            print(f"  Final Portfolio Value: ¥{trading_metrics.get('final_portfolio_value', 0):,.0f}")
        
        print("="*60)
    
    def plot_live_predictions(self, prediction_file: str = "prediction_history.json"):
        """
        Plot live prediction history with enhanced error analysis
        """
        prediction_path = os.path.join(self.results_path, prediction_file)
        
        if not os.path.exists(prediction_path):
            print(f"Prediction history file not found: {prediction_path}")
            return
        
        with open(prediction_path, 'r') as f:
            predictions = json.load(f)
        
        if not predictions:
            print("No prediction history found")
            return
        
        pred_df = pd.DataFrame(predictions)
        pred_df['timestamp'] = pd.to_datetime(pred_df['timestamp'])
        
        # Calculate prediction errors
        pred_df['absolute_error'] = abs(pred_df['predicted_price'] - pred_df['current_price'])
        pred_df['percentage_error'] = abs(pred_df['price_change_pct']) * 100
        pred_df['rolling_rmse'] = np.sqrt(pred_df['absolute_error'].rolling(window=10, min_periods=1).mean())
        
        # Calculate directional accuracy
        if len(pred_df) > 1:
            pred_direction = pred_df['price_change_pct'] > 0
            actual_direction = pred_df['current_price'].diff() > 0
            actual_direction_aligned = actual_direction.iloc[1:].reset_index(drop=True)
            pred_direction_aligned = pred_direction.iloc[:-1].reset_index(drop=True)
            
            direction_correct = (pred_direction_aligned == actual_direction_aligned)
            pred_df['direction_correct'] = [False] + direction_correct.tolist()
        else:
            pred_df['direction_correct'] = [True] * len(pred_df)
        
        # Create enhanced interactive plot
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=(
                'Actual vs Predicted Prices', 'Prediction Error Over Time',
                'Trading Signals on Price Chart', 'Rolling RMSE (10-period)',
                'Confidence Levels', 'Directional Accuracy',
                'Error Distribution', 'Prediction vs Actual Scatter'
            ),
            vertical_spacing=0.06,
            horizontal_spacing=0.08,
            specs=[[{"colspan": 2}, None],
                   [{}, {}],
                   [{}, {}],
                   [{}, {}]]
        )
        
        # Row 1: Main price comparison (full width)
        fig.add_trace(
            go.Scatter(x=pred_df['timestamp'], y=pred_df['current_price'],
                      name='実際価格 (Actual)', line=dict(color='black', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=pred_df['timestamp'], y=pred_df['predicted_price'],
                      name='予測価格 (Predicted)', line=dict(color='blue', width=2, dash='dash')),
            row=1, col=1
        )
        
        # Row 2 Left: Trading signals
        colors = {'BUY': 'green', 'SELL': 'red', 'HOLD': 'gray'}
        for signal in ['BUY', 'SELL', 'HOLD']:
            signal_data = pred_df[pred_df['signal'] == signal]
            if not signal_data.empty:
                fig.add_trace(
                    go.Scatter(x=signal_data['timestamp'], y=signal_data['current_price'],
                              mode='markers', name=f'{signal}シグナル',
                              marker=dict(color=colors[signal], size=8),
                              showlegend=True),
                    row=2, col=1
                )
        
        # Row 2 Right: Rolling RMSE
        fig.add_trace(
            go.Scatter(x=pred_df['timestamp'], y=pred_df['rolling_rmse'],
                      name='Rolling RMSE', line=dict(color='red', width=2)),
            row=2, col=2
        )
        
        # Row 3 Left: Confidence levels
        fig.add_trace(
            go.Scatter(x=pred_df['timestamp'], y=pred_df['confidence'],
                      name='信頼度 (Confidence)', fill='tonexty', 
                      line=dict(color='orange', width=1)),
            row=3, col=1
        )
        
        # Row 3 Right: Directional accuracy (rolling 10-period)
        rolling_accuracy = pred_df['direction_correct'].rolling(window=10, min_periods=1).mean()
        fig.add_trace(
            go.Scatter(x=pred_df['timestamp'], y=rolling_accuracy,
                      name='方向性精度 (Direction Accuracy)', 
                      line=dict(color='green', width=2)),
            row=3, col=2
        )
        fig.add_hline(y=0.5, line_dash="dash", line_color="red", 
                     annotation_text="Random (50%)", row=3, col=2)
        
        # Row 4 Left: Error distribution
        fig.add_trace(
            go.Histogram(x=pred_df['absolute_error'], name='誤差分布 (Error Distribution)',
                        marker_color='lightblue', opacity=0.7),
            row=4, col=1
        )
        
        # Row 4 Right: Scatter plot of predictions vs actuals
        fig.add_trace(
            go.Scatter(x=pred_df['current_price'], y=pred_df['predicted_price'],
                      mode='markers', name='予測 vs 実際',
                      marker=dict(color='purple', size=6, opacity=0.6)),
            row=4, col=2
        )
        # Perfect prediction line
        min_price = min(pred_df['current_price'].min(), pred_df['predicted_price'].min())
        max_price = max(pred_df['current_price'].max(), pred_df['predicted_price'].max())
        fig.add_trace(
            go.Scatter(x=[min_price, max_price], y=[min_price, max_price],
                      mode='lines', name='Perfect Prediction',
                      line=dict(color='red', dash='dash')),
            row=4, col=2
        )
        
        # Calculate and display summary statistics
        rmse = np.sqrt(np.mean(pred_df['absolute_error'] ** 2))
        mae = np.mean(pred_df['absolute_error'])
        direction_acc = np.mean(pred_df['direction_correct'])
        
        fig.update_layout(
            title=f"リアルタイム予測分析 - RMSE: {rmse:.4f}, MAE: {mae:.4f}, 方向性精度: {direction_acc:.2%}",
            height=1200,
            showlegend=True
        )
        
        # Update axis labels
        fig.update_yaxes(title_text="価格 (¥)", row=1, col=1)
        fig.update_yaxes(title_text="価格 (¥)", row=2, col=1)
        fig.update_yaxes(title_text="RMSE", row=2, col=2)
        fig.update_yaxes(title_text="信頼度", row=3, col=1)
        fig.update_yaxes(title_text="精度", row=3, col=2)
        fig.update_yaxes(title_text="頻度", row=4, col=1)
        fig.update_yaxes(title_text="予測価格", row=4, col=2)
        fig.update_xaxes(title_text="時間", row=3, col=1)
        fig.update_xaxes(title_text="時間", row=3, col=2)
        fig.update_xaxes(title_text="絶対誤差", row=4, col=1)
        fig.update_xaxes(title_text="実際価格", row=4, col=2)
        
        # Save and show
        fig.write_html(os.path.join(self.results_path, "live_predictions.html"))
        fig.show()
        
        # Print summary statistics
        print(f"\n予測精度サマリー (Prediction Accuracy Summary):")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  方向性精度: {direction_acc:.2%}")
        print(f"  平均信頼度: {pred_df['confidence'].mean():.2f}")
        print(f"  予測回数: {len(pred_df)}")
    
    def create_market_analysis(self, data_file: str = "processed_data.csv"):
        """
        Create market analysis visualization
        """
        data_path = os.path.join(config.DATA_SAVE_PATH, data_file)
        
        if not os.path.exists(data_path):
            print(f"Data file not found: {data_path}")
            return
        
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        
        # Create correlation matrix
        feature_cols = config.FEATURE_COLUMNS + [f'Market_{col}' for col in config.FEATURE_COLUMNS]
        correlation_matrix = data[feature_cols].corr()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Correlation heatmap
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   fmt='.2f', square=True, ax=ax1)
        ax1.set_title('Feature Correlation Matrix')
        
        # Price and volume over time
        ax2.plot(data.index, data['Close'], label='Toyota Close Price', alpha=0.8)
        ax2_twin = ax2.twinx()
        ax2_twin.bar(data.index, data['Volume'], alpha=0.3, color='orange', label='Volume')
        ax2.set_ylabel('Price (¥)', color='blue')
        ax2_twin.set_ylabel('Volume', color='orange')
        ax2.set_title('Price and Volume Over Time')
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        
        # Price distribution
        ax3.hist(data['Close'], bins=50, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Close Price (¥)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Price Distribution')
        ax3.grid(True, alpha=0.3)
        
        # Returns distribution
        returns = data['Close'].pct_change().dropna()
        ax4.hist(returns, bins=50, alpha=0.7, edgecolor='black', color='green')
        ax4.axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.4f}')
        ax4.set_xlabel('5-minute Returns')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Returns Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, "market_analysis.png"), dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """
    Main visualization interface
    """
    visualizer = ModelVisualizer()
    
    print("Stock Prediction AI - Visualization Dashboard")
    print("=" * 50)
    print("Available visualizations:")
    print("1. Training history")
    print("2. Backtest results")
    print("3. Prediction accuracy")
    print("4. Trading signals")
    print("5. Live predictions")
    print("6. Market analysis")
    print("7. Complete dashboard (all plots)")
    
    choice = input("Select option (1-7): ").strip()
    
    if choice == "1":
        visualizer.plot_training_history()
    elif choice == "2":
        visualizer.plot_backtest_results()
    elif choice == "3":
        visualizer.plot_prediction_accuracy()
    elif choice == "4":
        visualizer.plot_trading_signals()
    elif choice == "5":
        visualizer.plot_live_predictions()
    elif choice == "6":
        visualizer.create_market_analysis()
    elif choice == "7":
        visualizer.create_dashboard()
    else:
        print("Invalid choice")
        return
    
    print("Visualization completed!")


if __name__ == "__main__":
    main()