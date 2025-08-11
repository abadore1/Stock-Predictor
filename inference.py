"""
Real-time inference module for stock prediction
Provides live trading signals for day trading strategies
"""

import torch
import numpy as np
from datetime import datetime
import time
import json
import os
from typing import Dict

import config
from data_handler import StockDataHandler
from model import StockPredictionTransformer
from dataset import get_sequence_for_prediction


class RealTimePredictor:
    """
    Real-time prediction engine for live trading
    """
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.data_handler = StockDataHandler()
        self.signal_threshold = config.SIGNAL_THRESHOLD
        
        # Load model and scalers
        self.load_model(model_path)
        
        # Prediction history
        self.prediction_history = []
        
    def load_model(self, model_path: str):
        """
        Load trained model and scalers
        """
        try:
            # Load scalers first
            self.data_handler.load_data()  # This loads the scalers
            
            # Get input dimension
            target_features = config.FEATURE_COLUMNS
            market_features = [f'Market_{col}' for col in config.FEATURE_COLUMNS]
            input_dim = len(target_features + market_features)
            
            # Create and load model
            self.model = StockPredictionTransformer(input_dim)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def get_current_prediction(self) -> Dict[str, float]:
        """
        Get prediction for current market conditions
        """
        try:
            # Fetch latest data
            latest_data = self.data_handler.get_latest_data()
            
            if len(latest_data) < config.SEQUENCE_LENGTH:
                raise ValueError(f"Insufficient data: need {config.SEQUENCE_LENGTH}, got {len(latest_data)}")
            
            # Get current price (before normalization)
            raw_latest_data = self.data_handler.get_latest_data_raw()
            current_price = raw_latest_data['Close'].iloc[-1]
            current_timestamp = latest_data.index[-1]
            
            # Prepare sequence for prediction
            sequence = get_sequence_for_prediction(latest_data, config.SEQUENCE_LENGTH)
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                sequence = sequence.to(self.device)
                predicted_price_normalized = self.model(sequence).item()
            
            # Denormalize prediction
            # Note: This assumes the Close price was the first feature column normalized
            close_feature_idx = config.FEATURE_COLUMNS.index('Close')
            dummy_features = np.zeros((1, len(config.FEATURE_COLUMNS)))
            dummy_features[0, close_feature_idx] = predicted_price_normalized
            
            denormalized = self.data_handler.target_scaler.inverse_transform(dummy_features)
            predicted_price = denormalized[0, close_feature_idx]
            
            # Calculate price change
            price_change = predicted_price - current_price
            price_change_pct = price_change / current_price
            
            # Generate signal
            if price_change_pct > self.signal_threshold:
                signal = "BUY"
                confidence = min(price_change_pct / self.signal_threshold, 2.0)
            elif price_change_pct < -self.signal_threshold:
                signal = "SELL"
                confidence = min(abs(price_change_pct) / self.signal_threshold, 2.0)
            else:
                signal = "HOLD"
                confidence = 1.0 - (abs(price_change_pct) / self.signal_threshold)
            
            result = {
                'timestamp': current_timestamp.isoformat(),
                'current_price': float(current_price),
                'predicted_price': float(predicted_price),
                'price_change': float(price_change),
                'price_change_pct': float(price_change_pct),
                'signal': signal,
                'confidence': float(confidence)
            }
            
            # Store in history
            self.prediction_history.append(result)
            
            return result
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            raise
    
    def get_market_summary(self) -> Dict[str, float]:
        """
        Get current market summary
        """
        try:
            latest_data = self.data_handler.get_latest_data()
            
            if latest_data.empty:
                return {}
            
            current_data = latest_data.iloc[-1]
            previous_data = latest_data.iloc[-2] if len(latest_data) > 1 else current_data
            
            return {
                'current_price': float(current_data['Close']),
                'price_change': float(current_data['Close'] - previous_data['Close']),
                'price_change_pct': float((current_data['Close'] - previous_data['Close']) / previous_data['Close']),
                'volume': float(current_data['Volume']),
                'high': float(current_data['High']),
                'low': float(current_data['Low']),
                'market_close': float(current_data['Market_Close']),
                'market_change_pct': float((current_data['Market_Close'] - previous_data['Market_Close']) / previous_data['Market_Close'])
            }
            
        except Exception as e:
            print(f"Error getting market summary: {e}")
            return {}
    
    def save_prediction_history(self):
        """
        Save prediction history to file
        """
        if not self.prediction_history:
            return
        
        history_path = os.path.join(config.RESULTS_SAVE_PATH, "prediction_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.prediction_history, f, indent=2)
        


def run_live_monitoring(model_path: str, interval_minutes: int = 5, max_iterations: int = 100):
    """
    Run live monitoring and prediction
    
    Args:
        model_path: Path to trained model
        interval_minutes: How often to make predictions (in minutes)
        max_iterations: Maximum number of prediction cycles
    """
    predictor = RealTimePredictor(model_path)
    
    print(f"Starting live monitoring for {config.TARGET_SYMBOL}")
    print(f"Prediction interval: {interval_minutes} minutes")
    print("Press Ctrl+C to stop")
    
    iteration = 0
    
    try:
        while iteration < max_iterations:
            print(f"\n--- Iteration {iteration + 1} ---")
            print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            try:
                # Get market summary
                market_summary = predictor.get_market_summary()
                if market_summary:
                    print(f"Current Price: ¥{market_summary['current_price']:.2f}")
                    print(f"Price Change: {market_summary['price_change_pct']:.2%}")
                    print(f"Volume: {market_summary['volume']:,.0f}")
                
                # Get prediction
                prediction = predictor.get_current_prediction()
                
                print(f"Predicted Price: ¥{prediction['predicted_price']:.2f}")
                print(f"Expected Change: {prediction['price_change_pct']:.2%}")
                print(f"Signal: {prediction['signal']} (Confidence: {prediction['confidence']:.2f})")
                
                # Save prediction
                predictor.save_prediction_history()
                
            except Exception as e:
                print(f"Error in iteration {iteration + 1}: {e}")
            
            iteration += 1
            
            # Wait for next iteration
            if iteration < max_iterations:
                print(f"Waiting {interval_minutes} minutes for next prediction...")
                time.sleep(interval_minutes * 60)
                
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    
    # Final save
    predictor.save_prediction_history()


def single_prediction(model_path: str) -> Dict[str, float]:
    """
    Make a single prediction for current market conditions
    """
    predictor = RealTimePredictor(model_path)
    
    print(f"Making prediction for {config.TARGET_SYMBOL}...")
    
    # Get market summary
    market_summary = predictor.get_market_summary()
    if market_summary:
        print(f"Current market conditions:")
        print(f"  Price: ¥{market_summary['current_price']:.2f}")
        print(f"  Change: {market_summary['price_change_pct']:.2%}")
        print(f"  Volume: {market_summary['volume']:,.0f}")
        print(f"  Nikkei ETF: ¥{market_summary['market_close']:.2f} ({market_summary['market_change_pct']:.2%})")
    
    # Get prediction
    prediction = predictor.get_current_prediction()
    
    print(f"\nPrediction results:")
    print(f"  Predicted price: ¥{prediction['predicted_price']:.2f}")
    print(f"  Expected change: {prediction['price_change_pct']:.2%}")
    print(f"  Trading signal: {prediction['signal']}")
    print(f"  Confidence: {prediction['confidence']:.2f}")
    
    # Save prediction
    predictor.save_prediction_history()
    
    return prediction


def main():
    """
    Main inference interface
    """
    model_path = os.path.join(config.MODEL_SAVE_PATH, "best_model.pth")
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train the model first.")
        return
    
    print("Stock Prediction AI - Real-time Inference")
    print("=" * 50)
    print("1. Single prediction")
    print("2. Live monitoring")
    
    choice = input("Select option (1 or 2): ").strip()
    
    if choice == "1":
        single_prediction(model_path)
    elif choice == "2":
        interval = int(input("Prediction interval in minutes (default: 5): ") or "5")
        max_iter = int(input("Maximum iterations (default: 100): ") or "100")
        run_live_monitoring(model_path, interval, max_iter)
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()