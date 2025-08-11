"""
Enhanced inference module with uncertainty quantification and reliability metrics
Integrates data quality monitoring with prediction confidence intervals
"""

import torch
import numpy as np
from datetime import datetime
import time
import json
import os
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

import config
from data_handler import StockDataHandler
from model import StockPredictionTransformer
from dataset import get_sequence_for_prediction
from data_quality import DataQualityMetrics, QualityAlert, AlertLevel


class ReliabilityMetrics:
    """Container for prediction reliability metrics"""
    
    def __init__(self, confidence_interval: Tuple[float, float], 
                 uncertainty: float, data_quality_score: float,
                 prediction_reliability: float):
        self.confidence_interval = confidence_interval
        self.uncertainty = uncertainty
        self.data_quality_score = data_quality_score
        self.prediction_reliability = prediction_reliability
        
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'confidence_interval': {
                'lower': float(self.confidence_interval[0]),
                'upper': float(self.confidence_interval[1])
            },
            'uncertainty': float(self.uncertainty),
            'data_quality_score': float(self.data_quality_score),
            'prediction_reliability': float(self.prediction_reliability)
        }


class EnhancedRealTimePredictor:
    """
    Enhanced real-time prediction engine with uncertainty quantification and reliability assessment
    """
    
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.data_handler = StockDataHandler()
        self.signal_threshold = config.SIGNAL_THRESHOLD
        
        # Load model and scalers
        self.load_model(model_path)
        
        # Prediction and reliability history
        self.prediction_history = []
        self.reliability_history = []
        
        # Reliability thresholds
        self.min_reliability_threshold = 0.5
        self.min_data_quality_threshold = 0.6
        
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
            self.model.to(self.device)
            
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Model loaded from {model_path}")
            else:
                print(f"Warning: Model file {model_path} not found. Using untrained model.")
                
            self.model.eval()
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def calculate_prediction_reliability(self, uncertainty: float, 
                                       data_quality_metrics: DataQualityMetrics) -> float:
        """
        Calculate overall prediction reliability score based on uncertainty and data quality
        
        Args:
            uncertainty: Model prediction uncertainty (standard deviation)
            data_quality_metrics: Current data quality metrics
            
        Returns:
            Reliability score between 0 and 1 (higher = more reliable)
        """
        # Normalize uncertainty (assuming typical uncertainty range 0-10% of price)
        uncertainty_score = max(0, 1.0 - min(uncertainty / 0.1, 1.0))
        
        # Data quality score (already 0-1)
        quality_score = data_quality_metrics.quality_score
        
        # Combine with weights: 60% data quality, 40% model uncertainty
        reliability = 0.6 * quality_score + 0.4 * uncertainty_score
        
        return min(max(reliability, 0.0), 1.0)  # Clamp to [0, 1]
    
    def predict_with_reliability(self, confidence_level: float = 0.95, 
                               n_mc_samples: int = 50) -> Tuple[float, ReliabilityMetrics, List[QualityAlert]]:
        """
        Make prediction with comprehensive reliability assessment
        
        Args:
            confidence_level: Confidence level for prediction intervals (e.g., 0.95)
            n_mc_samples: Number of Monte Carlo samples for uncertainty estimation
            
        Returns:
            Tuple of (prediction, reliability_metrics, quality_alerts)
        """
        # Get latest data with quality monitoring
        latest_data = self.data_handler.get_latest_data()
        
        if len(latest_data) < config.SEQUENCE_LENGTH:
            raise ValueError(f"Insufficient data for prediction. Need {config.SEQUENCE_LENGTH}, got {len(latest_data)}")
        
        # Get data quality metrics and alerts
        quality_metrics, quality_alerts = self.data_handler.quality_monitor.monitor_data_quality(
            latest_data, update_reference=True
        )
        
        # Prepare sequence for prediction
        sequence = get_sequence_for_prediction(latest_data, config.SEQUENCE_LENGTH)
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        # Get prediction with confidence intervals
        with torch.no_grad():
            mean_pred, lower_bound, upper_bound = self.model.get_prediction_interval(
                sequence_tensor, confidence=confidence_level, n_samples=n_mc_samples
            )
            
            # Also get uncertainty estimate
            _, variance = self.model.predict_with_uncertainty(sequence_tensor, n_samples=n_mc_samples)
            uncertainty = torch.sqrt(variance).item()
        
        # Convert to actual price (denormalize)
        current_price = latest_data['Close'].iloc[-1]
        if hasattr(self.data_handler.target_scaler, 'data_min_') and hasattr(self.data_handler.target_scaler, 'data_max_'):
            # Denormalize prediction
            close_idx = config.FEATURE_COLUMNS.index('Close')
            scale = self.data_handler.target_scaler.data_max_[close_idx] - self.data_handler.target_scaler.data_min_[close_idx]
            min_val = self.data_handler.target_scaler.data_min_[close_idx]
            
            prediction = mean_pred.item() * scale + min_val
            lower_ci = lower_bound.item() * scale + min_val
            upper_ci = upper_bound.item() * scale + min_val
            uncertainty_actual = uncertainty * scale
        else:
            # Fallback: assume normalized prediction
            prediction = mean_pred.item()
            lower_ci = lower_bound.item()
            upper_ci = upper_bound.item()
            uncertainty_actual = uncertainty
        
        # Calculate prediction reliability
        reliability_score = self.calculate_prediction_reliability(uncertainty_actual, quality_metrics)
        
        # Create reliability metrics
        reliability_metrics = ReliabilityMetrics(
            confidence_interval=(lower_ci, upper_ci),
            uncertainty=uncertainty_actual,
            data_quality_score=quality_metrics.quality_score,
            prediction_reliability=reliability_score
        )
        
        return prediction, reliability_metrics, quality_alerts
    
    def generate_enhanced_signal(self, prediction: float, reliability_metrics: ReliabilityMetrics,
                                current_price: float) -> Dict:
        """
        Generate trading signal with reliability assessment
        
        Args:
            prediction: Predicted price
            reliability_metrics: Prediction reliability metrics
            current_price: Current market price
            
        Returns:
            Enhanced signal dictionary with reliability information
        """
        # Basic signal calculation
        price_change_pct = (prediction - current_price) / current_price
        
        # Adjust signal strength based on reliability
        reliability_factor = reliability_metrics.prediction_reliability
        adjusted_threshold = self.signal_threshold / reliability_factor if reliability_factor > 0 else float('inf')
        
        # Generate signal
        if reliability_metrics.prediction_reliability < self.min_reliability_threshold:
            signal = "HOLD"
            confidence = "LOW"
            reason = "Low prediction reliability"
        elif reliability_metrics.data_quality_score < self.min_data_quality_threshold:
            signal = "HOLD"
            confidence = "LOW"
            reason = "Poor data quality"
        elif price_change_pct > adjusted_threshold:
            signal = "BUY"
            confidence = "HIGH" if reliability_factor > 0.8 else "MEDIUM"
            reason = f"Strong buy signal with {reliability_factor:.1%} reliability"
        elif price_change_pct < -adjusted_threshold:
            signal = "SELL"
            confidence = "HIGH" if reliability_factor > 0.8 else "MEDIUM"
            reason = f"Strong sell signal with {reliability_factor:.1%} reliability"
        else:
            signal = "HOLD"
            confidence = "MEDIUM"
            reason = "Weak signal, holding position"
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reason': reason,
            'predicted_price': prediction,
            'current_price': current_price,
            'price_change_pct': price_change_pct,
            'reliability_metrics': reliability_metrics.to_dict(),
            'timestamp': datetime.now().isoformat()
        }
    
    def run_enhanced_inference(self, save_results: bool = True) -> Dict:
        """
        Run complete enhanced inference with reliability assessment
        
        Args:
            save_results: Whether to save results to file
            
        Returns:
            Complete inference results dictionary
        """
        try:
            # Make prediction with reliability assessment
            prediction, reliability_metrics, quality_alerts = self.predict_with_reliability()
            
            # Get current price for signal generation
            raw_data = self.data_handler.get_latest_data_raw()
            current_price = raw_data['Close'].iloc[-1] if len(raw_data) > 0 else None
            
            if current_price is None:
                raise ValueError("Could not get current price")
            
            # Generate enhanced signal
            signal_data = self.generate_enhanced_signal(prediction, reliability_metrics, current_price)
            
            # Compile results
            results = {
                'prediction': {
                    'value': prediction,
                    'confidence_interval': reliability_metrics.confidence_interval,
                    'uncertainty': reliability_metrics.uncertainty
                },
                'signal': signal_data,
                'data_quality': {
                    'score': reliability_metrics.data_quality_score,
                    'alerts': [
                        {
                            'level': alert.level.value,
                            'metric': alert.metric,
                            'message': alert.message,
                            'timestamp': alert.timestamp.isoformat()
                        } for alert in quality_alerts
                    ]
                },
                'reliability': {
                    'score': reliability_metrics.prediction_reliability,
                    'status': 'HIGH' if reliability_metrics.prediction_reliability > 0.8 
                             else 'MEDIUM' if reliability_metrics.prediction_reliability > 0.5 
                             else 'LOW'
                },
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'model_device': str(self.device),
                    'target_symbol': config.TARGET_SYMBOL
                }
            }
            
            # Save results if requested
            if save_results:
                self.save_inference_results(results)
            
            return results
            
        except Exception as e:
            error_result = {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'status': 'FAILED'
            }
            
            if save_results:
                self.save_inference_results(error_result)
            
            return error_result
    
    def save_inference_results(self, results: Dict):
        """Save inference results to file"""
        os.makedirs(config.RESULTS_SAVE_PATH, exist_ok=True)
        
        # Save latest results
        with open(os.path.join(config.RESULTS_SAVE_PATH, 'latest_inference.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Append to history
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        history_file = os.path.join(config.RESULTS_SAVE_PATH, f'inference_history_{timestamp}.json')
        with open(history_file, 'w') as f:
            json.dump(results, f, indent=2)


def main():
    """
    Test enhanced inference functionality
    """
    model_path = os.path.join(config.MODEL_SAVE_PATH, "best_model.pth")
    
    try:
        predictor = EnhancedRealTimePredictor(model_path)
        
        print("Running enhanced inference with reliability assessment...")
        print("=" * 60)
        
        results = predictor.run_enhanced_inference()
        
        if 'error' in results:
            print(f"Inference failed: {results['error']}")
            return
        
        # Display results
        print(f"Prediction: {results['prediction']['value']:.2f}")
        print(f"Confidence Interval: [{results['prediction']['confidence_interval'][0]:.2f}, {results['prediction']['confidence_interval'][1]:.2f}]")
        print(f"Uncertainty: ±{results['prediction']['uncertainty']:.2f}")
        print(f"Signal: {results['signal']['signal']} ({results['signal']['confidence']} confidence)")
        print(f"Reason: {results['signal']['reason']}")
        print(f"Data Quality Score: {results['data_quality']['score']:.3f}")
        print(f"Prediction Reliability: {results['reliability']['score']:.3f} ({results['reliability']['status']})")
        
        if results['data_quality']['alerts']:
            print(f"\nData Quality Alerts ({len(results['data_quality']['alerts'])}):")
            for alert in results['data_quality']['alerts']:
                print(f"  {alert['level'].upper()}: {alert['message']}")
        
        print(f"\nResults saved to: {config.RESULTS_SAVE_PATH}")
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()