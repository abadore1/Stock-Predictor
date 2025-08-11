"""
Test suite for data quality monitoring and reliability improvement system
Simple validation tests that don't require network access
"""

import numpy as np
import pandas as pd
import torch
import os
import tempfile
import shutil
from datetime import datetime, timedelta
from typing import Dict, Any

import config
from data_quality import DataQualityMonitor, StatisticalOutlierDetector, DataDriftDetector
from model import StockPredictionTransformer


def create_sample_stock_data(n_samples: int = 1000, add_issues: bool = False) -> pd.DataFrame:
    """
    Create sample stock data for testing
    """
    np.random.seed(42)
    # Use recent dates to avoid freshness alerts
    end_date = datetime.now()
    dates = pd.date_range(end=end_date, periods=n_samples, freq='5min')
    
    # Generate realistic price movements
    base_price = 100.0
    returns = np.random.normal(0, 0.01, n_samples)  # 1% volatility
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    data = pd.DataFrame({
        'Open': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices], 
        'Close': prices,
        'Volume': np.random.normal(10000, 2000, n_samples)
    }, index=dates)
    
    # Ensure High >= Low >= Close relationship
    data['High'] = np.maximum(data['High'], data[['Open', 'Close']].max(axis=1))
    data['Low'] = np.minimum(data['Low'], data[['Open', 'Close']].min(axis=1))
    
    if add_issues:
        # Add missing values
        missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.02), replace=False)
        data.loc[data.index[missing_indices], 'Volume'] = np.nan
        
        # Add outliers
        outlier_indices = np.random.choice(n_samples, size=int(n_samples * 0.01), replace=False)
        data.loc[data.index[outlier_indices], 'Close'] *= np.random.choice([2, 0.5], len(outlier_indices))
        
        # Add drift (shift in recent data)
        drift_start = int(n_samples * 0.8)
        data.loc[data.index[drift_start:], 'Close'] *= 1.2
    
    return data


def test_outlier_detection():
    """Test statistical outlier detection"""
    print("Testing Statistical Outlier Detection...")
    
    # Create data with known outliers
    data = create_sample_stock_data(1000, add_issues=True)
    
    detector = StatisticalOutlierDetector()
    outliers = detector.detect_outliers(data, config.FEATURE_COLUMNS)
    
    # Check that outliers were detected
    outlier_rates = {}
    for col in outliers.columns:
        outlier_rate = outliers[col].sum() / len(outliers)
        outlier_rates[col] = outlier_rate
        print(f"  {col}: {outlier_rate:.3%} outliers detected")
    
    # Should detect some outliers but not too many
    total_outlier_rate = sum(outlier_rates.values()) / len(outlier_rates)
    assert 0.001 <= total_outlier_rate <= 0.1, f"Outlier rate {total_outlier_rate:.3%} seems unreasonable"
    
    print("  ✓ Outlier detection test passed")
    return True


def test_drift_detection():
    """Test data drift detection"""
    print("Testing Data Drift Detection...")
    
    # Create stable reference data
    reference_data = create_sample_stock_data(500, add_issues=False)
    
    # Create drifted data
    drifted_data = create_sample_stock_data(500, add_issues=False)
    drifted_data['Close'] *= 1.3  # Simulate drift
    
    detector = DataDriftDetector()
    detector.update_reference_distribution(reference_data, config.FEATURE_COLUMNS)
    
    # Test no drift (should be low)
    no_drift_scores = detector.detect_drift(reference_data.tail(100), config.FEATURE_COLUMNS)
    print(f"  No drift scores: {no_drift_scores}")
    
    # Test with drift (should be higher)
    drift_scores = detector.detect_drift(drifted_data.tail(100), config.FEATURE_COLUMNS)
    print(f"  Drift scores: {drift_scores}")
    
    # Drift scores should be higher for drifted data
    assert drift_scores['Close'] > no_drift_scores['Close'], "Drift detection failed"
    
    print("  ✓ Drift detection test passed")
    return True


def test_data_quality_monitor():
    """Test comprehensive data quality monitoring"""
    print("Testing Data Quality Monitor...")
    
    # Test with healthy data
    healthy_data = create_sample_stock_data(1000, add_issues=False)
    monitor = DataQualityMonitor()
    
    metrics, alerts = monitor.monitor_data_quality(healthy_data)
    print(f"  Healthy data quality score: {metrics.quality_score:.3f}")
    print(f"  Healthy data alerts: {len(alerts)}")
    
    # Should have high quality score and few/no alerts (excluding freshness alerts for test data)
    non_freshness_alerts = [a for a in alerts if not any(keyword in a.message.lower() for keyword in ['stale', 'fresh', 'hours old'])]
    assert metrics.quality_score > 0.7, f"Healthy data should have reasonable quality score, got {metrics.quality_score:.3f}"
    print(f"  Non-freshness alerts: {len(non_freshness_alerts)}")
    
    # Test with problematic data
    problem_data = create_sample_stock_data(1000, add_issues=True)
    metrics_bad, alerts_bad = monitor.monitor_data_quality(problem_data)
    print(f"  Problem data quality score: {metrics_bad.quality_score:.3f}")
    print(f"  Problem data alerts: {len(alerts_bad)}")
    
    # Should have lower quality score and more alerts
    assert metrics_bad.quality_score < metrics.quality_score, "Problem data should have lower quality score"
    
    print("  ✓ Data quality monitor test passed")
    return True


def test_model_uncertainty():
    """Test model uncertainty quantification"""
    print("Testing Model Uncertainty Quantification...")
    
    # Create a simple model
    input_dim = len(config.FEATURE_COLUMNS) * 2  # target + market features
    model = StockPredictionTransformer(input_dim)
    model.eval()
    
    # Create sample input
    batch_size = 4
    seq_len = config.SEQUENCE_LENGTH
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Test basic forward pass
    prediction = model(x)
    assert prediction.shape == (batch_size, 1), f"Expected shape {(batch_size, 1)}, got {prediction.shape}"
    
    # Test uncertainty quantification
    mean_pred, variance = model.predict_with_uncertainty(x, n_samples=10)
    assert mean_pred.shape == (batch_size, 1), "Mean prediction shape mismatch"
    assert variance.shape == (batch_size, 1), "Variance shape mismatch"
    assert torch.all(variance >= 0), "Variance should be non-negative"
    
    # Test confidence intervals
    mean_pred, lower, upper = model.get_prediction_interval(x, confidence=0.95, n_samples=10)
    assert torch.all(lower <= mean_pred), "Lower bound should be <= mean"
    assert torch.all(mean_pred <= upper), "Mean should be <= upper bound"
    
    print("  ✓ Model uncertainty test passed")
    return True


def test_integration():
    """Test integration between components"""
    print("Testing Component Integration...")
    
    # Create sample data
    data = create_sample_stock_data(500, add_issues=False)
    
    # Test data quality monitoring
    monitor = DataQualityMonitor()
    quality_summary = monitor.get_quality_summary(data)
    
    assert 'status' in quality_summary, "Quality summary missing status"
    assert 'metrics' in quality_summary, "Quality summary missing metrics"
    assert 'alerts' in quality_summary, "Quality summary missing alerts"
    
    print(f"  Integration test quality score: {quality_summary['metrics']['quality_score']:.3f}")
    print(f"  Integration test status: {quality_summary['status']}")
    
    print("  ✓ Integration test passed")
    return True


def run_all_tests():
    """Run all validation tests"""
    print("Running Data Quality and Reliability System Tests")
    print("=" * 60)
    
    tests = [
        test_outlier_detection,
        test_drift_detection,
        test_data_quality_monitor,
        test_model_uncertainty,
        test_integration
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"  ✗ {test_func.__name__} failed")
        except Exception as e:
            failed += 1
            print(f"  ✗ {test_func.__name__} failed with error: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✓ All tests passed! Data quality and reliability system is working correctly.")
    else:
        print("✗ Some tests failed. Please check the implementation.")
    
    return failed == 0


if __name__ == "__main__":
    run_all_tests()