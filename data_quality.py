"""
Data Quality Monitoring and Validation System
Implements statistical outlier detection, data drift detection, and health monitoring
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import warnings
from dataclasses import dataclass
from enum import Enum
import json
import os

import config


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning" 
    CRITICAL = "critical"


@dataclass
class DataQualityMetrics:
    """Data quality metrics container"""
    missing_value_rate: float
    outlier_rate: float
    drift_score: float
    data_freshness_hours: float
    completeness_score: float
    quality_score: float  # Overall quality score 0-1


@dataclass
class QualityAlert:
    """Data quality alert"""
    timestamp: datetime
    level: AlertLevel
    metric: str
    message: str
    value: float
    threshold: float


class StatisticalOutlierDetector:
    """
    Statistical outlier detection using multiple methods
    """
    
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        
    def detect_iqr_outliers(self, data: pd.Series) -> pd.Series:
        """Detect outliers using Interquartile Range method"""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        return (data < lower_bound) | (data > upper_bound)
    
    def detect_zscore_outliers(self, data: pd.Series, threshold: float = 3.0) -> pd.Series:
        """Detect outliers using Z-score method"""
        z_scores = np.abs((data - data.mean()) / data.std())
        return z_scores > threshold
    
    def detect_modified_zscore_outliers(self, data: pd.Series, threshold: float = 3.5) -> pd.Series:
        """Detect outliers using Modified Z-score (more robust)"""
        median = data.median()
        mad = np.median(np.abs(data - median))
        modified_z_scores = 0.6745 * (data - median) / mad
        return np.abs(modified_z_scores) > threshold
    
    def detect_outliers(self, data: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """
        Comprehensive outlier detection combining multiple methods
        
        Returns:
            DataFrame with boolean columns indicating outliers for each feature
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_results = pd.DataFrame(index=data.index)
        
        for col in columns:
            if col not in data.columns:
                continue
                
            series_data = data[col].dropna()
            if len(series_data) < 10:  # Not enough data
                outlier_results[f'{col}_outlier'] = False
                continue
            
            # Apply multiple methods
            iqr_outliers = self.detect_iqr_outliers(series_data)
            zscore_outliers = self.detect_zscore_outliers(series_data)
            modified_z_outliers = self.detect_modified_zscore_outliers(series_data)
            
            # Combine results - consider it an outlier if detected by 2+ methods
            combined_outliers = (iqr_outliers.astype(int) + 
                               zscore_outliers.astype(int) + 
                               modified_z_outliers.astype(int)) >= 2
            
            # Align with original dataframe
            outlier_column = pd.Series(False, index=data.index)
            outlier_column.loc[series_data.index] = combined_outliers
            outlier_results[f'{col}_outlier'] = outlier_column
        
        return outlier_results


class DataDriftDetector:
    """
    Data drift detection using statistical tests
    """
    
    def __init__(self, reference_window: int = 30, detection_window: int = 10):
        self.reference_window = reference_window
        self.detection_window = detection_window
        self.reference_stats = {}
    
    def calculate_distribution_stats(self, data: pd.Series) -> Dict[str, float]:
        """Calculate statistical properties of data distribution"""
        return {
            'mean': data.mean(),
            'std': data.std(),
            'median': data.median(),
            'q25': data.quantile(0.25),
            'q75': data.quantile(0.75),
            'skewness': data.skew(),
            'kurtosis': data.kurtosis()
        }
    
    def update_reference_distribution(self, data: pd.DataFrame, columns: List[str] = None):
        """Update reference distribution using recent stable data"""
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Use the most recent stable period as reference
        reference_data = data.tail(self.reference_window)
        
        for col in columns:
            if col in reference_data.columns:
                self.reference_stats[col] = self.calculate_distribution_stats(reference_data[col])
    
    def detect_drift(self, recent_data: pd.DataFrame, columns: List[str] = None) -> Dict[str, float]:
        """
        Detect data drift by comparing recent data to reference distribution
        
        Returns:
            Dictionary with drift scores for each column (0-1, higher = more drift)
        """
        if columns is None:
            columns = recent_data.select_dtypes(include=[np.number]).columns.tolist()
        
        drift_scores = {}
        current_window = recent_data.tail(self.detection_window)
        
        for col in columns:
            if col not in self.reference_stats or col not in current_window.columns:
                drift_scores[col] = 0.0
                continue
            
            current_stats = self.calculate_distribution_stats(current_window[col])
            reference_stats = self.reference_stats[col]
            
            # Calculate normalized differences
            score = 0.0
            weight_sum = 0.0
            
            # Compare key statistics with appropriate weights
            stats_weights = {
                'mean': 0.3,
                'std': 0.25,
                'median': 0.2,
                'skewness': 0.15,
                'kurtosis': 0.1
            }
            
            for stat, weight in stats_weights.items():
                if reference_stats[stat] != 0:
                    normalized_diff = abs(current_stats[stat] - reference_stats[stat]) / abs(reference_stats[stat])
                    score += weight * min(normalized_diff, 1.0)  # Cap at 1.0
                    weight_sum += weight
            
            drift_scores[col] = score / weight_sum if weight_sum > 0 else 0.0
        
        return drift_scores


class DataQualityMonitor:
    """
    Main data quality monitoring system
    """
    
    def __init__(self, alert_threshold_warning: float = 0.3, alert_threshold_critical: float = 0.7):
        self.outlier_detector = StatisticalOutlierDetector()
        self.drift_detector = DataDriftDetector()
        self.alert_threshold_warning = alert_threshold_warning
        self.alert_threshold_critical = alert_threshold_critical
        self.alerts = []
        
        # Create alerts directory
        os.makedirs("alerts", exist_ok=True)
    
    def assess_data_quality(self, data: pd.DataFrame, columns: List[str] = None) -> DataQualityMetrics:
        """
        Comprehensive data quality assessment
        
        Returns:
            DataQualityMetrics object with quality scores
        """
        if columns is None:
            columns = config.FEATURE_COLUMNS
        
        # 1. Missing value analysis
        missing_rates = []
        for col in columns:
            if col in data.columns:
                missing_rate = data[col].isnull().sum() / len(data)
                missing_rates.append(missing_rate)
        avg_missing_rate = np.mean(missing_rates) if missing_rates else 0.0
        
        # 2. Outlier detection
        outlier_results = self.outlier_detector.detect_outliers(data, columns)
        outlier_rates = []
        for col in outlier_results.columns:
            outlier_rate = outlier_results[col].sum() / len(outlier_results)
            outlier_rates.append(outlier_rate)
        avg_outlier_rate = np.mean(outlier_rates) if outlier_rates else 0.0
        
        # 3. Data drift detection
        drift_scores = self.drift_detector.detect_drift(data, columns)
        avg_drift_score = np.mean(list(drift_scores.values())) if drift_scores else 0.0
        
        # 4. Data freshness (time since last update)
        if len(data) > 0 and hasattr(data.index, 'max'):
            last_timestamp = pd.to_datetime(data.index.max())
            data_freshness_hours = (datetime.now() - last_timestamp).total_seconds() / 3600
        else:
            data_freshness_hours = float('inf')
        
        # 5. Completeness score
        completeness_score = 1.0 - avg_missing_rate
        
        # 6. Overall quality score
        quality_components = [
            (1.0 - avg_missing_rate) * 0.3,      # Missing values (30%)
            (1.0 - min(avg_outlier_rate, 1.0)) * 0.25,  # Outliers (25%)
            (1.0 - min(avg_drift_score, 1.0)) * 0.25,   # Drift (25%)
            (1.0 if data_freshness_hours < 1 else max(0, 1.0 - data_freshness_hours/24)) * 0.2  # Freshness (20%)
        ]
        quality_score = sum(quality_components)
        
        return DataQualityMetrics(
            missing_value_rate=avg_missing_rate,
            outlier_rate=avg_outlier_rate,
            drift_score=avg_drift_score,
            data_freshness_hours=data_freshness_hours,
            completeness_score=completeness_score,
            quality_score=quality_score
        )
    
    def check_and_alert(self, metrics: DataQualityMetrics) -> List[QualityAlert]:
        """
        Check metrics against thresholds and generate alerts
        """
        new_alerts = []
        timestamp = datetime.now()
        
        # Define alert conditions
        alert_conditions = [
            {
                'metric': 'missing_value_rate',
                'value': metrics.missing_value_rate,
                'warning_threshold': 0.05,
                'critical_threshold': 0.15,
                'message_template': 'High missing value rate: {:.2%}'
            },
            {
                'metric': 'outlier_rate', 
                'value': metrics.outlier_rate,
                'warning_threshold': 0.10,
                'critical_threshold': 0.25,
                'message_template': 'High outlier rate detected: {:.2%}'
            },
            {
                'metric': 'drift_score',
                'value': metrics.drift_score,
                'warning_threshold': 0.30,
                'critical_threshold': 0.60,
                'message_template': 'Data drift detected: score {:.3f}'
            },
            {
                'metric': 'data_freshness_hours',
                'value': metrics.data_freshness_hours,
                'warning_threshold': 2.0,
                'critical_threshold': 24.0,
                'message_template': 'Data staleness: {:.1f} hours old'
            },
            {
                'metric': 'quality_score',
                'value': 1.0 - metrics.quality_score,  # Invert so higher = worse
                'warning_threshold': 0.30,
                'critical_threshold': 0.50,
                'message_template': 'Low overall data quality: {:.2%}'
            }
        ]
        
        for condition in alert_conditions:
            value = condition['value']
            
            if value >= condition['critical_threshold']:
                alert = QualityAlert(
                    timestamp=timestamp,
                    level=AlertLevel.CRITICAL,
                    metric=condition['metric'],
                    message=condition['message_template'].format(value),
                    value=value,
                    threshold=condition['critical_threshold']
                )
                new_alerts.append(alert)
                
            elif value >= condition['warning_threshold']:
                alert = QualityAlert(
                    timestamp=timestamp,
                    level=AlertLevel.WARNING,
                    metric=condition['metric'],
                    message=condition['message_template'].format(value),
                    value=value,
                    threshold=condition['warning_threshold']
                )
                new_alerts.append(alert)
        
        self.alerts.extend(new_alerts)
        
        # Save alerts to file
        if new_alerts:
            self.save_alerts()
            
        return new_alerts
    
    def save_alerts(self):
        """Save alerts to JSON file"""
        alert_data = []
        for alert in self.alerts[-50:]:  # Keep last 50 alerts
            alert_data.append({
                'timestamp': alert.timestamp.isoformat(),
                'level': alert.level.value,
                'metric': alert.metric,
                'message': alert.message,
                'value': alert.value,
                'threshold': alert.threshold
            })
        
        with open('alerts/data_quality_alerts.json', 'w') as f:
            json.dump(alert_data, f, indent=2)
    
    def monitor_data_quality(self, data: pd.DataFrame, update_reference: bool = True) -> Tuple[DataQualityMetrics, List[QualityAlert]]:
        """
        Main monitoring function - assess quality and generate alerts
        
        Args:
            data: Input data to monitor
            update_reference: Whether to update reference distributions for drift detection
            
        Returns:
            Tuple of (quality_metrics, new_alerts)
        """
        # Update reference distribution if requested
        if update_reference and len(data) >= self.drift_detector.reference_window:
            self.drift_detector.update_reference_distribution(data)
        
        # Assess current data quality
        metrics = self.assess_data_quality(data)
        
        # Generate alerts
        new_alerts = self.check_and_alert(metrics)
        
        return metrics, new_alerts
    
    def get_quality_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get a comprehensive quality summary for reporting
        """
        metrics, alerts = self.monitor_data_quality(data, update_reference=False)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'data_shape': data.shape,
            'metrics': {
                'missing_value_rate': round(metrics.missing_value_rate, 4),
                'outlier_rate': round(metrics.outlier_rate, 4),
                'drift_score': round(metrics.drift_score, 4),
                'data_freshness_hours': round(metrics.data_freshness_hours, 2),
                'completeness_score': round(metrics.completeness_score, 4),
                'quality_score': round(metrics.quality_score, 4)
            },
            'alerts': [
                {
                    'level': alert.level.value,
                    'metric': alert.metric,
                    'message': alert.message
                } for alert in alerts
            ],
            'status': 'critical' if any(a.level == AlertLevel.CRITICAL for a in alerts) 
                     else 'warning' if any(a.level == AlertLevel.WARNING for a in alerts)
                     else 'healthy'
        }


def main():
    """
    Test the data quality monitoring system
    """
    # Create sample data with some quality issues
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='5min')
    
    # Normal data
    normal_data = pd.DataFrame({
        'Open': np.random.normal(100, 5, 1000),
        'High': np.random.normal(102, 5, 1000),
        'Low': np.random.normal(98, 5, 1000),
        'Close': np.random.normal(100, 5, 1000),
        'Volume': np.random.normal(10000, 2000, 1000)
    }, index=dates)
    
    # Introduce quality issues
    # Missing values
    normal_data.loc[normal_data.index[100:110], 'Volume'] = np.nan
    
    # Outliers
    normal_data.loc[normal_data.index[200:205], 'Close'] = normal_data['Close'].iloc[200:205] * 3
    
    # Drift (shift in recent data)
    normal_data.loc[normal_data.index[800:], 'Close'] = normal_data.loc[normal_data.index[800:], 'Close'] + 20
    
    # Test the monitor
    monitor = DataQualityMonitor()
    
    print("Testing Data Quality Monitor...")
    print("=" * 50)
    
    # Get quality summary
    summary = monitor.get_quality_summary(normal_data)
    
    print(f"Data Quality Summary:")
    print(f"Status: {summary['status']}")
    print(f"Quality Score: {summary['metrics']['quality_score']:.3f}")
    print(f"Missing Value Rate: {summary['metrics']['missing_value_rate']:.3f}")
    print(f"Outlier Rate: {summary['metrics']['outlier_rate']:.3f}")
    print(f"Drift Score: {summary['metrics']['drift_score']:.3f}")
    
    if summary['alerts']:
        print(f"\nAlerts Generated ({len(summary['alerts'])}):")
        for alert in summary['alerts']:
            print(f"  {alert['level'].upper()}: {alert['message']}")
    else:
        print("\nNo alerts generated.")


if __name__ == "__main__":
    main()