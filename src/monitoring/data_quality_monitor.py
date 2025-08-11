"""
Enhanced Data Quality Monitoring for MovieLens RecSys
Extends existing ETL monitoring with production-grade drift detection and alerting
Integrates with feature store and streaming data pipeline
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
import json
import pickle
from dataclasses import dataclass, asdict
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import existing monitoring
try:
    from ..data.etl_monitor import ETLMonitor, PipelineRun
except ImportError:
    # Fallback for development
    ETLMonitor = object
    PipelineRun = object

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataDriftMetrics:
    """Data drift detection metrics"""
    feature_name: str
    drift_score: float
    p_value: float
    drift_detected: bool
    reference_mean: float
    current_mean: float
    reference_std: float
    current_std: float
    sample_size: int
    detection_method: str
    timestamp: datetime

@dataclass
class DataQualityAlert:
    """Data quality alert"""
    alert_id: str
    alert_type: str  # 'drift', 'quality', 'volume', 'freshness'
    severity: str    # 'critical', 'warning', 'info'
    message: str
    affected_feature: Optional[str]
    metric_value: float
    threshold_value: float
    timestamp: datetime
    resolved: bool = False

@dataclass
class QualityMetrics:
    """Comprehensive data quality metrics"""
    timestamp: datetime
    total_interactions: int
    unique_users: int
    unique_items: int
    rating_distribution: Dict[str, float]
    missing_values_pct: float
    duplicate_rate: float
    data_freshness_hours: float
    schema_compliance: bool
    drift_metrics: List[DataDriftMetrics]
    quality_score: float
    alerts: List[DataQualityAlert]

class DataQualityMonitor:
    """
    Production-grade data quality monitoring system
    Extends existing ETL monitoring with drift detection and alerting
    """
    
    def __init__(self, config_path: str = "config/monitoring_config.json"):
        self.config = self._load_config(config_path)
        self.monitoring_dir = Path("data/processed/monitoring")
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)
        
        # Quality thresholds from GAMEPLAN.md
        self.quality_thresholds = {
            'missing_values_max_pct': 0.05,      # 5% max missing values
            'duplicate_rate_max': 0.02,          # 2% max duplicate rate
            'freshness_max_hours': 24,           # 24 hours max data age
            'drift_p_value_threshold': 0.05,     # Statistical significance
            'min_interactions_per_hour': 100,    # Minimum activity level
            'rating_range_min': 0.5,             # Valid rating range
            'rating_range_max': 5.0,
            'user_interaction_volume_change_threshold': 0.30,  # From GAMEPLAN.md
            'cold_start_ratio_change_threshold': 0.25          # From GAMEPLAN.md
        }
        
        # Reference data for drift detection
        self.reference_data = None
        self.reference_stats = None
        
        # Alert history
        self.alert_history = []
        
        logger.info("DataQualityMonitor initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load monitoring configuration"""
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                return json.load(f)
        
        # Default configuration
        return {
            'drift_detection_window_hours': 24,
            'reference_data_days': 7,
            'alert_cooldown_hours': 1,
            'quality_check_interval_minutes': 30
        }
    
    def establish_baseline(self, reference_data: pd.DataFrame):
        """Establish baseline statistics for drift detection"""
        logger.info(f"Establishing data quality baseline with {len(reference_data)} samples")
        
        self.reference_data = reference_data.copy()
        
        # Calculate reference statistics
        self.reference_stats = {
            'rating_mean': reference_data['rating'].mean(),
            'rating_std': reference_data['rating'].std(),
            'user_interaction_counts': reference_data.groupby('user_id').size().describe(),
            'item_interaction_counts': reference_data.groupby('movie_id').size().describe(),
            'rating_distribution': reference_data['rating'].value_counts(normalize=True).to_dict(),
            'unique_users': reference_data['user_id'].nunique(),
            'unique_items': reference_data['movie_id'].nunique(),
            'total_interactions': len(reference_data),
            'baseline_timestamp': datetime.now().isoformat()
        }
        
        # Save baseline
        baseline_path = self.monitoring_dir / "baseline_stats.json"
        with open(baseline_path, 'w') as f:
            json.dump(self.reference_stats, f, indent=2, default=str)
        
        logger.info(f"Baseline established and saved to {baseline_path}")
    
    def monitor_batch_quality(self, batch_data: pd.DataFrame) -> QualityMetrics:
        """
        Comprehensive quality monitoring for a data batch
        Returns detailed quality metrics and alerts
        """
        logger.info(f"Monitoring quality for batch with {len(batch_data)} interactions")
        
        timestamp = datetime.now()
        alerts = []
        
        # Basic quality checks
        total_interactions = len(batch_data)
        unique_users = batch_data['user_id'].nunique()
        unique_items = batch_data['movie_id'].nunique()
        
        # Missing values check
        missing_values_pct = batch_data.isnull().sum().sum() / (len(batch_data) * len(batch_data.columns))
        if missing_values_pct > self.quality_thresholds['missing_values_max_pct']:
            alerts.append(DataQualityAlert(
                alert_id=f"missing_values_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                alert_type='quality',
                severity='warning',
                message=f"High missing values rate: {missing_values_pct:.3f}",
                affected_feature='global',
                metric_value=missing_values_pct,
                threshold_value=self.quality_thresholds['missing_values_max_pct'],
                timestamp=timestamp
            ))
        
        # Duplicate check
        duplicate_rate = batch_data.duplicated(['user_id', 'movie_id']).sum() / len(batch_data)
        if duplicate_rate > self.quality_thresholds['duplicate_rate_max']:
            alerts.append(DataQualityAlert(
                alert_id=f"duplicates_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                alert_type='quality',
                severity='critical',
                message=f"High duplicate rate: {duplicate_rate:.3f}",
                affected_feature='user_movie_pairs',
                metric_value=duplicate_rate,
                threshold_value=self.quality_thresholds['duplicate_rate_max'],
                timestamp=timestamp
            ))
        
        # Rating distribution
        rating_dist = batch_data['rating'].value_counts(normalize=True).to_dict()
        
        # Schema compliance
        required_columns = ['user_id', 'movie_id', 'rating']
        schema_compliance = all(col in batch_data.columns for col in required_columns)
        
        # Data type checks
        if not pd.api.types.is_numeric_dtype(batch_data['rating']):
            schema_compliance = False
            alerts.append(DataQualityAlert(
                alert_id=f"schema_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                alert_type='quality',
                severity='critical',
                message="Rating column is not numeric",
                affected_feature='rating',
                metric_value=0.0,
                threshold_value=1.0,
                timestamp=timestamp
            ))
        
        # Rating range validation
        invalid_ratings = batch_data[
            (batch_data['rating'] < self.quality_thresholds['rating_range_min']) |
            (batch_data['rating'] > self.quality_thresholds['rating_range_max'])
        ]
        
        if len(invalid_ratings) > 0:
            alerts.append(DataQualityAlert(
                alert_id=f"rating_range_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                alert_type='quality',
                severity='warning',
                message=f"Found {len(invalid_ratings)} invalid ratings",
                affected_feature='rating',
                metric_value=len(invalid_ratings) / len(batch_data),
                threshold_value=0.0,
                timestamp=timestamp
            ))
        
        # Drift detection
        drift_metrics = []
        if self.reference_stats:
            drift_metrics = self._detect_data_drift(batch_data)
            
            # Add drift alerts
            for drift_metric in drift_metrics:
                if drift_metric.drift_detected:
                    alerts.append(DataQualityAlert(
                        alert_id=f"drift_{drift_metric.feature_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                        alert_type='drift',
                        severity='warning',
                        message=f"Data drift detected in {drift_metric.feature_name}",
                        affected_feature=drift_metric.feature_name,
                        metric_value=drift_metric.drift_score,
                        threshold_value=self.quality_thresholds['drift_p_value_threshold'],
                        timestamp=timestamp
                    ))
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(
            missing_values_pct, duplicate_rate, schema_compliance, len(alerts)
        )
        
        # Create quality metrics
        metrics = QualityMetrics(
            timestamp=timestamp,
            total_interactions=total_interactions,
            unique_users=unique_users,
            unique_items=unique_items,
            rating_distribution=rating_dist,
            missing_values_pct=missing_values_pct,
            duplicate_rate=duplicate_rate,
            data_freshness_hours=0.0,  # Would calculate from timestamps
            schema_compliance=schema_compliance,
            drift_metrics=drift_metrics,
            quality_score=quality_score,
            alerts=alerts
        )
        
        # Store alerts
        self.alert_history.extend(alerts)
        
        # Log summary
        logger.info(f"Quality check completed: Score={quality_score:.3f}, Alerts={len(alerts)}")
        
        return metrics
    
    def _detect_data_drift(self, current_data: pd.DataFrame) -> List[DataDriftMetrics]:
        """Detect statistical drift between current and reference data"""
        drift_metrics = []
        
        if not self.reference_stats:
            logger.warning("No reference stats available for drift detection")
            return drift_metrics
        
        # Rating distribution drift
        current_rating_mean = current_data['rating'].mean()
        current_rating_std = current_data['rating'].std()
        
        # Kolmogorov-Smirnov test for distribution drift
        reference_ratings = self.reference_data['rating'].values
        current_ratings = current_data['rating'].values
        
        ks_stat, p_value = stats.ks_2samp(reference_ratings, current_ratings)
        drift_detected = p_value < self.quality_thresholds['drift_p_value_threshold']
        
        drift_metrics.append(DataDriftMetrics(
            feature_name='rating_distribution',
            drift_score=ks_stat,
            p_value=p_value,
            drift_detected=drift_detected,
            reference_mean=self.reference_stats['rating_mean'],
            current_mean=current_rating_mean,
            reference_std=self.reference_stats['rating_std'],
            current_std=current_rating_std,
            sample_size=len(current_data),
            detection_method='kolmogorov_smirnov',
            timestamp=datetime.now()
        ))
        
        # User interaction volume drift
        current_user_interactions = current_data.groupby('user_id').size()
        reference_user_interactions = self.reference_data.groupby('user_id').size()
        
        # Compare interaction volume distributions
        if len(current_user_interactions) > 10 and len(reference_user_interactions) > 10:
            volume_ks_stat, volume_p_value = stats.ks_2samp(
                reference_user_interactions.values,
                current_user_interactions.values
            )
            
            volume_drift_detected = volume_p_value < self.quality_thresholds['drift_p_value_threshold']
            
            drift_metrics.append(DataDriftMetrics(
                feature_name='user_interaction_volume',
                drift_score=volume_ks_stat,
                p_value=volume_p_value,
                drift_detected=volume_drift_detected,
                reference_mean=reference_user_interactions.mean(),
                current_mean=current_user_interactions.mean(),
                reference_std=reference_user_interactions.std(),
                current_std=current_user_interactions.std(),
                sample_size=len(current_user_interactions),
                detection_method='kolmogorov_smirnov',
                timestamp=datetime.now()
            ))
        
        return drift_metrics
    
    def _calculate_quality_score(self, missing_values_pct: float, duplicate_rate: float,
                                schema_compliance: bool, alert_count: int) -> float:
        """Calculate overall quality score (0-1)"""
        score = 1.0
        
        # Penalize missing values
        score -= min(0.3, missing_values_pct * 6)  # Max 30% penalty
        
        # Penalize duplicates
        score -= min(0.2, duplicate_rate * 10)  # Max 20% penalty
        
        # Penalize schema issues
        if not schema_compliance:
            score -= 0.3
        
        # Penalize alerts
        score -= min(0.2, alert_count * 0.05)  # 5% per alert, max 20%
        
        return max(0.0, score)
    
    def check_retraining_triggers(self, recent_metrics: List[QualityMetrics]) -> Dict[str, Any]:
        """
        Check if data quality issues should trigger model retraining
        Based on thresholds from GAMEPLAN.md
        """
        if not recent_metrics:
            return {'should_retrain': False, 'reason': 'No metrics available'}
        
        latest_metrics = recent_metrics[-1]
        
        # Check for critical data quality issues
        critical_alerts = [alert for alert in latest_metrics.alerts if alert.severity == 'critical']
        if critical_alerts:
            return {
                'should_retrain': True,
                'reason': f'Critical data quality issues: {[alert.message for alert in critical_alerts]}',
                'trigger_type': 'data_quality'
            }
        
        # Check drift patterns over time (if we have enough history)
        if len(recent_metrics) >= 7:  # Week of data
            drift_trend = self._analyze_drift_trend(recent_metrics[-7:])
            if drift_trend['significant_drift']:
                return {
                    'should_retrain': True,
                    'reason': f'Significant drift trend detected: {drift_trend["details"]}',
                    'trigger_type': 'data_drift'
                }
        
        # Check quality score degradation
        if latest_metrics.quality_score < 0.7:  # Quality score below 70%
            return {
                'should_retrain': True,
                'reason': f'Quality score degraded to {latest_metrics.quality_score:.3f}',
                'trigger_type': 'quality_degradation'
            }
        
        return {'should_retrain': False, 'reason': 'Data quality within acceptable thresholds'}
    
    def _analyze_drift_trend(self, metrics_window: List[QualityMetrics]) -> Dict[str, Any]:
        """Analyze drift trends over a time window"""
        drift_scores = []
        
        for metrics in metrics_window:
            avg_drift_score = np.mean([dm.drift_score for dm in metrics.drift_metrics])
            drift_scores.append(avg_drift_score)
        
        if len(drift_scores) < 3:
            return {'significant_drift': False, 'details': 'Insufficient data'}
        
        # Check if drift is consistently increasing
        increasing_trend = all(drift_scores[i] <= drift_scores[i+1] for i in range(len(drift_scores)-1))
        high_recent_drift = np.mean(drift_scores[-3:]) > 0.1  # High recent drift
        
        if increasing_trend and high_recent_drift:
            return {
                'significant_drift': True,
                'details': f'Increasing drift trend, recent avg: {np.mean(drift_scores[-3:]):.3f}'
            }
        
        return {'significant_drift': False, 'details': 'Drift within normal bounds'}
    
    def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard"""
        # Load recent monitoring history
        monitoring_files = list(self.monitoring_dir.glob("quality_metrics_*.json"))
        recent_metrics = []
        
        for file_path in sorted(monitoring_files)[-24:]:  # Last 24 files
            try:
                with open(file_path, 'r') as f:
                    metrics_data = json.load(f)
                    recent_metrics.append(metrics_data)
            except Exception as e:
                logger.error(f"Error loading metrics file {file_path}: {e}")
        
        # Calculate summary statistics
        if recent_metrics:
            quality_scores = [m.get('quality_score', 0) for m in recent_metrics]
            alert_counts = [len(m.get('alerts', [])) for m in recent_metrics]
            
            dashboard_data = {
                'current_quality_score': quality_scores[-1] if quality_scores else 0,
                'avg_quality_score_24h': np.mean(quality_scores) if quality_scores else 0,
                'total_alerts_24h': sum(alert_counts),
                'critical_alerts_24h': sum(1 for m in recent_metrics 
                                         for alert in m.get('alerts', []) 
                                         if alert.get('severity') == 'critical'),
                'quality_trend': 'improving' if len(quality_scores) > 1 and quality_scores[-1] > quality_scores[-2] else 'stable',
                'last_updated': datetime.now().isoformat(),
                'monitoring_active': True
            }
        else:
            dashboard_data = {
                'current_quality_score': 0,
                'avg_quality_score_24h': 0,
                'total_alerts_24h': 0,
                'critical_alerts_24h': 0,
                'quality_trend': 'unknown',
                'last_updated': datetime.now().isoformat(),
                'monitoring_active': False
            }
        
        return dashboard_data
    
    def save_metrics(self, metrics: QualityMetrics):
        """Save metrics to persistent storage"""
        filename = f"quality_metrics_{metrics.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.monitoring_dir / filename
        
        # Convert to serializable format
        metrics_dict = asdict(metrics)
        
        with open(filepath, 'w') as f:
            json.dump(metrics_dict, f, indent=2, default=str)
        
        logger.info(f"Quality metrics saved to {filepath}")

def main():
    """Test the data quality monitor"""
    monitor = DataQualityMonitor()
    
    # Create sample data
    reference_data = pd.DataFrame({
        'user_id': np.random.randint(1, 1000, 5000),
        'movie_id': np.random.randint(1, 2000, 5000),
        'rating': np.random.choice([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], 5000)
    })
    
    # Establish baseline
    monitor.establish_baseline(reference_data)
    
    # Create current batch with some drift
    current_data = pd.DataFrame({
        'user_id': np.random.randint(1, 1000, 1000),
        'movie_id': np.random.randint(1, 2000, 1000),
        'rating': np.random.choice([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], 1000)  # Higher ratings
    })
    
    # Monitor quality
    metrics = monitor.monitor_batch_quality(current_data)
    print(f"Quality metrics: Score={metrics.quality_score:.3f}, Alerts={len(metrics.alerts)}")
    
    # Check retraining triggers
    retraining_check = monitor.check_retraining_triggers([metrics])
    print(f"Retraining needed: {retraining_check}")
    
    # Get dashboard data
    dashboard_data = monitor.get_monitoring_dashboard_data()
    print(f"Dashboard data: {dashboard_data}")

if __name__ == "__main__":
    main()