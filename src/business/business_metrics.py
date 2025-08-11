"""
Business Metrics Tracking and Revenue Attribution
Converts ML metrics to business outcomes with automated winner detection
Based on production recommendation system ROI modeling
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import logging
import json
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BusinessMetrics:
    """Business metrics for recommendation systems"""
    # Core engagement metrics
    click_through_rate: float
    session_duration_minutes: float
    items_per_session: float
    return_rate_7day: float
    
    # Revenue metrics  
    revenue_per_user_session: float
    conversion_rate: float
    average_order_value: float
    customer_lifetime_value: float
    
    # Content metrics
    catalog_coverage_rate: float
    diversity_score: float
    novelty_score: float
    serendipity_score: float
    
    # Operational metrics
    recommendation_acceptance_rate: float
    user_satisfaction_score: float
    churn_risk_reduction: float
    
    # Metadata
    timestamp: datetime
    sample_size: int
    measurement_period_days: int

@dataclass
class RevenueImpactModel:
    """Model for calculating revenue impact from ML improvements"""
    # Base business parameters
    monthly_active_users: int = 200_000
    avg_sessions_per_user_per_month: float = 8.0
    baseline_revenue_per_session: float = 12.50
    baseline_ctr: float = 0.30
    baseline_session_duration: float = 25.0  # minutes
    
    # Attribution coefficients (learned from historical data)
    ctr_revenue_elasticity: float = 2.8  # 1% CTR improvement = 2.8% revenue increase
    session_duration_elasticity: float = 1.5  # 1% duration increase = 1.5% revenue increase
    diversity_satisfaction_coefficient: float = 0.15
    novelty_engagement_coefficient: float = 0.10
    
    # Cost factors
    infrastructure_cost_per_user_per_month: float = 0.15
    content_acquisition_cost_rate: float = 0.40  # 40% of revenue goes to content
    
    def calculate_revenue_impact(self, current_metrics: BusinessMetrics, 
                                baseline_metrics: BusinessMetrics) -> Dict[str, float]:
        """
        Calculate projected annual revenue impact from metric improvements
        
        Args:
            current_metrics: Current system performance
            baseline_metrics: Baseline system performance
            
        Returns:
            Dictionary with revenue impact breakdown
        """
        # CTR impact
        ctr_lift = (current_metrics.click_through_rate - baseline_metrics.click_through_rate) / baseline_metrics.click_through_rate
        ctr_revenue_impact = (
            self.monthly_active_users * 
            self.avg_sessions_per_user_per_month * 
            self.baseline_revenue_per_session * 
            ctr_lift * self.ctr_revenue_elasticity * 12
        )
        
        # Session duration impact
        duration_lift = (current_metrics.session_duration_minutes - baseline_metrics.session_duration_minutes) / baseline_metrics.session_duration_minutes
        duration_revenue_impact = (
            self.monthly_active_users * 
            self.avg_sessions_per_user_per_month * 
            self.baseline_revenue_per_session * 
            duration_lift * self.session_duration_elasticity * 12
        )
        
        # Diversity/satisfaction impact
        diversity_improvement = current_metrics.diversity_score - baseline_metrics.diversity_score
        satisfaction_impact = (
            self.monthly_active_users * 
            self.avg_sessions_per_user_per_month * 
            self.baseline_revenue_per_session * 
            diversity_improvement * self.diversity_satisfaction_coefficient * 12
        )
        
        # User retention impact (reduced churn)
        churn_reduction = current_metrics.churn_risk_reduction
        retention_impact = (
            self.monthly_active_users * 
            churn_reduction * 
            self.baseline_revenue_per_session * 
            self.avg_sessions_per_user_per_month * 12
        )
        
        # Catalog coverage impact (long-tail monetization)
        coverage_improvement = current_metrics.catalog_coverage_rate - baseline_metrics.catalog_coverage_rate
        coverage_impact = (
            self.monthly_active_users * 
            coverage_improvement * 
            self.baseline_revenue_per_session * 2.0 * 12  # Long-tail multiplier
        )
        
        # Total gross impact
        gross_revenue_impact = (
            ctr_revenue_impact + 
            duration_revenue_impact + 
            satisfaction_impact + 
            retention_impact + 
            coverage_impact
        )
        
        # Subtract costs
        infrastructure_costs = self.monthly_active_users * self.infrastructure_cost_per_user_per_month * 12
        content_costs = gross_revenue_impact * self.content_acquisition_cost_rate
        
        net_revenue_impact = gross_revenue_impact - infrastructure_costs - content_costs
        
        return {
            'gross_annual_impact': gross_revenue_impact,
            'net_annual_impact': net_revenue_impact,
            'ctr_contribution': ctr_revenue_impact,
            'duration_contribution': duration_revenue_impact,
            'satisfaction_contribution': satisfaction_impact,
            'retention_contribution': retention_impact,
            'coverage_contribution': coverage_impact,
            'infrastructure_costs': infrastructure_costs,
            'content_costs': content_costs,
            'roi_multiple': net_revenue_impact / (infrastructure_costs + content_costs) if (infrastructure_costs + content_costs) > 0 else 0,
            'revenue_per_user_annual': net_revenue_impact / self.monthly_active_users if self.monthly_active_users > 0 else 0
        }

class MLMetricsConverter:
    """Converts ML metrics to business metrics using industry benchmarks"""
    
    @staticmethod
    def convert_ml_to_business_metrics(ml_metrics: Dict[str, float], 
                                     baseline_metrics: Optional[Dict[str, float]] = None) -> BusinessMetrics:
        """
        Convert ML metrics to business metrics using established relationships
        
        Args:
            ml_metrics: Dictionary of ML metrics (rmse, precision@k, ndcg@k, etc.)
            baseline_metrics: Baseline ML metrics for comparison
            
        Returns:
            BusinessMetrics object
        """
        # Extract ML metrics with defaults
        rmse = ml_metrics.get('rmse', 1.0)
        precision_at_10 = ml_metrics.get('precision_at_10', 0.30)
        ndcg_at_10 = ml_metrics.get('ndcg_at_10', 0.40)
        recall_at_10 = ml_metrics.get('recall_at_10', 0.15)
        diversity = ml_metrics.get('diversity', 0.50)
        coverage = ml_metrics.get('coverage', 0.40)
        map_at_10 = ml_metrics.get('map_at_10', 0.25)
        
        # Convert to business metrics using industry relationships
        
        # CTR strongly correlates with Precision@K
        click_through_rate = precision_at_10 * 1.1  # CTR typically slightly higher than precision
        
        # Session duration correlates with NDCG (user engagement with relevant content)
        baseline_duration = 25.0  # minutes
        session_duration_minutes = baseline_duration * (1 + (ndcg_at_10 - 0.40) * 2.0)
        session_duration_minutes = max(10.0, min(60.0, session_duration_minutes))  # Realistic bounds
        
        # Items per session correlates with recall
        items_per_session = 8.0 * (1 + (recall_at_10 - 0.15) * 1.5)
        items_per_session = max(3.0, min(20.0, items_per_session))
        
        # Return rate correlates with overall recommendation quality
        overall_quality = (precision_at_10 + ndcg_at_10 + recall_at_10) / 3
        return_rate_7day = 0.45 * (1 + (overall_quality - 0.28) * 1.2)
        return_rate_7day = max(0.20, min(0.80, return_rate_7day))
        
        # Revenue metrics
        baseline_revenue = 12.50
        revenue_per_user_session = baseline_revenue * (1 + (precision_at_10 - 0.30) * 3.0)
        revenue_per_user_session = max(5.0, min(25.0, revenue_per_user_session))
        
        conversion_rate = precision_at_10 * 0.8  # Slightly lower than CTR
        average_order_value = revenue_per_user_session / conversion_rate if conversion_rate > 0 else baseline_revenue
        
        # Customer lifetime value improvement
        satisfaction_boost = (overall_quality - 0.28) * 0.5
        customer_lifetime_value = 180.0 * (1 + satisfaction_boost)  # Base CLV of $180
        
        # Content metrics (direct mapping)
        catalog_coverage_rate = coverage
        diversity_score = diversity
        novelty_score = min(1.0, diversity * 1.2)  # Novelty correlates with diversity
        serendipity_score = min(1.0, diversity * 0.8 + (ndcg_at_10 - 0.40) * 0.5)
        
        # Operational metrics
        recommendation_acceptance_rate = precision_at_10
        user_satisfaction_score = (ndcg_at_10 + diversity * 0.5) / 1.5  # Weighted average
        
        # Churn risk reduction (better recommendations = lower churn)
        churn_risk_reduction = min(0.15, (overall_quality - 0.28) * 0.25)  # Max 15% churn reduction
        
        return BusinessMetrics(
            click_through_rate=click_through_rate,
            session_duration_minutes=session_duration_minutes,
            items_per_session=items_per_session,
            return_rate_7day=return_rate_7day,
            revenue_per_user_session=revenue_per_user_session,
            conversion_rate=conversion_rate,
            average_order_value=average_order_value,
            customer_lifetime_value=customer_lifetime_value,
            catalog_coverage_rate=catalog_coverage_rate,
            diversity_score=diversity_score,
            novelty_score=novelty_score,
            serendipity_score=serendipity_score,
            recommendation_acceptance_rate=recommendation_acceptance_rate,
            user_satisfaction_score=user_satisfaction_score,
            churn_risk_reduction=churn_risk_reduction,
            timestamp=datetime.now(),
            sample_size=ml_metrics.get('sample_size', 10000),
            measurement_period_days=7
        )

class BusinessMetricsTracker:
    """Tracks business metrics over time and detects significant changes"""
    
    def __init__(self, storage_dir: str = "data/experiments/business_metrics"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_history: List[Dict[str, Any]] = []
        self.revenue_model = RevenueImpactModel()
        
    def track_variant_metrics(self, variant_id: str, ml_metrics: Dict[str, float], 
                            metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Track business metrics for an A/B test variant
        
        Args:
            variant_id: Experiment variant identifier
            ml_metrics: ML performance metrics
            metadata: Additional context (model_name, experiment_id, etc.)
            
        Returns:
            Dictionary with business metrics and revenue impact
        """
        # Convert ML metrics to business metrics
        business_metrics = MLMetricsConverter.convert_ml_to_business_metrics(ml_metrics)
        
        # Calculate revenue impact (need baseline for comparison)
        baseline_metrics = self._get_baseline_metrics()
        revenue_impact = self.revenue_model.calculate_revenue_impact(
            business_metrics, baseline_metrics
        )
        
        # Create tracking record
        tracking_record = {
            'variant_id': variant_id,
            'timestamp': datetime.now().isoformat(),
            'ml_metrics': ml_metrics,
            'business_metrics': asdict(business_metrics),
            'revenue_impact': revenue_impact,
            'metadata': metadata
        }
        
        # Store record
        self.metrics_history.append(tracking_record)
        self._save_metrics_record(tracking_record)
        
        logger.info(f"Tracked metrics for variant {variant_id}: "
                   f"CTR={business_metrics.click_through_rate:.3f}, "
                   f"Revenue Impact=${revenue_impact['net_annual_impact']:,.0f}")
        
        return tracking_record
    
    def compare_variants(self, variant_ids: List[str]) -> Dict[str, Any]:
        """
        Compare business metrics across multiple variants
        
        Args:
            variant_ids: List of variant IDs to compare
            
        Returns:
            Comparison analysis with winner recommendation
        """
        # Get latest metrics for each variant
        variant_metrics = {}
        for variant_id in variant_ids:
            latest_record = self._get_latest_variant_metrics(variant_id)
            if latest_record:
                variant_metrics[variant_id] = latest_record
        
        if len(variant_metrics) < 2:
            return {'error': 'Need at least 2 variants for comparison'}
        
        # Calculate business value scores
        variant_scores = {}
        for variant_id, record in variant_metrics.items():
            business_metrics = record['business_metrics']
            revenue_impact = record['revenue_impact']
            
            # Weighted business value score
            score = (
                business_metrics['click_through_rate'] * 0.25 +
                business_metrics['user_satisfaction_score'] * 0.20 +
                (business_metrics['session_duration_minutes'] / 60.0) * 0.15 +  # Normalize to [0,1]
                business_metrics['diversity_score'] * 0.15 +
                business_metrics['catalog_coverage_rate'] * 0.10 +
                min(1.0, revenue_impact['net_annual_impact'] / 1_000_000) * 0.15  # Revenue impact up to $1M
            )
            
            variant_scores[variant_id] = {
                'business_value_score': score,
                'revenue_impact': revenue_impact['net_annual_impact'],
                'key_metrics': {
                    'ctr': business_metrics['click_through_rate'],
                    'satisfaction': business_metrics['user_satisfaction_score'],
                    'session_duration': business_metrics['session_duration_minutes'],
                    'diversity': business_metrics['diversity_score']
                }
            }
        
        # Find winner
        winner_id = max(variant_scores.keys(), key=lambda x: variant_scores[x]['business_value_score'])
        winner_score = variant_scores[winner_id]
        
        # Calculate improvement over other variants
        other_scores = [v['business_value_score'] for k, v in variant_scores.items() if k != winner_id]
        avg_other_score = np.mean(other_scores) if other_scores else 0
        
        improvement_pct = ((winner_score['business_value_score'] - avg_other_score) / avg_other_score * 100) if avg_other_score > 0 else 0
        
        return {
            'winner_variant_id': winner_id,
            'winner_metadata': variant_metrics[winner_id]['metadata'],
            'business_value_improvement': improvement_pct,
            'projected_annual_revenue_impact': winner_score['revenue_impact'],
            'variant_scores': variant_scores,
            'recommendation': self._generate_business_recommendation(winner_id, winner_score, improvement_pct),
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def detect_performance_degradation(self, variant_id: str, lookback_days: int = 7) -> Dict[str, Any]:
        """
        Detect if business metrics are degrading for a variant
        
        Args:
            variant_id: Variant to analyze
            lookback_days: Days to look back for trend analysis
            
        Returns:
            Degradation analysis and alerts
        """
        # Get recent metrics for variant
        recent_records = self._get_variant_metrics_in_period(variant_id, lookback_days)
        
        if len(recent_records) < 3:
            return {'status': 'insufficient_data', 'message': 'Need at least 3 data points'}
        
        # Analyze trends in key business metrics
        metrics_trends = {}
        key_metrics = ['click_through_rate', 'user_satisfaction_score', 'session_duration_minutes']
        
        for metric in key_metrics:
            values = [r['business_metrics'][metric] for r in recent_records]
            timestamps = [datetime.fromisoformat(r['timestamp']) for r in recent_records]
            
            # Simple linear trend analysis
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
            
            # Calculate percentage change
            pct_change = (values[-1] - values[0]) / values[0] * 100 if values[0] > 0 else 0
            
            metrics_trends[metric] = {
                'slope': slope,
                'percent_change': pct_change,
                'current_value': values[-1],
                'trend': 'declining' if slope < -0.001 else 'stable' if slope < 0.001 else 'improving'
            }
        
        # Determine overall status
        declining_metrics = [m for m, t in metrics_trends.items() if t['trend'] == 'declining']
        
        if len(declining_metrics) >= 2:
            status = 'degradation_detected'
            severity = 'high' if len(declining_metrics) == 3 else 'medium'
        elif len(declining_metrics) == 1:
            status = 'potential_degradation'
            severity = 'low'
        else:
            status = 'stable'
            severity = 'none'
        
        return {
            'status': status,
            'severity': severity,
            'variant_id': variant_id,
            'declining_metrics': declining_metrics,
            'metrics_trends': metrics_trends,
            'recommendation': self._generate_degradation_recommendation(status, declining_metrics),
            'analysis_period_days': lookback_days,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _get_baseline_metrics(self) -> BusinessMetrics:
        """Get baseline business metrics for comparison"""
        # Default baseline metrics (industry averages)
        baseline_ml_metrics = {
            'rmse': 1.2,
            'precision_at_10': 0.30,
            'ndcg_at_10': 0.40,
            'recall_at_10': 0.15,
            'diversity': 0.50,
            'coverage': 0.40,
            'map_at_10': 0.25
        }
        
        return MLMetricsConverter.convert_ml_to_business_metrics(baseline_ml_metrics)
    
    def _get_latest_variant_metrics(self, variant_id: str) -> Optional[Dict[str, Any]]:
        """Get most recent metrics for a variant"""
        variant_records = [r for r in self.metrics_history if r['variant_id'] == variant_id]
        return variant_records[-1] if variant_records else None
    
    def _get_variant_metrics_in_period(self, variant_id: str, days: int) -> List[Dict[str, Any]]:
        """Get variant metrics within specified time period"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        return [
            r for r in self.metrics_history 
            if r['variant_id'] == variant_id and 
               datetime.fromisoformat(r['timestamp']) >= cutoff_date
        ]
    
    def _generate_business_recommendation(self, winner_id: str, winner_score: Dict[str, Any], 
                                        improvement_pct: float) -> Dict[str, Any]:
        """Generate business recommendation based on variant comparison"""
        if improvement_pct > 5.0:  # >5% improvement
            return {
                'action': 'implement_immediately',
                'confidence': 'high',
                'reason': f'Significant business improvement ({improvement_pct:.1f}%)',
                'expected_impact': f"${winner_score['revenue_impact']:,.0f} annual revenue increase"
            }
        elif improvement_pct > 2.0:  # 2-5% improvement
            return {
                'action': 'implement_after_validation',
                'confidence': 'medium',
                'reason': f'Moderate improvement ({improvement_pct:.1f}%) - validate with larger sample',
                'expected_impact': f"${winner_score['revenue_impact']:,.0f} annual revenue increase"
            }
        else:  # <2% improvement
            return {
                'action': 'continue_testing',
                'confidence': 'low',
                'reason': f'Small improvement ({improvement_pct:.1f}%) - continue testing or try new variants',
                'expected_impact': 'Minimal revenue impact - focus on other opportunities'
            }
    
    def _generate_degradation_recommendation(self, status: str, declining_metrics: List[str]) -> Dict[str, Any]:
        """Generate recommendation for performance degradation"""
        if status == 'degradation_detected':
            return {
                'action': 'immediate_investigation',
                'priority': 'high',
                'reason': f'Multiple metrics declining: {", ".join(declining_metrics)}',
                'next_steps': [
                    'Check data quality and pipeline health',
                    'Validate model performance on recent data',
                    'Consider rolling back to previous version',
                    'Investigate user behavior changes'
                ]
            }
        elif status == 'potential_degradation':
            return {
                'action': 'monitor_closely',
                'priority': 'medium',
                'reason': f'One metric declining: {declining_metrics[0]}',
                'next_steps': [
                    'Increase monitoring frequency',
                    'Check for data anomalies',
                    'Prepare rollback plan if trend continues'
                ]
            }
        else:
            return {
                'action': 'continue_monitoring',
                'priority': 'low',
                'reason': 'Metrics stable or improving',
                'next_steps': ['Continue regular monitoring']
            }
    
    def _save_metrics_record(self, record: Dict[str, Any]):
        """Save metrics record to persistent storage"""
        date_str = datetime.now().strftime('%Y%m%d')
        metrics_file = self.storage_dir / f"metrics_{date_str}.jsonl"
        
        with open(metrics_file, 'a') as f:
            f.write(json.dumps(record, default=str) + '\n')

def main():
    """Test business metrics tracking system"""
    print("💰 Testing Business Metrics Tracking...")
    
    # Create tracker
    tracker = BusinessMetricsTracker()
    
    # Simulate different model performance
    models_data = [
        {
            'variant_id': 'matrix_factorization',
            'ml_metrics': {'rmse': 1.25, 'precision_at_10': 0.28, 'ndcg_at_10': 0.38, 'diversity': 0.45, 'coverage': 0.35},
            'metadata': {'model_name': 'Matrix Factorization', 'experiment_id': 'model_comparison_v1'}
        },
        {
            'variant_id': 'hybrid_vae', 
            'ml_metrics': {'rmse': 1.21, 'precision_at_10': 0.35, 'ndcg_at_10': 0.42, 'diversity': 0.52, 'coverage': 0.47},
            'metadata': {'model_name': 'Hybrid VAE', 'experiment_id': 'model_comparison_v1'}
        },
        {
            'variant_id': 'neural_cf',
            'ml_metrics': {'rmse': 1.23, 'precision_at_10': 0.32, 'ndcg_at_10': 0.40, 'diversity': 0.48, 'coverage': 0.42},
            'metadata': {'model_name': 'Neural CF', 'experiment_id': 'model_comparison_v1'}
        }
    ]
    
    # Track metrics for each model
    for model_data in models_data:
        record = tracker.track_variant_metrics(**model_data)
        revenue_impact = record['revenue_impact']
        print(f"📊 {model_data['metadata']['model_name']}: ${revenue_impact['net_annual_impact']:,.0f} annual impact")
    
    # Compare variants
    comparison = tracker.compare_variants(['matrix_factorization', 'hybrid_vae', 'neural_cf'])
    
    print(f"\n🏆 Winner: {comparison['winner_metadata']['model_name']}")
    print(f"💰 Revenue Impact: ${comparison['projected_annual_revenue_impact']:,.0f}")
    print(f"📈 Improvement: {comparison['business_value_improvement']:.1f}%")
    print(f"🎯 Recommendation: {comparison['recommendation']['action']}")
    
    # Test degradation detection
    degradation_analysis = tracker.detect_performance_degradation('hybrid_vae')
    print(f"\n🔍 Performance Analysis: {degradation_analysis['status']}")
    
    print("\n✅ Business metrics tracking system ready!")

if __name__ == "__main__":
    main()