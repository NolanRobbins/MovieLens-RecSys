"""
A/B Testing Infrastructure for MovieLens RecSys
Deterministic user bucketing with statistical validation
Integrates with existing business logic and model variants
"""

import numpy as np
import pandas as pd
import hashlib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import scipy.stats as stats
from collections import defaultdict

# Import business logic
from .business_logic_system import BusinessRulesEngine, UserProfile

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExperimentStatus(Enum):
    """Experiment status enumeration"""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

@dataclass
class ExperimentVariant:
    """A/B test variant configuration"""
    variant_id: str
    variant_name: str
    model_name: str
    model_path: Optional[str]
    traffic_allocation: float  # 0.0 to 1.0
    business_rules_config: Dict[str, Any]
    description: str
    is_control: bool = False

@dataclass
class ExperimentConfig:
    """A/B test experiment configuration"""
    experiment_id: str
    experiment_name: str
    description: str
    variants: List[ExperimentVariant]
    start_date: datetime
    end_date: Optional[datetime]
    target_sample_size: int
    success_metrics: List[str]
    minimum_effect_size: float
    statistical_power: float
    significance_level: float
    status: ExperimentStatus
    created_by: str
    tags: List[str]

@dataclass
class UserInteraction:
    """User interaction event for A/B test tracking"""
    user_id: int
    experiment_id: str
    variant_id: str
    timestamp: datetime
    interaction_type: str  # 'view', 'click', 'rating', 'purchase'
    movie_id: Optional[int]
    rating: Optional[float]
    session_id: str
    context: Dict[str, Any]

@dataclass
class VariantMetrics:
    """Metrics for a single A/B test variant"""
    variant_id: str
    sample_size: int
    click_through_rate: float
    avg_rating: float
    session_duration: float
    recommendations_accepted: int
    total_recommendations: int
    unique_users: int
    conversion_rate: float
    revenue_per_user: float
    diversity_score: float
    coverage_score: float

class UserBucketing:
    """Deterministic user bucketing system"""
    
    @staticmethod
    def get_user_bucket(user_id: int, experiment_id: str, num_buckets: int = 100) -> int:
        """
        Get deterministic bucket for user in experiment
        
        Args:
            user_id: User identifier
            experiment_id: Experiment identifier
            num_buckets: Number of buckets (default 100 for percentage allocation)
            
        Returns:
            Bucket number (0 to num_buckets-1)
        """
        # Create deterministic hash from user_id and experiment_id
        hash_input = f"{user_id}_{experiment_id}".encode('utf-8')
        hash_value = hashlib.md5(hash_input).hexdigest()
        
        # Convert to integer and mod by number of buckets
        bucket = int(hash_value, 16) % num_buckets
        
        return bucket
    
    @staticmethod
    def assign_user_to_variant(user_id: int, experiment_config: ExperimentConfig) -> Optional[ExperimentVariant]:
        """
        Assign user to experiment variant based on traffic allocation
        
        Args:
            user_id: User identifier
            experiment_config: Experiment configuration
            
        Returns:
            Assigned variant or None if user not in experiment
        """
        if experiment_config.status != ExperimentStatus.ACTIVE:
            return None
        
        # Get user bucket (0-99)
        bucket = UserBucketing.get_user_bucket(user_id, experiment_config.experiment_id, 100)
        
        # Assign based on traffic allocation
        cumulative_allocation = 0
        
        for variant in experiment_config.variants:
            cumulative_allocation += variant.traffic_allocation
            
            if bucket < (cumulative_allocation * 100):
                return variant
        
        # User not in any variant (outside experiment)
        return None

class StatisticalTesting:
    """Statistical significance testing for A/B experiments"""
    
    @staticmethod
    def chi_square_test(control_conversions: int, control_sample_size: int,
                       treatment_conversions: int, treatment_sample_size: int) -> Dict[str, float]:
        """
        Chi-square test for conversion rates
        
        Returns:
            Dictionary with test statistics
        """
        # Create contingency table
        converted = [control_conversions, treatment_conversions]
        not_converted = [
            control_sample_size - control_conversions,
            treatment_sample_size - treatment_conversions
        ]
        
        contingency_table = [converted, not_converted]
        
        # Perform chi-square test
        chi2_stat, p_value = stats.chi2_contingency(contingency_table)[:2]
        
        # Calculate effect size (Cramér's V)
        n = control_sample_size + treatment_sample_size
        cramers_v = np.sqrt(chi2_stat / (n * min(1, 1)))
        
        return {
            'chi2_statistic': float(chi2_stat),
            'p_value': float(p_value),
            'cramers_v': float(cramers_v),
            'significant': p_value < 0.05
        }
    
    @staticmethod
    def t_test(control_values: List[float], treatment_values: List[float]) -> Dict[str, float]:
        """
        Two-sample t-test for continuous metrics
        
        Returns:
            Dictionary with test statistics
        """
        # Perform Welch's t-test (unequal variances)
        t_stat, p_value = stats.ttest_ind(treatment_values, control_values, equal_var=False)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(control_values) - 1) * np.var(control_values, ddof=1) +
             (len(treatment_values) - 1) * np.var(treatment_values, ddof=1)) /
            (len(control_values) + len(treatment_values) - 2)
        )
        
        cohens_d = (np.mean(treatment_values) - np.mean(control_values)) / pooled_std if pooled_std > 0 else 0
        
        return {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'control_mean': float(np.mean(control_values)),
            'treatment_mean': float(np.mean(treatment_values)),
            'significant': p_value < 0.05
        }
    
    @staticmethod
    def sequential_testing(control_data: List[float], treatment_data: List[float],
                         alpha: float = 0.05, power: float = 0.8) -> Dict[str, Any]:
        """
        Sequential testing with early stopping rules
        
        Returns:
            Testing recommendation and statistics
        """
        if len(control_data) < 30 or len(treatment_data) < 30:
            return {
                'recommendation': 'continue',
                'reason': 'Insufficient sample size for reliable testing',
                'sample_sizes': {'control': len(control_data), 'treatment': len(treatment_data)}
            }
        
        # Perform t-test
        t_test_result = StatisticalTesting.t_test(control_data, treatment_data)
        
        # Check for significance
        if t_test_result['significant']:
            effect_size = abs(t_test_result['cohens_d'])
            
            if effect_size >= 0.2:  # Minimum meaningful effect size
                return {
                    'recommendation': 'stop_significant',
                    'reason': f"Statistically significant result (p={t_test_result['p_value']:.4f})",
                    'winning_variant': 'treatment' if t_test_result['treatment_mean'] > t_test_result['control_mean'] else 'control',
                    'effect_size': effect_size,
                    'statistics': t_test_result
                }
            else:
                return {
                    'recommendation': 'stop_no_effect',
                    'reason': f"Significant but small effect size ({effect_size:.3f})",
                    'statistics': t_test_result
                }
        
        # Check if we have enough power to detect minimum effect
        max_sample_size = max(len(control_data), len(treatment_data))
        if max_sample_size > 10000:  # Large sample, likely no meaningful difference
            return {
                'recommendation': 'stop_no_effect',
                'reason': 'Large sample size with no significant difference',
                'statistics': t_test_result
            }
        
        return {
            'recommendation': 'continue',
            'reason': 'Not yet significant, continue collecting data',
            'statistics': t_test_result
        }

class ABTestingManager:
    """Main A/B testing manager"""
    
    def __init__(self, storage_dir: str = "data/experiments/ab_tests"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory storage for active experiments
        self.active_experiments: Dict[str, ExperimentConfig] = {}
        self.user_interactions: List[UserInteraction] = []
        
        # Load existing experiments
        self._load_experiments()
    
    def create_experiment(self, experiment_config: ExperimentConfig) -> bool:
        """Create new A/B test experiment"""
        try:
            # Validate experiment configuration
            validation_result = self._validate_experiment_config(experiment_config)
            if not validation_result['valid']:
                logger.error(f"Invalid experiment config: {validation_result['errors']}")
                return False
            
            # Store experiment
            self.active_experiments[experiment_config.experiment_id] = experiment_config
            
            # Save to disk
            self._save_experiment(experiment_config)
            
            logger.info(f"Created experiment: {experiment_config.experiment_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating experiment: {e}")
            return False
    
    def get_user_variant(self, user_id: int, experiment_id: str) -> Optional[ExperimentVariant]:
        """Get variant assignment for user in experiment"""
        if experiment_id not in self.active_experiments:
            return None
        
        experiment_config = self.active_experiments[experiment_id]
        return UserBucketing.assign_user_to_variant(user_id, experiment_config)
    
    def track_interaction(self, interaction: UserInteraction):
        """Track user interaction event"""
        self.user_interactions.append(interaction)
        
        # Persist interaction (in production, this would go to a database)
        self._save_interaction(interaction)
    
    def get_experiment_metrics(self, experiment_id: str) -> Dict[str, VariantMetrics]:
        """Calculate metrics for all variants in experiment"""
        if experiment_id not in self.active_experiments:
            return {}
        
        experiment_config = self.active_experiments[experiment_id]
        variant_metrics = {}
        
        # Group interactions by variant
        interactions_by_variant = defaultdict(list)
        for interaction in self.user_interactions:
            if interaction.experiment_id == experiment_id:
                interactions_by_variant[interaction.variant_id].append(interaction)
        
        # Calculate metrics for each variant
        for variant in experiment_config.variants:
            variant_interactions = interactions_by_variant[variant.variant_id]
            metrics = self._calculate_variant_metrics(variant.variant_id, variant_interactions)
            variant_metrics[variant.variant_id] = metrics
        
        return variant_metrics
    
    def analyze_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Perform statistical analysis of experiment"""
        if experiment_id not in self.active_experiments:
            return {'error': 'Experiment not found'}
        
        experiment_config = self.active_experiments[experiment_id]
        variant_metrics = self.get_experiment_metrics(experiment_id)
        
        if len(variant_metrics) < 2:
            return {'error': 'Need at least 2 variants for analysis'}
        
        # Find control variant
        control_variant_id = None
        for variant in experiment_config.variants:
            if variant.is_control:
                control_variant_id = variant.variant_id
                break
        
        if not control_variant_id or control_variant_id not in variant_metrics:
            return {'error': 'Control variant not found'}
        
        control_metrics = variant_metrics[control_variant_id]
        analysis_results = {}
        
        # Compare each treatment variant to control
        for variant_id, treatment_metrics in variant_metrics.items():
            if variant_id == control_variant_id:
                continue
            
            # CTR comparison
            ctr_test = StatisticalTesting.chi_square_test(
                int(control_metrics.recommendations_accepted),
                control_metrics.total_recommendations,
                int(treatment_metrics.recommendations_accepted),
                treatment_metrics.total_recommendations
            )
            
            # Rating comparison (if available)
            # This would need actual interaction data - using mock values
            control_ratings = [control_metrics.avg_rating] * control_metrics.sample_size
            treatment_ratings = [treatment_metrics.avg_rating] * treatment_metrics.sample_size
            
            rating_test = StatisticalTesting.t_test(control_ratings, treatment_ratings)
            
            analysis_results[variant_id] = {
                'variant_name': next(v.variant_name for v in experiment_config.variants if v.variant_id == variant_id),
                'sample_size': treatment_metrics.sample_size,
                'ctr_test': ctr_test,
                'rating_test': rating_test,
                'metrics_comparison': {
                    'control_ctr': control_metrics.click_through_rate,
                    'treatment_ctr': treatment_metrics.click_through_rate,
                    'ctr_lift': ((treatment_metrics.click_through_rate - control_metrics.click_through_rate) / 
                               control_metrics.click_through_rate * 100) if control_metrics.click_through_rate > 0 else 0,
                    'control_rating': control_metrics.avg_rating,
                    'treatment_rating': treatment_metrics.avg_rating,
                    'rating_lift': treatment_metrics.avg_rating - control_metrics.avg_rating
                }
            }
        
        return {
            'experiment_id': experiment_id,
            'experiment_name': experiment_config.experiment_name,
            'analysis_timestamp': datetime.now().isoformat(),
            'control_variant': control_variant_id,
            'variant_analyses': analysis_results,
            'overall_recommendation': self._get_experiment_recommendation(analysis_results)
        }
    
    def _validate_experiment_config(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Validate experiment configuration"""
        errors = []
        
        # Check traffic allocation sums to <= 1.0
        total_allocation = sum(v.traffic_allocation for v in config.variants)
        if total_allocation > 1.0:
            errors.append(f"Traffic allocation exceeds 100%: {total_allocation}")
        
        # Check for control variant
        control_variants = [v for v in config.variants if v.is_control]
        if len(control_variants) != 1:
            errors.append("Exactly one control variant required")
        
        # Check experiment dates
        if config.end_date and config.start_date >= config.end_date:
            errors.append("End date must be after start date")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def _calculate_variant_metrics(self, variant_id: str, interactions: List[UserInteraction]) -> VariantMetrics:
        """Calculate metrics for a single variant"""
        if not interactions:
            return VariantMetrics(
                variant_id=variant_id,
                sample_size=0,
                click_through_rate=0.0,
                avg_rating=0.0,
                session_duration=0.0,
                recommendations_accepted=0,
                total_recommendations=0,
                unique_users=0,
                conversion_rate=0.0,
                revenue_per_user=0.0,
                diversity_score=0.0,
                coverage_score=0.0
            )
        
        # Calculate basic metrics
        unique_users = len(set(i.user_id for i in interactions))
        total_recommendations = len([i for i in interactions if i.interaction_type == 'view'])
        recommendations_accepted = len([i for i in interactions if i.interaction_type == 'click'])
        
        # Mock calculations (in production, these would be computed from real data)
        return VariantMetrics(
            variant_id=variant_id,
            sample_size=len(interactions),
            click_through_rate=recommendations_accepted / total_recommendations if total_recommendations > 0 else 0,
            avg_rating=np.mean([i.rating for i in interactions if i.rating is not None] or [3.5]),
            session_duration=np.mean([i.context.get('session_duration', 300) for i in interactions]),
            recommendations_accepted=recommendations_accepted,
            total_recommendations=total_recommendations,
            unique_users=unique_users,
            conversion_rate=recommendations_accepted / unique_users if unique_users > 0 else 0,
            revenue_per_user=np.random.uniform(5, 15),  # Mock revenue
            diversity_score=np.random.uniform(0.4, 0.8),  # Mock diversity
            coverage_score=np.random.uniform(0.3, 0.6)  # Mock coverage
        )
    
    def _get_experiment_recommendation(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Get overall recommendation for experiment"""
        significant_results = []
        
        for variant_id, analysis in analysis_results.items():
            if analysis['ctr_test']['significant'] or analysis['rating_test']['significant']:
                significant_results.append({
                    'variant_id': variant_id,
                    'variant_name': analysis['variant_name'],
                    'ctr_lift': analysis['metrics_comparison']['ctr_lift'],
                    'rating_lift': analysis['metrics_comparison']['rating_lift']
                })
        
        if not significant_results:
            return {
                'recommendation': 'continue_or_stop',
                'reason': 'No statistically significant differences found',
                'action': 'Consider increasing sample size or stopping experiment'
            }
        
        # Find best performing variant
        best_variant = max(significant_results, 
                          key=lambda x: x['ctr_lift'] + x['rating_lift'])
        
        return {
            'recommendation': 'implement_winner',
            'winning_variant': best_variant,
            'reason': f"Significant improvement in key metrics",
            'action': f"Implement {best_variant['variant_name']} as new default"
        }
    
    def _save_experiment(self, experiment_config: ExperimentConfig):
        """Save experiment configuration to disk"""
        experiment_path = self.storage_dir / f"{experiment_config.experiment_id}.json"
        
        with open(experiment_path, 'w') as f:
            json.dump(asdict(experiment_config), f, indent=2, default=str)
    
    def _save_interaction(self, interaction: UserInteraction):
        """Save interaction to disk"""
        interaction_path = self.storage_dir / "interactions" / f"{interaction.timestamp.strftime('%Y%m%d')}.jsonl"
        interaction_path.parent.mkdir(exist_ok=True)
        
        with open(interaction_path, 'a') as f:
            f.write(json.dumps(asdict(interaction), default=str) + '\n')
    
    def _load_experiments(self):
        """Load existing experiments from disk"""
        if not self.storage_dir.exists():
            return
        
        for experiment_file in self.storage_dir.glob("*.json"):
            try:
                with open(experiment_file, 'r') as f:
                    experiment_data = json.load(f)
                
                # Reconstruct experiment config (simplified)
                # In production, this would properly deserialize all objects
                experiment_id = experiment_data['experiment_id']
                self.active_experiments[experiment_id] = experiment_data
                
            except Exception as e:
                logger.error(f"Error loading experiment {experiment_file}: {e}")

def main():
    """Test A/B testing infrastructure"""
    print("🧪 Testing A/B Testing Infrastructure...")
    
    # Create A/B testing manager
    ab_manager = ABTestingManager()
    
    # Create test experiment
    experiment_config = ExperimentConfig(
        experiment_id="model_comparison_v1",
        experiment_name="Hybrid VAE vs Matrix Factorization",
        description="Compare Hybrid VAE against MF baseline",
        variants=[
            ExperimentVariant(
                variant_id="control_mf",
                variant_name="Matrix Factorization",
                model_name="matrix_factorization",
                model_path="data/models/mf_baseline.pt",
                traffic_allocation=0.5,
                business_rules_config={'diversity_weight': 0.3},
                description="Matrix Factorization baseline",
                is_control=True
            ),
            ExperimentVariant(
                variant_id="treatment_vae",
                variant_name="Hybrid VAE",
                model_name="hybrid_vae",
                model_path="data/models/hybrid_vae_best.pt",
                traffic_allocation=0.5,
                business_rules_config={'diversity_weight': 0.4},
                description="Hybrid VAE with improved ranking",
                is_control=False
            )
        ],
        start_date=datetime.now(),
        end_date=datetime.now() + timedelta(days=14),
        target_sample_size=10000,
        success_metrics=['ctr', 'avg_rating', 'session_duration'],
        minimum_effect_size=0.05,
        statistical_power=0.8,
        significance_level=0.05,
        status=ExperimentStatus.ACTIVE,
        created_by="ml_engineer",
        tags=['model_comparison', 'baseline_test']
    )
    
    # Create experiment
    success = ab_manager.create_experiment(experiment_config)
    print(f"✅ Experiment created: {success}")
    
    # Test user bucketing
    test_users = [1, 100, 500, 1000, 5000]
    print("\n🪣 User Bucketing Test:")
    
    for user_id in test_users:
        variant = ab_manager.get_user_variant(user_id, "model_comparison_v1")
        if variant:
            print(f"   User {user_id}: {variant.variant_name}")
        else:
            print(f"   User {user_id}: Not in experiment")
    
    # Test statistical testing
    print("\n📊 Statistical Testing:")
    control_data = np.random.normal(3.5, 0.5, 1000)
    treatment_data = np.random.normal(3.7, 0.5, 1000)  # Slight improvement
    
    t_test_result = StatisticalTesting.t_test(control_data.tolist(), treatment_data.tolist())
    print(f"   T-test p-value: {t_test_result['p_value']:.4f}")
    print(f"   Effect size (Cohen's d): {t_test_result['cohens_d']:.3f}")
    print(f"   Significant: {t_test_result['significant']}")
    
    print("\n🎉 A/B Testing infrastructure ready!")

if __name__ == "__main__":
    main()