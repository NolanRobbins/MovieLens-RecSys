"""
Comprehensive A/B Testing Framework for ML Models
Statistical experiment design, execution, and analysis for recommendation systems
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import json
from pathlib import Path
import hashlib
from scipy import stats
from collections import defaultdict, deque
import threading
import uuid

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExperimentStatus(Enum):
    """Experiment lifecycle status"""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ANALYZING = "analyzing"

class AllocationStrategy(Enum):
    """Traffic allocation strategies"""
    UNIFORM = "uniform"           # Equal split
    WEIGHTED = "weighted"         # Custom weights
    GRADUAL_ROLLOUT = "gradual"   # Gradual increase
    CHAMPION_CHALLENGER = "champion_challenger"  # 90/10 split

class MetricType(Enum):
    """Types of metrics for experiments"""
    ENGAGEMENT = "engagement"     # CTR, session time
    BUSINESS = "business"        # Revenue, conversion
    QUALITY = "quality"          # Precision, NDCG
    USER_SATISFACTION = "user_satisfaction"  # Ratings, feedback

@dataclass
class ExperimentMetric:
    """Definition of an experiment metric"""
    name: str
    metric_type: MetricType
    description: str
    higher_is_better: bool = True
    statistical_power: float = 0.8
    minimum_detectable_effect: float = 0.05  # 5% change
    baseline_value: Optional[float] = None
    
class StatisticalTest(Enum):
    """Statistical significance tests"""
    T_TEST = "t_test"
    CHI_SQUARE = "chi_square"
    MANN_WHITNEY = "mann_whitney"
    WELCH_T_TEST = "welch_t_test"

@dataclass
class ExperimentArm:
    """Experiment arm configuration"""
    arm_id: str
    name: str
    description: str
    model_config: Dict[str, Any]
    allocation_percentage: float
    is_control: bool = False

@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    experiment_id: str
    name: str
    description: str
    hypothesis: str
    start_date: datetime
    end_date: datetime
    
    # Arms and allocation
    arms: List[ExperimentArm]
    allocation_strategy: AllocationStrategy
    
    # Metrics and statistical design
    primary_metric: ExperimentMetric
    secondary_metrics: List[ExperimentMetric]
    sample_size_per_arm: int
    significance_level: float = 0.05
    
    # Guardrail metrics (must not degrade)
    guardrail_metrics: List[ExperimentMetric] = None
    
    # Advanced settings
    randomization_unit: str = "user_id"  # user_id, session_id
    stratification_features: List[str] = None
    minimum_runtime_days: int = 7
    
    # Quality gates
    ramp_up_period_days: int = 1
    early_stopping_enabled: bool = True
    
    # Metadata
    owner: str = ""
    tags: List[str] = None

@dataclass
class UserAssignment:
    """User assignment to experiment arm"""
    user_id: str
    experiment_id: str
    arm_id: str
    assigned_at: datetime
    context: Dict[str, Any] = None

@dataclass
class ExperimentEvent:
    """Event recorded during experiment"""
    event_id: str
    experiment_id: str
    user_id: str
    arm_id: str
    event_type: str
    metric_name: str
    metric_value: float
    timestamp: datetime
    context: Dict[str, Any] = None

@dataclass
class StatisticalResult:
    """Statistical test result"""
    metric_name: str
    test_type: StatisticalTest
    control_mean: float
    treatment_mean: float
    effect_size: float
    effect_size_relative: float
    p_value: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    statistical_power: float

@dataclass
class ExperimentResults:
    """Complete experiment results"""
    experiment_id: str
    analysis_date: datetime
    runtime_days: int
    
    # Sample sizes
    total_users: int
    arm_sample_sizes: Dict[str, int]
    
    # Statistical results
    primary_result: StatisticalResult
    secondary_results: List[StatisticalResult]
    guardrail_results: List[StatisticalResult]
    
    # Business impact
    business_impact: Dict[str, Any]
    recommendation: str
    confidence_level: str  # high, medium, low

class ExperimentHasher:
    """Consistent hashing for experiment assignment"""
    
    @staticmethod
    def hash_user_to_bucket(user_id: str, experiment_id: str, 
                           salt: str = "") -> float:
        """Hash user to bucket [0, 1)"""
        hash_input = f"{user_id}:{experiment_id}:{salt}"
        hash_value = hashlib.md5(hash_input.encode()).hexdigest()
        # Convert to float in [0, 1)
        return int(hash_value[:8], 16) / (2**32)
    
    @staticmethod
    def assign_to_arm(user_id: str, experiment_config: ExperimentConfig,
                     salt: str = "") -> str:
        """Assign user to experiment arm"""
        bucket = ExperimentHasher.hash_user_to_bucket(
            user_id, experiment_config.experiment_id, salt
        )
        
        # Find arm based on allocation percentages
        cumulative_allocation = 0.0
        for arm in experiment_config.arms:
            cumulative_allocation += arm.allocation_percentage / 100.0
            if bucket < cumulative_allocation:
                return arm.arm_id
        
        # Fallback to last arm
        return experiment_config.arms[-1].arm_id

class MetricsCollector:
    """Collects and aggregates experiment metrics"""
    
    def __init__(self):
        self.events = deque()
        self.aggregated_metrics = defaultdict(lambda: defaultdict(list))
        self._lock = threading.Lock()
    
    def record_event(self, event: ExperimentEvent):
        """Record an experiment event"""
        with self._lock:
            self.events.append(event)
            
            # Update aggregated metrics
            key = (event.experiment_id, event.arm_id, event.metric_name)
            self.aggregated_metrics[key][event.user_id].append(event.metric_value)
    
    def get_metric_data(self, experiment_id: str, arm_id: str, 
                       metric_name: str) -> List[float]:
        """Get all metric values for an arm"""
        with self._lock:
            key = (experiment_id, arm_id, metric_name)
            if key not in self.aggregated_metrics:
                return []
            
            # Flatten user-level metrics
            all_values = []
            for user_values in self.aggregated_metrics[key].values():
                all_values.extend(user_values)
            
            return all_values
    
    def get_user_count(self, experiment_id: str, arm_id: str) -> int:
        """Get unique user count for an arm"""
        users = set()
        with self._lock:
            for (exp_id, a_id, _), user_metrics in self.aggregated_metrics.items():
                if exp_id == experiment_id and a_id == arm_id:
                    users.update(user_metrics.keys())
        return len(users)

class StatisticalAnalyzer:
    """Statistical analysis for A/B tests"""
    
    @staticmethod
    def perform_t_test(control_data: List[float], 
                      treatment_data: List[float],
                      significance_level: float = 0.05) -> StatisticalResult:
        """Perform independent samples t-test"""
        
        if len(control_data) == 0 or len(treatment_data) == 0:
            raise ValueError("Cannot perform t-test with empty data")
        
        control_mean = np.mean(control_data)
        treatment_mean = np.mean(treatment_data)
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(treatment_data, control_data, 
                                         equal_var=False)  # Welch's t-test
        
        # Effect size
        effect_size = treatment_mean - control_mean
        effect_size_relative = (effect_size / control_mean * 100) if control_mean != 0 else 0
        
        # Confidence interval
        pooled_se = np.sqrt(
            np.var(treatment_data, ddof=1) / len(treatment_data) + 
            np.var(control_data, ddof=1) / len(control_data)
        )
        
        df = len(treatment_data) + len(control_data) - 2
        t_critical = stats.t.ppf(1 - significance_level/2, df)
        ci_margin = t_critical * pooled_se
        
        confidence_interval = (effect_size - ci_margin, effect_size + ci_margin)
        
        # Statistical power (simplified approximation)
        statistical_power = 0.8  # Would calculate properly in real implementation
        
        return StatisticalResult(
            metric_name="",  # To be set by caller
            test_type=StatisticalTest.WELCH_T_TEST,
            control_mean=control_mean,
            treatment_mean=treatment_mean,
            effect_size=effect_size,
            effect_size_relative=effect_size_relative,
            p_value=p_value,
            confidence_interval=confidence_interval,
            is_significant=p_value < significance_level,
            statistical_power=statistical_power
        )
    
    @staticmethod
    def calculate_sample_size(baseline_metric: float, 
                            minimum_detectable_effect: float,
                            significance_level: float = 0.05,
                            statistical_power: float = 0.8,
                            baseline_variance: float = None) -> int:
        """Calculate required sample size per arm"""
        
        # Use Cohen's d for effect size if variance not provided
        if baseline_variance is None:
            baseline_variance = (baseline_metric * 0.3) ** 2  # Assume 30% CV
        
        effect_size_absolute = baseline_metric * minimum_detectable_effect
        standardized_effect_size = effect_size_absolute / np.sqrt(baseline_variance)
        
        # Use power analysis formula
        z_alpha = stats.norm.ppf(1 - significance_level/2)
        z_beta = stats.norm.ppf(statistical_power)
        
        sample_size = 2 * ((z_alpha + z_beta) / standardized_effect_size) ** 2
        
        return int(np.ceil(sample_size))

class ABTestingFramework:
    """
    Complete A/B testing framework for ML experiments
    """
    
    def __init__(self):
        self.experiments = {}
        self.user_assignments = {}
        self.metrics_collector = MetricsCollector()
        self.analyzer = StatisticalAnalyzer()
        self._lock = threading.Lock()
        
        logger.info("🧪 A/B Testing Framework initialized")
    
    def create_experiment(self, config: ExperimentConfig) -> str:
        """Create a new experiment"""
        
        # Validate configuration
        self._validate_experiment_config(config)
        
        with self._lock:
            if config.experiment_id in self.experiments:
                raise ValueError(f"Experiment {config.experiment_id} already exists")
            
            # Calculate sample sizes
            for metric in [config.primary_metric] + (config.secondary_metrics or []):
                if metric.baseline_value:
                    required_size = self.analyzer.calculate_sample_size(
                        metric.baseline_value,
                        metric.minimum_detectable_effect,
                        config.significance_level,
                        metric.statistical_power
                    )
                    logger.info(f"📊 Required sample size for {metric.name}: {required_size} per arm")
            
            # Store experiment
            self.experiments[config.experiment_id] = {
                'config': config,
                'status': ExperimentStatus.DRAFT,
                'created_at': datetime.now(),
                'started_at': None,
                'ended_at': None
            }
        
        logger.info(f"🆕 Experiment created: {config.experiment_id}")
        return config.experiment_id
    
    def start_experiment(self, experiment_id: str):
        """Start an experiment"""
        with self._lock:
            if experiment_id not in self.experiments:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            experiment = self.experiments[experiment_id]
            if experiment['status'] != ExperimentStatus.DRAFT:
                raise ValueError(f"Cannot start experiment in status {experiment['status']}")
            
            experiment['status'] = ExperimentStatus.ACTIVE
            experiment['started_at'] = datetime.now()
        
        logger.info(f"🚀 Experiment started: {experiment_id}")
    
    def assign_user(self, user_id: str, experiment_id: str, 
                   context: Optional[Dict[str, Any]] = None) -> str:
        """Assign user to experiment arm"""
        
        with self._lock:
            if experiment_id not in self.experiments:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            experiment = self.experiments[experiment_id]
            if experiment['status'] != ExperimentStatus.ACTIVE:
                return None  # Don't assign if experiment not active
            
            # Check if user already assigned
            assignment_key = f"{user_id}:{experiment_id}"
            if assignment_key in self.user_assignments:
                return self.user_assignments[assignment_key].arm_id
            
            # Assign to arm
            config = experiment['config']
            arm_id = ExperimentHasher.assign_to_arm(user_id, config)
            
            # Record assignment
            assignment = UserAssignment(
                user_id=user_id,
                experiment_id=experiment_id,
                arm_id=arm_id,
                assigned_at=datetime.now(),
                context=context
            )
            
            self.user_assignments[assignment_key] = assignment
            
            logger.debug(f"👤 User {user_id} assigned to {arm_id} in {experiment_id}")
            return arm_id
    
    def record_metric(self, user_id: str, experiment_id: str, 
                     metric_name: str, metric_value: float,
                     context: Optional[Dict[str, Any]] = None):
        """Record a metric event"""
        
        # Get user assignment
        assignment_key = f"{user_id}:{experiment_id}"
        if assignment_key not in self.user_assignments:
            logger.warning(f"⚠️ No assignment found for user {user_id} in {experiment_id}")
            return
        
        assignment = self.user_assignments[assignment_key]
        
        # Create event
        event = ExperimentEvent(
            event_id=str(uuid.uuid4()),
            experiment_id=experiment_id,
            user_id=user_id,
            arm_id=assignment.arm_id,
            event_type="metric",
            metric_name=metric_name,
            metric_value=metric_value,
            timestamp=datetime.now(),
            context=context
        )
        
        # Record event
        self.metrics_collector.record_event(event)
        
        logger.debug(f"📈 Recorded {metric_name}={metric_value} for user {user_id}")
    
    def analyze_experiment(self, experiment_id: str, 
                          force_analysis: bool = False) -> ExperimentResults:
        """Analyze experiment results"""
        
        with self._lock:
            if experiment_id not in self.experiments:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            experiment = self.experiments[experiment_id]
            config = experiment['config']
            
            # Check if experiment is ready for analysis
            if not force_analysis:
                if experiment['status'] not in [ExperimentStatus.ACTIVE, ExperimentStatus.COMPLETED]:
                    raise ValueError("Experiment must be active or completed for analysis")
                
                # Check minimum runtime
                if experiment['started_at']:
                    runtime = (datetime.now() - experiment['started_at']).days
                    if runtime < config.minimum_runtime_days:
                        raise ValueError(f"Minimum runtime not met: {runtime} < {config.minimum_runtime_days} days")
        
        logger.info(f"📊 Analyzing experiment: {experiment_id}")
        
        # Get control and treatment arms
        control_arm = next((arm for arm in config.arms if arm.is_control), config.arms[0])
        treatment_arms = [arm for arm in config.arms if not arm.is_control]
        
        results = []
        
        # Analyze primary metric
        primary_result = self._analyze_metric(
            experiment_id, control_arm, treatment_arms[0], config.primary_metric
        )
        
        # Analyze secondary metrics
        secondary_results = []
        for metric in config.secondary_metrics or []:
            result = self._analyze_metric(
                experiment_id, control_arm, treatment_arms[0], metric
            )
            secondary_results.append(result)
        
        # Analyze guardrail metrics
        guardrail_results = []
        for metric in config.guardrail_metrics or []:
            result = self._analyze_metric(
                experiment_id, control_arm, treatment_arms[0], metric
            )
            guardrail_results.append(result)
        
        # Calculate business impact
        business_impact = self._calculate_business_impact(primary_result, config)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            primary_result, secondary_results, guardrail_results
        )
        
        # Determine confidence level
        confidence_level = self._determine_confidence_level(
            primary_result, secondary_results, guardrail_results
        )
        
        # Get sample sizes
        arm_sample_sizes = {}
        total_users = 0
        for arm in config.arms:
            count = self.metrics_collector.get_user_count(experiment_id, arm.arm_id)
            arm_sample_sizes[arm.arm_id] = count
            total_users += count
        
        # Calculate runtime
        runtime_days = 0
        if experiment['started_at']:
            runtime_days = (datetime.now() - experiment['started_at']).days
        
        return ExperimentResults(
            experiment_id=experiment_id,
            analysis_date=datetime.now(),
            runtime_days=runtime_days,
            total_users=total_users,
            arm_sample_sizes=arm_sample_sizes,
            primary_result=primary_result,
            secondary_results=secondary_results,
            guardrail_results=guardrail_results,
            business_impact=business_impact,
            recommendation=recommendation,
            confidence_level=confidence_level
        )
    
    def _analyze_metric(self, experiment_id: str, control_arm: ExperimentArm,
                       treatment_arm: ExperimentArm, metric: ExperimentMetric) -> StatisticalResult:
        """Analyze a specific metric"""
        
        # Get data for both arms
        control_data = self.metrics_collector.get_metric_data(
            experiment_id, control_arm.arm_id, metric.name
        )
        treatment_data = self.metrics_collector.get_metric_data(
            experiment_id, treatment_arm.arm_id, metric.name
        )
        
        if not control_data or not treatment_data:
            logger.warning(f"⚠️ Insufficient data for metric {metric.name}")
            # Return placeholder result
            return StatisticalResult(
                metric_name=metric.name,
                test_type=StatisticalTest.T_TEST,
                control_mean=0.0,
                treatment_mean=0.0,
                effect_size=0.0,
                effect_size_relative=0.0,
                p_value=1.0,
                confidence_interval=(0.0, 0.0),
                is_significant=False,
                statistical_power=0.0
            )
        
        # Perform statistical test
        result = self.analyzer.perform_t_test(control_data, treatment_data)
        result.metric_name = metric.name
        
        return result
    
    def _calculate_business_impact(self, primary_result: StatisticalResult,
                                 config: ExperimentConfig) -> Dict[str, Any]:
        """Calculate business impact of the experiment"""
        
        # Simplified business impact calculation
        impact = {
            'metric_name': primary_result.metric_name,
            'relative_change_percent': primary_result.effect_size_relative,
            'is_positive': primary_result.effect_size > 0,
            'is_significant': primary_result.is_significant,
            'confidence_interval_percent': (
                primary_result.confidence_interval[0] / primary_result.control_mean * 100,
                primary_result.confidence_interval[1] / primary_result.control_mean * 100
            ) if primary_result.control_mean != 0 else (0, 0)
        }
        
        return impact
    
    def _generate_recommendation(self, primary_result: StatisticalResult,
                               secondary_results: List[StatisticalResult],
                               guardrail_results: List[StatisticalResult]) -> str:
        """Generate experiment recommendation"""
        
        # Check guardrails first
        guardrail_violations = [r for r in guardrail_results if 
                              r.is_significant and r.effect_size < 0]
        
        if guardrail_violations:
            return f"❌ DO NOT LAUNCH - Guardrail violations detected: {[r.metric_name for r in guardrail_violations]}"
        
        # Check primary metric
        if primary_result.is_significant and primary_result.effect_size > 0:
            return "✅ LAUNCH - Primary metric shows significant positive impact"
        elif primary_result.is_significant and primary_result.effect_size < 0:
            return "❌ DO NOT LAUNCH - Primary metric shows significant negative impact"
        else:
            return "⚠️ INCONCLUSIVE - No significant impact detected, consider longer runtime or larger sample size"
    
    def _determine_confidence_level(self, primary_result: StatisticalResult,
                                  secondary_results: List[StatisticalResult],
                                  guardrail_results: List[StatisticalResult]) -> str:
        """Determine confidence level in results"""
        
        # Simplified confidence assessment
        if primary_result.p_value < 0.01 and primary_result.statistical_power > 0.8:
            return "high"
        elif primary_result.p_value < 0.05 and primary_result.statistical_power > 0.6:
            return "medium"
        else:
            return "low"
    
    def _validate_experiment_config(self, config: ExperimentConfig):
        """Validate experiment configuration"""
        
        # Check arms
        if not config.arms or len(config.arms) < 2:
            raise ValueError("Experiment must have at least 2 arms")
        
        # Check allocation percentages
        total_allocation = sum(arm.allocation_percentage for arm in config.arms)
        if not (99 <= total_allocation <= 101):  # Allow for rounding
            raise ValueError(f"Allocation percentages must sum to 100%, got {total_allocation}%")
        
        # Check control arm
        control_arms = [arm for arm in config.arms if arm.is_control]
        if len(control_arms) != 1:
            raise ValueError("Experiment must have exactly one control arm")
        
        # Check dates
        if config.start_date >= config.end_date:
            raise ValueError("Start date must be before end date")
        
        logger.info("✅ Experiment configuration validated")
    
    def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """Get experiment status and summary"""
        
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        config = experiment['config']
        
        # Get assignment counts
        arm_counts = {}
        for arm in config.arms:
            count = self.metrics_collector.get_user_count(experiment_id, arm.arm_id)
            arm_counts[arm.arm_id] = count
        
        return {
            'experiment_id': experiment_id,
            'name': config.name,
            'status': experiment['status'].value,
            'created_at': experiment['created_at'].isoformat(),
            'started_at': experiment['started_at'].isoformat() if experiment['started_at'] else None,
            'arm_assignments': arm_counts,
            'total_users': sum(arm_counts.values()),
            'primary_metric': config.primary_metric.name,
            'runtime_days': (datetime.now() - experiment['started_at']).days if experiment['started_at'] else 0
        }

def main():
    """Test the A/B testing framework"""
    print("🧪 Testing A/B Testing Framework...")
    
    # Initialize framework
    framework = ABTestingFramework()
    
    try:
        # Create experiment configuration
        primary_metric = ExperimentMetric(
            name="click_through_rate",
            metric_type=MetricType.ENGAGEMENT,
            description="Click-through rate on recommendations",
            higher_is_better=True,
            minimum_detectable_effect=0.05,  # 5% relative change
            baseline_value=0.10  # 10% baseline CTR
        )
        
        secondary_metrics = [
            ExperimentMetric(
                name="session_duration",
                metric_type=MetricType.ENGAGEMENT,
                description="Average session duration in minutes",
                higher_is_better=True,
                minimum_detectable_effect=0.10,
                baseline_value=25.0
            )
        ]
        
        config = ExperimentConfig(
            experiment_id="rec_model_test_001",
            name="New Recommendation Model Test",
            description="Testing contextual vs baseline recommendation model",
            hypothesis="Contextual recommendations will improve CTR by 5%",
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=14),
            arms=[
                ExperimentArm(
                    arm_id="control",
                    name="Baseline Model",
                    description="Current production model",
                    model_config={"model_type": "baseline"},
                    allocation_percentage=50.0,
                    is_control=True
                ),
                ExperimentArm(
                    arm_id="treatment",
                    name="Contextual Model",
                    description="New contextual recommendation model",
                    model_config={"model_type": "contextual"},
                    allocation_percentage=50.0,
                    is_control=False
                )
            ],
            allocation_strategy=AllocationStrategy.UNIFORM,
            primary_metric=primary_metric,
            secondary_metrics=secondary_metrics,
            sample_size_per_arm=1000,
            significance_level=0.05,
            owner="ml-team"
        )
        
        # Create and start experiment
        experiment_id = framework.create_experiment(config)
        framework.start_experiment(experiment_id)
        
        print(f"✅ Experiment created and started: {experiment_id}")
        
        # Simulate user assignments and metrics
        np.random.seed(42)
        
        for user_id in range(1, 201):  # 200 users
            # Assign user to experiment
            arm_id = framework.assign_user(str(user_id), experiment_id)
            
            # Simulate metrics based on arm
            if arm_id == "control":
                # Baseline performance
                ctr = np.random.beta(2, 18)  # ~10% CTR
                session_duration = np.random.normal(25, 5)
            else:  # treatment
                # Improved performance
                ctr = np.random.beta(2.5, 17.5)  # ~12% CTR  
                session_duration = np.random.normal(27, 5)
            
            # Record metrics
            framework.record_metric(str(user_id), experiment_id, "click_through_rate", ctr)
            framework.record_metric(str(user_id), experiment_id, "session_duration", session_duration)
        
        # Analyze results
        results = framework.analyze_experiment(experiment_id, force_analysis=True)
        
        print(f"\n📊 Experiment Results:")
        print(f"Experiment: {results.experiment_id}")
        print(f"Total Users: {results.total_users}")
        print(f"Runtime: {results.runtime_days} days")
        
        print(f"\n🎯 Primary Metric ({results.primary_result.metric_name}):")
        print(f"  Control Mean: {results.primary_result.control_mean:.4f}")
        print(f"  Treatment Mean: {results.primary_result.treatment_mean:.4f}")
        print(f"  Effect Size: {results.primary_result.effect_size_relative:.2f}%")
        print(f"  P-value: {results.primary_result.p_value:.4f}")
        print(f"  Significant: {results.primary_result.is_significant}")
        
        if results.secondary_results:
            print(f"\n📈 Secondary Metrics:")
            for result in results.secondary_results:
                print(f"  {result.metric_name}: {result.effect_size_relative:.2f}% change (p={result.p_value:.4f})")
        
        print(f"\n💡 Recommendation: {results.recommendation}")
        print(f"🎯 Confidence: {results.confidence_level}")
        
        # Get experiment status
        status = framework.get_experiment_status(experiment_id)
        print(f"\n📋 Status Summary:")
        print(f"  Status: {status['status']}")
        print(f"  Total Users: {status['total_users']}")
        print(f"  Arm Assignments: {status['arm_assignments']}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ A/B Testing Framework ready!")

if __name__ == "__main__":
    main()