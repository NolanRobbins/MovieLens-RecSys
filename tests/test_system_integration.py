"""
Comprehensive System Integration Tests
Tests the complete MovieLens recommendation system end-to-end
"""

import pytest
import sys
import os
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import asyncio
import time
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import system components
from data.etl_pipeline import ETLPipeline
from data.feature_store import FeatureStore
from business.business_metrics import BusinessMetricsTracker
from monitoring.data_quality_monitor import DataQualityMonitor
from validation.model_validator import ModelValidator
from deployment.model_versioning import ModelVersionManager
from personalization.contextual_recommender import ContextualRecommendationSystem
from serving.realtime_server import RealTimeRecommendationServer, RecommendationRequest, ServingMode
from experimentation.ab_testing_framework import ABTestingFramework, ExperimentConfig, ExperimentArm, ExperimentMetric, MetricType, AllocationStrategy
from coldstart.cold_start_handler import ColdStartHandler, UserProfile, ColdStartStrategy
from optimization.performance_optimizer import PerformanceOptimizer
from config.config_manager import ConfigManager

class TestSystemIntegration:
    """Integration tests for the complete recommendation system"""
    
    @pytest.fixture(scope="class")
    def sample_data(self):
        """Create sample MovieLens-like data for testing"""
        np.random.seed(42)
        
        # Create sample ratings data
        n_users = 100
        n_movies = 200
        n_ratings = 2000
        
        ratings_data = {
            'userId': np.random.choice(range(1, n_users + 1), n_ratings),
            'movieId': np.random.choice(range(1, n_movies + 1), n_ratings),
            'rating': np.random.choice([1, 2, 3, 4, 5], n_ratings, p=[0.1, 0.1, 0.2, 0.3, 0.3]),
            'timestamp': np.random.randint(1000000000, 1700000000, n_ratings)
        }
        ratings_df = pd.DataFrame(ratings_data)
        
        # Create sample movies data
        genres_list = ['Action', 'Comedy', 'Drama', 'Romance', 'Sci-Fi', 'Horror', 'Thriller', 'Animation']
        movies_data = {
            'movieId': range(1, n_movies + 1),
            'title': [f'Movie {i}' for i in range(1, n_movies + 1)],
            'genres': ['|'.join(np.random.choice(genres_list, np.random.randint(1, 4))) for _ in range(n_movies)],
            'year': np.random.choice(range(1990, 2024), n_movies)
        }
        movies_df = pd.DataFrame(movies_data)
        
        # Create sample users data
        users_data = {
            'userId': range(1, n_users + 1),
            'age': np.random.choice([18, 25, 35, 45, 55, 65], n_users),
            'gender': np.random.choice(['M', 'F'], n_users),
            'occupation': np.random.choice(['student', 'engineer', 'teacher', 'doctor'], n_users)
        }
        users_df = pd.DataFrame(users_data)
        
        return {
            'ratings': ratings_df,
            'movies': movies_df, 
            'users': users_df
        }
    
    @pytest.fixture(scope="class")
    def temp_dir(self):
        """Create temporary directory for test files"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield tmp_dir
    
    def test_configuration_management(self, temp_dir):
        """Test configuration management system"""
        print("\n🔧 Testing Configuration Management...")
        
        # Test config loading and validation
        config_manager = ConfigManager()
        
        # Test configuration access
        assert config_manager.get('environment.name') is not None
        assert config_manager.get('models.default_model') is not None
        assert config_manager.get('business_thresholds.max_rmse') is not None
        
        # Test configuration validation
        validation_result = config_manager.validate()
        assert validation_result.is_valid or len(validation_result.errors) == 0
        
        print("✅ Configuration management working")
    
    def test_etl_pipeline(self, sample_data, temp_dir):
        """Test complete ETL pipeline"""
        print("\n🔄 Testing ETL Pipeline...")
        
        # Initialize ETL pipeline
        etl = ETLPipeline(output_dir=temp_dir)
        
        # Test data loading and processing
        processed_data = etl.process_ratings_data(sample_data['ratings'])
        assert not processed_data.empty
        assert 'rating_zscore' in processed_data.columns
        
        # Test feature engineering
        features_data = etl.engineer_features(processed_data, sample_data['movies'])
        assert not features_data.empty
        assert 'user_avg_rating' in features_data.columns
        
        print("✅ ETL pipeline working")
    
    def test_feature_store(self, sample_data, temp_dir):
        """Test feature store functionality"""
        print("\n🗄️ Testing Feature Store...")
        
        # Initialize feature store
        feature_store = FeatureStore(db_path=os.path.join(temp_dir, 'test_features.db'))
        
        # Test feature storage and retrieval
        test_features = {
            'user_1': {'avg_rating': 3.5, 'total_ratings': 25},
            'user_2': {'avg_rating': 4.0, 'total_ratings': 40}
        }
        
        for feature_key, features in test_features.items():
            feature_store.store_features('user_features', feature_key, features)
        
        # Test retrieval
        retrieved_features = feature_store.get_features('user_features', 'user_1')
        assert retrieved_features is not None
        assert retrieved_features['avg_rating'] == 3.5
        
        print("✅ Feature store working")
    
    def test_business_metrics(self, sample_data):
        """Test business metrics tracking"""
        print("\n📊 Testing Business Metrics...")
        
        # Initialize metrics tracker
        metrics_tracker = BusinessMetricsTracker()
        
        # Test metric computation
        business_metrics = metrics_tracker.compute_recommendation_metrics(
            sample_data['ratings'], 
            sample_data['movies']
        )
        
        assert 'user_engagement' in business_metrics
        assert 'content_diversity' in business_metrics
        assert business_metrics['total_users'] > 0
        assert business_metrics['total_items'] > 0
        
        print("✅ Business metrics working")
    
    def test_data_quality_monitoring(self, sample_data):
        """Test data quality monitoring"""
        print("\n🔍 Testing Data Quality Monitoring...")
        
        # Initialize monitor
        monitor = DataQualityMonitor()
        
        # Test data quality assessment
        quality_results = monitor.assess_data_quality(sample_data['ratings'])
        
        assert quality_results.overall_quality > 0
        assert len(quality_results.quality_checks) > 0
        assert quality_results.data_profile is not None
        
        print("✅ Data quality monitoring working")
    
    def test_model_validation(self, temp_dir):
        """Test model validation system"""
        print("\n✅ Testing Model Validation...")
        
        # Initialize validator
        validator = ModelValidator()
        
        # Create a mock model file for testing
        mock_model_data = {
            'model_config': {'type': 'test_model'},
            'training_results': {
                'best_val_loss': 0.82,
                'epochs_trained': 25,
                'training_time': 3600,
                'param_count': 50000,
                'data_quality_score': 0.85
            }
        }
        
        model_path = os.path.join(temp_dir, 'test_model.pt')
        torch.save(mock_model_data, model_path)
        
        # Test model validation
        validation_report = validator.validate_model(model_path, 'test_model')
        
        assert validation_report.overall_status is not None
        assert validation_report.passed_checks >= 0
        assert validation_report.failed_checks >= 0
        assert validation_report.deploy_approved is not None
        
        print("✅ Model validation working")
    
    def test_model_versioning(self, temp_dir):
        """Test model versioning and rollback"""
        print("\n📦 Testing Model Versioning...")
        
        # Initialize version manager
        version_manager = ModelVersionManager(
            models_dir=os.path.join(temp_dir, 'models'),
            versions_dir=os.path.join(temp_dir, 'versions')
        )
        
        # Create mock model
        mock_model_path = os.path.join(temp_dir, 'mock_model.pt')
        torch.save({'test': 'data'}, mock_model_path)
        
        # Test model deployment
        version_id = version_manager.deploy_model(
            model_path=mock_model_path,
            model_name='test_model',
            validation_results={'rmse': 0.8},
            deployment_notes='Test deployment'
        )
        
        assert version_id is not None
        assert version_id.startswith('test_model_v')
        
        # Test version listing
        versions = version_manager.list_versions('test_model')
        assert len(versions) == 1
        assert versions[0].version_id == version_id
        
        # Test health check
        health_report = version_manager.health_check()
        assert 'total_versions' in health_report
        assert 'active_models' in health_report
        
        print("✅ Model versioning working")
    
    def test_cold_start_handler(self, sample_data):
        """Test cold start recommendation handling"""
        print("\n🆕 Testing Cold Start Handler...")
        
        # Initialize cold start handler
        cold_start_handler = ColdStartHandler()
        cold_start_handler.fit(sample_data['ratings'], sample_data['movies'], sample_data['users'])
        
        # Test new user scenario
        new_user_profile = UserProfile(
            user_id=999,
            preferred_genres=['Action', 'Sci-Fi'],
            signup_date=datetime.now()
        )
        
        # Test cold start recommendations
        recommendations = cold_start_handler.get_cold_start_recommendations(
            user_id=999,
            user_profile=new_user_profile,
            n_recommendations=5,
            strategy=ColdStartStrategy.HYBRID
        )
        
        assert len(recommendations) > 0
        assert all(hasattr(rec, 'item_id') for rec in recommendations)
        assert all(hasattr(rec, 'score') for rec in recommendations)
        assert all(hasattr(rec, 'strategy') for rec in recommendations)
        
        # Test onboarding flow
        onboarding_data = cold_start_handler.handle_onboarding_flow(999)
        assert 'onboarding_items' in onboarding_data
        assert len(onboarding_data['onboarding_items']) > 0
        
        print("✅ Cold start handler working")
    
    def test_ab_testing_framework(self):
        """Test A/B testing framework"""
        print("\n🧪 Testing A/B Testing Framework...")
        
        # Initialize framework
        ab_framework = ABTestingFramework()
        
        # Create test experiment
        primary_metric = ExperimentMetric(
            name="click_through_rate",
            metric_type=MetricType.ENGAGEMENT,
            description="Click-through rate on recommendations",
            higher_is_better=True,
            minimum_detectable_effect=0.05,
            baseline_value=0.10
        )
        
        config = ExperimentConfig(
            experiment_id="test_experiment_001",
            name="Test Experiment",
            description="Testing the A/B framework",
            hypothesis="Treatment will improve CTR by 5%",
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=7),
            arms=[
                ExperimentArm(
                    arm_id="control",
                    name="Control",
                    description="Current model",
                    model_config={"type": "baseline"},
                    allocation_percentage=50.0,
                    is_control=True
                ),
                ExperimentArm(
                    arm_id="treatment", 
                    name="Treatment",
                    description="New model",
                    model_config={"type": "new_model"},
                    allocation_percentage=50.0,
                    is_control=False
                )
            ],
            allocation_strategy=AllocationStrategy.UNIFORM,
            primary_metric=primary_metric,
            sample_size_per_arm=100,
            owner="test-team"
        )
        
        # Test experiment creation and management
        experiment_id = ab_framework.create_experiment(config)
        assert experiment_id == "test_experiment_001"
        
        ab_framework.start_experiment(experiment_id)
        
        # Test user assignment
        user_assignments = []
        for user_id in range(1, 21):
            arm_id = ab_framework.assign_user(str(user_id), experiment_id)
            assert arm_id in ['control', 'treatment']
            user_assignments.append(arm_id)
        
        # Should have roughly balanced assignment
        control_count = user_assignments.count('control')
        treatment_count = user_assignments.count('treatment')
        assert 5 <= control_count <= 15  # Allow for randomness
        assert 5 <= treatment_count <= 15
        
        print("✅ A/B testing framework working")
    
    def test_performance_optimizer(self):
        """Test performance optimization system"""
        print("\n⚡ Testing Performance Optimizer...")
        
        # Initialize optimizer
        optimizer = PerformanceOptimizer({
            'cache_memory_mb': 50,
            'max_batch_size': 16
        })
        
        optimizer.start()
        
        try:
            # Test caching functionality
            def test_computation(x):
                time.sleep(0.001)  # Small delay
                return x * 2
            
            # Test cache miss and hit
            result1 = optimizer.cached_prediction(['test', 123], lambda: test_computation(123))
            result2 = optimizer.cached_prediction(['test', 123], lambda: test_computation(123))
            
            assert result1 == result2 == 246
            
            # Test performance summary
            summary = optimizer.get_performance_summary()
            assert 'cache_stats' in summary
            assert 'batch_stats' in summary
            assert summary['cache_stats']['hit_rate'] > 0  # Should have some cache hits
            
            print("✅ Performance optimizer working")
            
        finally:
            optimizer.stop()
    
    @pytest.mark.asyncio
    async def test_realtime_server(self):
        """Test real-time recommendation server"""
        print("\n⚡ Testing Real-time Server...")
        
        # Initialize server
        server = RealTimeRecommendationServer(cache_size=100, cache_ttl=300)
        
        # Initialize with mock data
        user_ids = list(range(1, 51))
        item_ids = list(range(1, 101))
        popular_items = list(range(1, 21))
        
        server.initialize_cache(user_ids, item_ids, popular_items)
        
        # Test recommendation request
        request = RecommendationRequest(
            user_id=42,
            request_id="test-001",
            timestamp=datetime.now(),
            n_recommendations=5,
            serving_mode=ServingMode.REAL_TIME
        )
        
        # Get recommendations
        response = await server.get_recommendations(request)
        
        assert response.request_id == "test-001"
        assert response.user_id == 42
        assert len(response.recommendations) == 5
        assert response.response_time_ms >= 0
        assert response.model_version is not None
        
        # Test health check
        health = server.health_check()
        assert health['status'] in ['healthy', 'degraded', 'unhealthy']
        assert 'models_loaded' in health
        
        print("✅ Real-time server working")
    
    def test_end_to_end_workflow(self, sample_data, temp_dir):
        """Test complete end-to-end workflow"""
        print("\n🔄 Testing End-to-End Workflow...")
        
        # 1. ETL Pipeline
        etl = ETLPipeline(output_dir=temp_dir)
        processed_data = etl.process_ratings_data(sample_data['ratings'])
        features_data = etl.engineer_features(processed_data, sample_data['movies'])
        
        # 2. Data Quality Monitoring
        monitor = DataQualityMonitor()
        quality_results = monitor.assess_data_quality(features_data)
        assert quality_results.overall_quality > 0.5  # Reasonable quality threshold
        
        # 3. Business Metrics
        metrics_tracker = BusinessMetricsTracker()
        business_metrics = metrics_tracker.compute_recommendation_metrics(
            features_data, sample_data['movies']
        )
        assert business_metrics['total_users'] > 0
        
        # 4. Cold Start Setup
        cold_start_handler = ColdStartHandler()
        cold_start_handler.fit(features_data, sample_data['movies'], sample_data['users'])
        
        # 5. Model Validation (with mock model)
        validator = ModelValidator()
        mock_model_data = {
            'training_results': {'best_val_loss': 0.75, 'epochs_trained': 30}
        }
        model_path = os.path.join(temp_dir, 'e2e_model.pt')
        torch.save(mock_model_data, model_path)
        
        validation_report = validator.validate_model(model_path, 'e2e_model')
        
        # 6. Model Versioning
        version_manager = ModelVersionManager(
            models_dir=os.path.join(temp_dir, 'models'),
            versions_dir=os.path.join(temp_dir, 'versions')
        )
        
        if validation_report.deploy_approved:
            version_id = version_manager.deploy_model(
                model_path=model_path,
                model_name='e2e_model',
                validation_results=validation_report.business_impact_summary,
                deployment_notes='End-to-end test deployment'
            )
            assert version_id is not None
        
        print("✅ End-to-end workflow completed successfully")
    
    def test_system_health_checks(self, sample_data, temp_dir):
        """Test system-wide health checks"""
        print("\n🏥 Testing System Health Checks...")
        
        health_summary = {}
        
        # Test each component's health
        components = [
            ('ETL Pipeline', lambda: ETLPipeline(output_dir=temp_dir)),
            ('Feature Store', lambda: FeatureStore(db_path=os.path.join(temp_dir, 'health_test.db'))),
            ('Business Metrics', lambda: BusinessMetricsTracker()),
            ('Data Quality Monitor', lambda: DataQualityMonitor()),
            ('Model Validator', lambda: ModelValidator()),
            ('Cold Start Handler', lambda: ColdStartHandler()),
            ('A/B Testing Framework', lambda: ABTestingFramework())
        ]
        
        for component_name, component_factory in components:
            try:
                component = component_factory()
                health_summary[component_name] = 'healthy'
            except Exception as e:
                health_summary[component_name] = f'unhealthy: {str(e)}'
        
        # Check that most components are healthy
        healthy_count = sum(1 for status in health_summary.values() if status == 'healthy')
        total_count = len(health_summary)
        
        print(f"Component Health Summary:")
        for component, status in health_summary.items():
            print(f"  {component}: {status}")
        
        assert healthy_count >= total_count * 0.8  # At least 80% healthy
        
        print(f"✅ System health check passed: {healthy_count}/{total_count} components healthy")

def run_integration_tests():
    """Run all integration tests"""
    print("🧪 Running MovieLens System Integration Tests...\n")
    
    # Run pytest
    test_file = __file__
    exit_code = pytest.main([
        test_file,
        "-v",  # Verbose output
        "-s",  # Don't capture stdout
        "--tb=short",  # Short traceback format
        "-x"   # Stop on first failure
    ])
    
    if exit_code == 0:
        print("\n🎉 All integration tests passed!")
        print("✅ MovieLens recommendation system is ready for production!")
    else:
        print("\n❌ Some integration tests failed")
        print("🔍 Please review the test output above")
    
    return exit_code == 0

if __name__ == "__main__":
    # Run tests directly
    success = run_integration_tests()
    exit(0 if success else 1)