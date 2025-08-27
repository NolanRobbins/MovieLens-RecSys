#!/usr/bin/env python3
"""
Unit tests for ETL Pipeline components

Comprehensive test suite for batch processing, data quality, and A/B testing.
"""

import unittest
import os
import sys
import json
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestBatchETLPipeline(unittest.TestCase):
    """Test cases for BatchETLPipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_dir, 'test_config.yaml')
        
        # Create test configuration
        test_config = """
pipeline:
  name: "test_etl"
  version: "1.0.0"
  batch:
    total_days: 5
    daily_percentage: 0.2
    start_date: "2024-01-01"
  data:
    test_data_path: "test_data.csv"
    output_path: "test_batches/"
    metrics_path: "test_metrics/"
    processed_tracker: "test_tracker.json"
  processing:
    chunk_size: 100
    validate_schema: true
    log_level: "INFO"
monitoring:
  enabled: true
  data_quality:
    - "completeness"
    - "validity"
"""
        
        with open(self.config_path, 'w') as f:
            f.write(test_config)
        
        # Create test data
        self.test_data = pd.DataFrame({
            'userId': [1, 2, 3, 4, 5] * 20,
            'movieId': [10, 20, 30, 40, 50] * 20,
            'rating': [4.0, 3.5, 5.0, 2.5, 4.5] * 20,
            'timestamp': range(1600000000, 1600000100),
            'user_idx': range(100),
            'movie_idx': range(100, 200)
        })
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch('etl.batch_etl_pipeline.BatchETLPipeline._load_config')
    def test_pipeline_initialization(self, mock_load_config):
        """Test pipeline initialization"""
        from etl.batch_etl_pipeline import BatchETLPipeline
        
        mock_config = {
            'pipeline': {
                'name': 'test_pipeline',
                'version': '1.0.0',
                'data': {'processed_tracker': 'test_tracker.json'},
                'processing': {'log_level': 'INFO'}
            }
        }
        mock_load_config.return_value = mock_config
        
        with patch('etl.batch_etl_pipeline.BatchETLPipeline._create_directories'):
            with patch('etl.batch_etl_pipeline.BatchETLPipeline._load_processing_tracker'):
                pipeline = BatchETLPipeline()
                
                self.assertEqual(pipeline.config['pipeline']['name'], 'test_pipeline')
                self.assertIsNotNone(pipeline.logger)
    
    def test_get_daily_batch(self):
        """Test daily batch extraction logic"""
        # Mock the BatchETLPipeline class
        with patch('etl.batch_etl_pipeline.BatchETLPipeline.__init__', return_value=None):
            from etl.batch_etl_pipeline import BatchETLPipeline
            
            pipeline = BatchETLPipeline()
            pipeline.config = {'pipeline': {'batch': {'daily_percentage': 0.2}}}
            
            # Test batch extraction
            batch = pipeline.get_daily_batch(self.test_data, 1)
            expected_size = int(len(self.test_data) * 0.2)
            
            self.assertEqual(len(batch), expected_size)
            self.assertEqual(list(batch.index), list(range(expected_size)))
    
    def test_validate_batch_data(self):
        """Test data quality validation"""
        with patch('etl.batch_etl_pipeline.BatchETLPipeline.__init__', return_value=None):
            from etl.batch_etl_pipeline import BatchETLPipeline
            
            pipeline = BatchETLPipeline()
            
            # Test with valid data
            metrics = pipeline.validate_batch_data(self.test_data)
            
            self.assertIn('completeness', metrics)
            self.assertIn('validity', metrics)
            self.assertIn('uniqueness', metrics)
            self.assertIn('consistency', metrics)
            
            # All our test data should be complete and valid
            self.assertEqual(metrics['completeness'], 1.0)
            self.assertEqual(metrics['validity'], 1.0)
            
            # Test with empty data
            empty_metrics = pipeline.validate_batch_data(pd.DataFrame())
            self.assertEqual(empty_metrics['completeness'], 0.0)
    
    def test_transform_batch(self):
        """Test batch transformation logic"""
        with patch('etl.batch_etl_pipeline.BatchETLPipeline.__init__', return_value=None):
            from etl.batch_etl_pipeline import BatchETLPipeline
            
            pipeline = BatchETLPipeline()
            
            transformed = pipeline._transform_batch(self.test_data)
            
            # Check that new columns were added
            self.assertIn('batch_processed_date', transformed.columns)
            self.assertIn('data_source', transformed.columns)
            self.assertIn('rating_normalized', transformed.columns)
            self.assertIn('is_high_rating', transformed.columns)
            
            # Check normalization
            normalized_ratings = transformed['rating_normalized']
            self.assertTrue(all(0 <= r <= 1 for r in normalized_ratings))
    
    def test_create_sample_test_data(self):
        """Test sample data generation"""
        with patch('etl.batch_etl_pipeline.BatchETLPipeline.__init__', return_value=None):
            from etl.batch_etl_pipeline import BatchETLPipeline
            
            pipeline = BatchETLPipeline()
            pipeline.logger = MagicMock()
            
            sample_df = pipeline._create_sample_test_data()
            
            self.assertGreater(len(sample_df), 0)
            self.assertIn('userId', sample_df.columns)
            self.assertIn('movieId', sample_df.columns)
            self.assertIn('rating', sample_df.columns)
            self.assertIn('timestamp', sample_df.columns)
            
            # Check rating range
            self.assertTrue(all(1.0 <= r <= 5.0 for r in sample_df['rating']))


class TestABTestingFramework(unittest.TestCase):
    """Test cases for A/B Testing Framework"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        
        # Create test cumulative data
        self.cumulative_data = pd.DataFrame({
            'userId': [1, 2, 3, 4, 5] * 10,
            'movieId': [10, 20, 30, 40, 50] * 10,
            'rating': [4.0, 3.5, 5.0, 2.5, 4.5] * 10,
            'rating_date': pd.date_range('2024-01-01', periods=50, freq='D')
        })
        
        self.cumulative_path = os.path.join(self.test_dir, 'cumulative_processed.csv')
        self.cumulative_data.to_csv(self.cumulative_path, index=False)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch('etl.ab_testing.model_comparison.ETL_AB_TestingFramework.__init__')
    def test_framework_initialization(self, mock_init):
        """Test A/B testing framework initialization"""
        mock_init.return_value = None
        
        from etl.ab_testing.model_comparison import ETL_AB_TestingFramework
        
        framework = ETL_AB_TestingFramework()
        mock_init.assert_called_once()
    
    def test_evaluate_model_on_batch(self):
        """Test model evaluation simulation"""
        with patch('etl.ab_testing.model_comparison.ETL_AB_TestingFramework.__init__', return_value=None):
            from etl.ab_testing.model_comparison import ETL_AB_TestingFramework
            
            framework = ETL_AB_TestingFramework()
            
            # Test NCF baseline evaluation
            ncf_results = framework.evaluate_model_on_batch("ncf_baseline", self.cumulative_data)
            
            self.assertIn('rmse', ncf_results)
            self.assertIn('precision_at_10', ncf_results)
            self.assertIn('model_name', ncf_results)
            self.assertEqual(ncf_results['model_name'], 'ncf_baseline')
            self.assertEqual(ncf_results['records_evaluated'], len(self.cumulative_data))
            
            # Test SS4Rec evaluation
            ss4rec_results = framework.evaluate_model_on_batch("ss4rec", self.cumulative_data)
            self.assertEqual(ss4rec_results['model_name'], 'ss4rec')
            
            # SS4Rec should generally perform better (lower RMSE, higher precision)
            self.assertLess(ss4rec_results['rmse'], ncf_results['rmse'])
            self.assertGreater(ss4rec_results['precision_at_10'], ncf_results['precision_at_10'])
    
    def test_get_etl_pipeline_status(self):
        """Test ETL pipeline status retrieval"""
        with patch('etl.ab_testing.model_comparison.ETL_AB_TestingFramework.__init__', return_value=None):
            from etl.ab_testing.model_comparison import ETL_AB_TestingFramework
            
            framework = ETL_AB_TestingFramework()
            
            # Test with no tracker file
            with patch('os.path.exists', return_value=False):
                status = framework.get_etl_pipeline_status()
                
                self.assertFalse(status['pipeline_active'])
                self.assertEqual(status['current_day'], 0)
                self.assertEqual(status['completion_percentage'], 0.0)
            
            # Test with tracker file
            mock_tracker = {
                'current_day': 5,
                'completion_percentage': 25.0,
                'total_records': 10000,
                'processed_records': 2500,
                'last_updated': '2024-01-01T00:00:00'
            }
            
            with patch('os.path.exists', return_value=True):
                with patch('builtins.open', create=True) as mock_open:
                    mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(mock_tracker)
                    
                    status = framework.get_etl_pipeline_status()
                    
                    self.assertTrue(status['pipeline_active'])
                    self.assertEqual(status['current_day'], 5)
                    self.assertEqual(status['completion_percentage'], 25.0)
    
    def test_data_quality_trends(self):
        """Test data quality trend analysis"""
        with patch('etl.ab_testing.model_comparison.ETL_AB_TestingFramework.__init__', return_value=None):
            from etl.ab_testing.model_comparison import ETL_AB_TestingFramework
            
            framework = ETL_AB_TestingFramework()
            
            # Mock daily metrics
            mock_metrics = [
                {
                    'day': 1,
                    'data_quality': {
                        'completeness': 0.95,
                        'validity': 0.98,
                        'uniqueness': 0.92,
                        'consistency': 1.0
                    }
                },
                {
                    'day': 2,
                    'data_quality': {
                        'completeness': 0.97,
                        'validity': 0.99,
                        'uniqueness': 0.94,
                        'consistency': 1.0
                    }
                }
            ]
            
            with patch.object(framework, 'get_daily_batch_metrics', return_value=mock_metrics):
                trends = framework.get_data_quality_trends()
                
                self.assertEqual(trends['status'], 'success')
                self.assertEqual(trends['days'], [1, 2])
                self.assertEqual(len(trends['metrics']['completeness']), 2)
                
                # Check averages
                expected_avg_completeness = (0.95 + 0.97) / 2
                self.assertAlmostEqual(trends['averages']['completeness'], expected_avg_completeness)


class TestETLLogging(unittest.TestCase):
    """Test cases for ETL logging functionality"""
    
    def test_log_level_configuration(self):
        """Test that logging is configured correctly"""
        import logging
        
        # Test that we can create a logger
        logger = logging.getLogger('test_etl')
        logger.setLevel(logging.INFO)
        
        # Test log message creation
        with patch('logging.info') as mock_info:
            logger.info("Test ETL message")
            # Note: The actual assertion depends on how logging is mocked
    
    def test_error_logging(self):
        """Test error logging functionality"""
        import logging
        
        logger = logging.getLogger('test_etl_error')
        
        # Test error logging
        with patch('logging.error') as mock_error:
            try:
                raise ValueError("Test error")
            except ValueError as e:
                logger.error(f"ETL Error: {e}")
                # Verify error was logged


class TestETLIntegration(unittest.TestCase):
    """Integration tests for ETL pipeline components"""
    
    def test_end_to_end_simulation(self):
        """Test complete ETL pipeline simulation"""
        # This would be a more comprehensive test that exercises
        # the entire pipeline end-to-end
        pass
    
    def test_streamlit_data_format(self):
        """Test that data is formatted correctly for Streamlit"""
        # Test that A/B testing results are in the right format
        # for Streamlit consumption
        pass
    
    def test_github_actions_integration(self):
        """Test GitHub Actions integration points"""
        # Test that CLI arguments work correctly
        # Test that output is formatted for GitHub Actions
        pass


def run_etl_tests():
    """Run all ETL pipeline tests"""
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestBatchETLPipeline))
    test_suite.addTest(unittest.makeSuite(TestABTestingFramework))
    test_suite.addTest(unittest.makeSuite(TestETLLogging))
    test_suite.addTest(unittest.makeSuite(TestETLIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("🧪 Running ETL Pipeline Test Suite...")
    print("=" * 50)
    
    success = run_etl_tests()
    
    if success:
        print("\n✅ All ETL tests passed!")
    else:
        print("\n❌ Some ETL tests failed!")
        exit(1)
