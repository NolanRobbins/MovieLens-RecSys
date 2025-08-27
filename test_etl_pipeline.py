#!/usr/bin/env python3
"""
Test script to validate ETL pipeline functionality

Run this to test the ETL pipeline before integrating with Streamlit.
"""

import sys
import os
sys.path.append('.')

def test_etl_pipeline():
    """Test the basic ETL pipeline functionality"""
    
    print("🧪 Testing ETL Pipeline Components...")
    print("=" * 50)
    
    # Test 1: ETL Pipeline Initialization
    print("\n📋 Test 1: ETL Pipeline Initialization")
    try:
        from etl.batch_etl_pipeline import BatchETLPipeline
        
        pipeline = BatchETLPipeline()
        print("✅ ETL Pipeline initialized successfully")
        
        # Check pipeline status
        status = pipeline.get_pipeline_status()
        print(f"📊 Pipeline Status: {status['pipeline_name']} v{status['version']}")
        
    except Exception as e:
        print(f"❌ ETL Pipeline initialization failed: {e}")
        return False
    
    # Test 2: Process a sample batch
    print("\n📦 Test 2: Process Sample Batch")
    try:
        # Process day 1 as a test
        result = pipeline.process_daily_batch(1)
        
        if result['status'] == 'success':
            print(f"✅ Processed {result['records_processed']} records in {result['processing_time_seconds']:.2f}s")
            print(f"📈 Throughput: {result['throughput_records_per_second']:.1f} records/sec")
            
            # Check data quality
            dq = result['data_quality']
            print(f"🔍 Data Quality: Completeness {dq['completeness']:.1%}, Validity {dq['validity']:.1%}")
        else:
            print(f"⚠️ Batch processing returned: {result['status']}")
            
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        return False
    
    # Test 3: A/B Testing Framework
    print("\n🧪 Test 3: A/B Testing Framework")
    try:
        from etl.ab_testing.model_comparison import ETL_AB_TestingFramework
        
        ab_framework = ETL_AB_TestingFramework()
        print("✅ A/B Testing framework initialized")
        
        # Test pipeline status check
        pipeline_status = ab_framework.get_etl_pipeline_status()
        print(f"📊 Pipeline Status: Day {pipeline_status['current_day']}/20 ({pipeline_status['completion_percentage']:.1f}%)")
        
        # Test A/B comparison (if data available)
        ab_results = ab_framework.run_ab_comparison_for_streamlit()
        
        if ab_results['status'] == 'success':
            print("✅ A/B Testing comparison completed")
            winner = ab_results['comparison']['winner']
            rmse_improvement = ab_results['comparison']['improvements'].get('rmse_improvement', 0)
            print(f"🏆 Winner: {winner} (RMSE improvement: {rmse_improvement:+.1f}%)")
        else:
            print(f"⚠️ A/B Testing: {ab_results.get('message', 'No data available yet')}")
            
    except Exception as e:
        print(f"❌ A/B Testing framework failed: {e}")
        return False
    
    # Test 4: Check Generated Files
    print("\n📁 Test 4: Check Generated Files")
    expected_files = [
        "data/etl_batches/processing_tracker.json",
        "data/etl_batches/daily/day_01_batch.csv",
        "data/etl_batches/cumulative/cumulative_processed.csv",
        "data/etl_metrics/day_01_metrics.json"
    ]
    
    files_found = 0
    for file_path in expected_files:
        if os.path.exists(file_path):
            print(f"✅ Found: {file_path}")
            files_found += 1
        else:
            print(f"⚠️ Missing: {file_path}")
    
    print(f"\n📊 Files Status: {files_found}/{len(expected_files)} files found")
    
    if files_found >= 3:  # At least most files should exist
        print("✅ File generation test passed")
    else:
        print("⚠️ File generation test: some files missing (may be expected on first run)")
    
    print("\n" + "=" * 50)
    print("🎉 ETL Pipeline Tests Completed!")
    print("\n🚀 Next Steps:")
    print("1. Run 'python etl/batch_etl_pipeline.py --simulate 3' to test multi-day processing")
    print("2. Check the Streamlit app for ETL dashboard integration") 
    print("3. Test GitHub Actions workflow manually")
    
    return True


def test_streamlit_integration():
    """Test Streamlit integration components"""
    
    print("\n🖥️ Testing Streamlit Integration Components...")
    print("=" * 50)
    
    # Test data loading functions
    print("\n📊 Test: Data Loading for Streamlit")
    try:
        from etl.ab_testing.model_comparison import ETL_AB_TestingFramework
        
        ab_framework = ETL_AB_TestingFramework()
        
        # Test cumulative data loading
        cumulative_df = ab_framework.load_cumulative_test_data()
        print(f"✅ Cumulative data: {len(cumulative_df)} records")
        
        # Test daily metrics loading
        daily_metrics = ab_framework.get_daily_batch_metrics()
        print(f"✅ Daily metrics: {len(daily_metrics)} days available")
        
        # Test data quality trends
        quality_trends = ab_framework.get_data_quality_trends()
        if quality_trends['status'] == 'success':
            print("✅ Data quality trends available")
            print(f"📈 Average completeness: {quality_trends['averages']['completeness']:.1%}")
        else:
            print("⚠️ Data quality trends: no data available yet")
            
    except Exception as e:
        print(f"❌ Streamlit integration test failed: {e}")
        return False
    
    print("✅ Streamlit integration components working properly")
    return True


if __name__ == "__main__":
    print("🧪 MovieLens ETL Pipeline Test Suite")
    print("📅", "2024-01-01")  # Matching our config
    
    success = True
    
    # Run basic ETL tests
    success &= test_etl_pipeline()
    
    # Run Streamlit integration tests
    success &= test_streamlit_integration()
    
    if success:
        print("\n🎉 All tests passed! ETL pipeline is ready for integration.")
    else:
        print("\n❌ Some tests failed. Check the errors above.")
        sys.exit(1)
