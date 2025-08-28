#!/usr/bin/env python3
"""
Test Official SS4Rec Implementation

This script tests the official SS4Rec implementation to ensure all dependencies
are working correctly before running on RunPod.

Usage:
    python test_official_ss4rec.py
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))


def setup_logging():
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def test_dependencies():
    """Test all required dependencies"""
    logging.info("🧪 Testing Official SS4Rec Dependencies")
    
    dependencies = [
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('recbole', 'RecBole'),
        ('mamba_ssm', 'Mamba SSM'),
        ('s5_pytorch', 'S5 PyTorch'), 
        ('causal_conv1d', 'Causal Conv1D')
    ]
    
    missing = []
    for package, name in dependencies:
        try:
            __import__(package)
            logging.info(f"✅ {name} available")
        except ImportError as e:
            logging.error(f"❌ {name} not available: {e}")
            missing.append((package, name))
    
    return len(missing) == 0


def test_model_creation():
    """Test SS4Rec model creation"""
    logging.info("🧪 Testing SS4Rec Model Creation")
    
    try:
        from models.official_ss4rec import SS4RecOfficial, create_ss4rec_config
        
        # Create dummy dataset
        class DummyDataset:
            def __init__(self):
                self.field2token_id = {'item_id': {'<pad>': 0}}
                
        # Create config and model
        config = create_ss4rec_config()
        config['n_items'] = 1000  # Add required field
        
        # Enhanced dummy dataset
        class DummyDataset:
            def __init__(self):
                self.field2token_id = {'item_id': {'<pad>': 0}}
                self.num_items = 1000
                
        dataset = DummyDataset()
        
        # Test model creation
        model = SS4RecOfficial(config, dataset)
        logging.info(f"✅ Model created successfully")
        logging.info(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        logging.info(f"  - Hidden size: {model.hidden_size}")
        logging.info(f"  - Layers: {model.n_layers}")
        
        return True
        
    except Exception as e:
        logging.error(f"❌ Model creation failed: {e}")
        logging.error("Full error:", exc_info=True)
        return False


def test_data_conversion():
    """Test data conversion to RecBole format"""
    logging.info("🧪 Testing Data Conversion")
    
    try:
        # Check if converted data exists
        recbole_data = project_root / 'data' / 'recbole_format' / 'movielens.inter'
        if recbole_data.exists():
            logging.info("✅ RecBole format data already exists")
            
            # Check data format
            import pandas as pd
            data = pd.read_csv(recbole_data, sep='\t', nrows=10)
            logging.info(f"  - Columns: {list(data.columns)}")
            logging.info(f"  - Data types: {dict(data.dtypes)}")
            logging.info(f"  - Sample size: {len(data)} (first 10 rows)")
            
            return True
        else:
            logging.warning("⚠️ RecBole format data not found")
            logging.info("💡 Run: python data/recbole_format/movielens_adapter.py")
            return False
            
    except Exception as e:
        logging.error(f"❌ Data conversion test failed: {e}")
        return False


def test_training_config():
    """Test training configuration loading"""
    logging.info("🧪 Testing Training Configuration")
    
    try:
        config_file = project_root / 'configs' / 'official' / 'ss4rec_official.yaml'
        if config_file.exists():
            import yaml
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            logging.info("✅ Configuration loaded successfully")
            logging.info(f"  - Model: {config.get('model', 'N/A')}")
            logging.info(f"  - Hidden size: {config.get('hidden_size', 'N/A')}")
            logging.info(f"  - Batch size: {config.get('train_batch_size', 'N/A')}")
            logging.info(f"  - Learning rate: {config.get('learning_rate', 'N/A')}")
            
            return True
        else:
            logging.error(f"❌ Configuration file not found: {config_file}")
            return False
            
    except Exception as e:
        logging.error(f"❌ Configuration loading failed: {e}")
        return False


def main():
    """Main test function"""
    setup_logging()
    
    logging.info("🎬 Testing Official SS4Rec Implementation")
    logging.info("=" * 50)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Model Creation", test_model_creation), 
        ("Data Conversion", test_data_conversion),
        ("Training Config", test_training_config)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logging.info(f"\n🔍 Running {test_name} Test...")
        try:
            if test_func():
                logging.info(f"✅ {test_name} Test PASSED")
                passed += 1
            else:
                logging.error(f"❌ {test_name} Test FAILED")
        except Exception as e:
            logging.error(f"❌ {test_name} Test ERROR: {e}")
    
    # Summary
    logging.info("\n" + "=" * 50)
    logging.info(f"🏁 Test Results: {passed}/{total} PASSED")
    
    if passed == total:
        logging.info("🎉 All tests passed! Ready for RunPod training:")
        logging.info("   ./runpod_entrypoint.sh --model ss4rec-official")
        return 0
    else:
        logging.error(f"❌ {total - passed} tests failed")
        logging.info("💡 Install missing dependencies:")
        logging.info("   uv pip install -r requirements_ss4rec.txt")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)