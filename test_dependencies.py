#!/usr/bin/env python3
"""
Comprehensive dependency test to avoid RunPod failures
Tests all critical imports and basic functionality
"""

import sys
import traceback
from pathlib import Path

def test_basic_imports():
    """Test basic Python packages"""
    print("🔍 Testing basic imports...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU count: {torch.cuda.device_count()}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✅ Pandas: {pd.__version__}")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    return True

def test_recbole_imports():
    """Test RecBole imports"""
    print("\n🔍 Testing RecBole imports...")
    
    try:
        import recbole
        print(f"✅ RecBole: {recbole.__version__}")
    except ImportError as e:
        print(f"❌ RecBole import failed: {e}")
        print("💡 Install with: pip install recbole==1.2.0")
        return False
    
    try:
        from recbole.quick_start import run_recbole
        from recbole.config import Config
        from recbole.data import create_dataset, data_preparation
        from recbole.trainer import Trainer
        from recbole.utils import init_seed, init_logger
        print("✅ RecBole core modules imported successfully")
    except ImportError as e:
        print(f"❌ RecBole core import failed: {e}")
        return False
    
    return True

def test_ss4rec_dependencies():
    """Test SS4Rec specific dependencies"""
    print("\n🔍 Testing SS4Rec dependencies...")
    
    try:
        import mamba_ssm
        print(f"✅ Mamba-SSM: {mamba_ssm.__version__}")
    except ImportError as e:
        print(f"❌ Mamba-SSM import failed: {e}")
        print("💡 Install with: pip install mamba-ssm==2.2.2")
        return False
    
    try:
        import s5
        print(f"✅ S5-PyTorch: {s5.__version__}")
    except ImportError as e:
        print(f"❌ S5-PyTorch import failed: {e}")
        print("💡 Install with: pip install s5-pytorch==0.2.1")
        return False
    
    try:
        import causal_conv1d
        print(f"✅ Causal Conv1D: {causal_conv1d.__version__}")
    except ImportError as e:
        print(f"❌ Causal Conv1D import failed: {e}")
        print("💡 Install with: pip install causal-conv1d>=1.2.0")
        return False
    
    return True

def test_our_model_imports():
    """Test our custom model imports"""
    print("\n🔍 Testing our model imports...")
    
    try:
        # Add project root to path
        project_root = Path(__file__).parent
        sys.path.append(str(project_root))
        
        from models.official_ss4rec.ss4rec_official import SS4RecOfficial, create_ss4rec_config
        print("✅ SS4RecOfficial model imported successfully")
        
        # Test config creation
        config = create_ss4rec_config()
        print("✅ SS4Rec config creation successful")
        
    except ImportError as e:
        print(f"❌ SS4Rec model import failed: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"❌ SS4Rec model test failed: {e}")
        traceback.print_exc()
        return False
    
    return True

def test_data_files():
    """Test data file existence and format"""
    print("\n🔍 Testing data files...")
    
    # Check required files
    required_files = [
        "data/processed/movielens_past.inter",
        "data/recbole_format/movielens/movielens.inter",
        "configs/official/ss4rec_official.yaml"
    ]
    
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"✅ {file_path} exists ({size_mb:.1f} MB)")
        else:
            print(f"❌ {file_path} missing")
            return False
    
    # Test data format
    try:
        import pandas as pd
        sample = pd.read_csv("data/processed/movielens_past.inter", sep='\t', nrows=5)
        expected_columns = ['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float']
        actual_columns = list(sample.columns)
        
        if actual_columns == expected_columns:
            print("✅ Data format validation passed")
        else:
            print(f"❌ Data format mismatch: {actual_columns}")
            return False
            
    except Exception as e:
        print(f"❌ Data format test failed: {e}")
        return False
    
    return True

def main():
    """Run all dependency tests"""
    print("🛡️ COMPREHENSIVE DEPENDENCY VALIDATION")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("RecBole Imports", test_recbole_imports),
        ("SS4Rec Dependencies", test_ss4rec_dependencies),
        ("Our Model Imports", test_our_model_imports),
        ("Data Files", test_data_files)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if not result:
            all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED - Ready for RunPod!")
        print("💡 You can proceed with confidence")
    else:
        print("⚠️ SOME TESTS FAILED - Fix issues before RunPod")
        print("💡 Address the failures above before spending GPU credits")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
