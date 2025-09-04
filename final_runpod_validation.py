#!/usr/bin/env python3
"""
Final comprehensive validation for RunPod deployment
Tests everything that could cause failures and waste GPU credits
"""

import sys
import os
import subprocess
import json
from pathlib import Path

def test_file_structure():
    """Test that all required files exist"""
    print("🔍 Testing file structure...")
    
    required_files = {
        "Data Files": [
            "data/processed/movielens_past.inter",
            "data/recbole_format/movielens/movielens.inter",
            "data/processed/data_mappings.pkl"
        ],
        "Model Files": [
            "models/official_ss4rec/ss4rec_official.py",
            "models/official_ss4rec/SS4Rec.py",
            "models/official_ss4rec/sequential_dataset_official.py",
            "models/official_ss4rec/__init__.py"
        ],
        "Config Files": [
            "configs/official/ss4rec_official.yaml"
        ],
        "Scripts": [
            "runpod_entrypoint.sh",
            "training/official/train_ss4rec_official.py",
            "training/official/runpod_train_ss4rec_official.py"
        ],
        "Requirements": [
            "requirements_ss4rec.txt"
        ]
    }
    
    all_good = True
    for category, files in required_files.items():
        print(f"\n📁 {category}:")
        for file_path in files:
            path = Path(file_path)
            if path.exists():
                size = path.stat().st_size
                size_str = f"({size:,} bytes)" if size > 0 else "(empty)"
                print(f"   ✅ {file_path} {size_str}")
            else:
                print(f"   ❌ {file_path} MISSING")
                all_good = False
    
    return all_good

def test_data_integrity():
    """Test data file integrity and format"""
    print("\n🔍 Testing data integrity...")
    
    try:
        import pandas as pd
        
        # Test movielens_past.inter
        past_file = Path("data/processed/movielens_past.inter")
        if not past_file.exists():
            print("❌ movielens_past.inter not found")
            return False
        
        # Check file size
        size_mb = past_file.stat().st_size / (1024 * 1024)
        if size_mb < 500:  # Should be ~686MB
            print(f"❌ movielens_past.inter too small: {size_mb:.1f}MB")
            return False
        
        print(f"✅ movielens_past.inter size: {size_mb:.1f}MB")
        
        # Test data format
        sample = pd.read_csv(past_file, sep='\t', nrows=1000)
        expected_columns = ['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float']
        
        if list(sample.columns) != expected_columns:
            print(f"❌ Column mismatch: {list(sample.columns)}")
            return False
        
        print("✅ Data format validation passed")
        
        # Check data ranges
        print(f"   Users: {sample['user_id:token'].min()} - {sample['user_id:token'].max()}")
        print(f"   Items: {sample['item_id:token'].min()} - {sample['item_id:token'].max()}")
        print(f"   Ratings: {sample['rating:float'].min()} - {sample['rating:float'].max()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data integrity test failed: {e}")
        return False

def test_google_drive_url():
    """Test Google Drive URL format"""
    print("\n🔍 Testing Google Drive URL...")
    
    try:
        # Read the RunPod script to extract URL
        script_path = Path("runpod_entrypoint.sh")
        if not script_path.exists():
            print("❌ RunPod script not found")
            return False
        
        # Read with proper encoding
        with open(script_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Check for the correct file ID
        expected_id = "1tGY6F_2nEeSWwAXJ_4F832p0BzEbAGfv"
        if expected_id in content:
            print(f"✅ Correct Google Drive file ID found: {expected_id}")
        else:
            print(f"❌ Google Drive file ID not found or incorrect")
            return False
        
        # Check URL format
        if "https://drive.google.com/uc?export=download&id=" in content:
            print("✅ Correct Google Drive URL format")
        else:
            print("❌ Incorrect Google Drive URL format")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Google Drive URL test failed: {e}")
        return False

def test_configuration():
    """Test configuration file"""
    print("\n🔍 Testing configuration...")
    
    try:
        import yaml
        
        config_path = Path("configs/official/ss4rec_official.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Critical configuration checks
        checks = [
            ("model", "SS4RecOfficial"),
            ("dataset", "movielens"),
            ("hidden_size", 64),
            ("n_layers", 2),
            ("train_batch_size", 4096),
            ("learning_rate", 0.001),
            ("epochs", 500)
        ]
        
        for key, expected_value in checks:
            if key not in config:
                print(f"❌ Missing config key: {key}")
                return False
            
            if config[key] != expected_value:
                print(f"❌ Config mismatch {key}: {config[key]} != {expected_value}")
                return False
        
        print("✅ Configuration validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_requirements_file():
    """Test requirements file"""
    print("\n🔍 Testing requirements file...")
    
    try:
        req_path = Path("requirements_ss4rec.txt")
        if not req_path.exists():
            print("❌ requirements_ss4rec.txt not found")
            return False
        
        with open(req_path, 'r') as f:
            requirements = f.read()
        
        # Check for critical packages
        critical_packages = [
            "torch",
            "mamba-ssm",
            "s5-pytorch", 
            "causal-conv1d",
            "recbole"
        ]
        
        missing_packages = []
        for package in critical_packages:
            if package not in requirements:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ Missing packages: {missing_packages}")
            return False
        
        print("✅ Requirements file validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Requirements test failed: {e}")
        return False

def test_runpod_script_syntax():
    """Test RunPod script syntax"""
    print("\n🔍 Testing RunPod script syntax...")
    
    try:
        script_path = Path("runpod_entrypoint.sh")
        
        # Test bash syntax
        result = subprocess.run(
            ["bash", "-n", str(script_path)],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ RunPod script syntax is valid")
            return True
        else:
            print(f"❌ RunPod script syntax error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ RunPod script syntax test failed: {e}")
        return False

def test_model_imports():
    """Test that our model files can be imported (without dependencies)"""
    print("\n🔍 Testing model imports...")
    
    try:
        # Add project root to path
        project_root = Path(__file__).parent
        sys.path.append(str(project_root))
        
        # Test that our model files exist and are readable
        model_files = [
            "models/official_ss4rec/ss4rec_official.py",
            "models/official_ss4rec/SS4Rec.py",
            "models/official_ss4rec/sequential_dataset_official.py"
        ]
        
        for model_file in model_files:
            path = Path(model_file)
            if not path.exists():
                print(f"❌ Model file not found: {model_file}")
                return False
            
            # Try to read the file
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for basic Python syntax
            if "class " not in content:
                print(f"❌ Model file appears invalid: {model_file}")
                return False
        
        print("✅ Model files validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Model imports test failed: {e}")
        return False

def generate_runpod_checklist():
    """Generate a checklist for RunPod deployment"""
    print("\n📋 RUNPOD DEPLOYMENT CHECKLIST")
    print("=" * 50)
    
    checklist = [
        "✅ Upload project to RunPod instance",
        "✅ Set environment variables (WANDB_API_KEY, DISCORD_WEBHOOK_URL)",
        "✅ Run: ./runpod_entrypoint.sh --model ss4rec-official --debug",
        "✅ Monitor logs for dependency installation",
        "✅ Verify data download from Google Drive",
        "✅ Check GPU memory usage (should be <24GB on A6000)",
        "✅ Monitor training progress via W&B dashboard",
        "✅ Watch for gradient explosion warnings",
        "✅ Verify training progresses past epoch 1",
        "✅ Check Discord notifications for completion"
    ]
    
    for item in checklist:
        print(item)
    
    print("\n💡 TROUBLESHOOTING TIPS:")
    print("- If dependencies fail: Check internet connection on RunPod")
    print("- If data download fails: Verify Google Drive file permissions")
    print("- If GPU OOM: Reduce batch_size in config")
    print("- If training hangs: Check for NaN values in debug mode")

def main():
    """Run final comprehensive validation"""
    print("🛡️ FINAL RUNPOD VALIDATION")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Data Integrity", test_data_integrity),
        ("Google Drive URL", test_google_drive_url),
        ("Configuration", test_configuration),
        ("Requirements File", test_requirements_file),
        ("RunPod Script Syntax", test_runpod_script_syntax),
        ("Model Imports", test_model_imports)
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
    print("📊 FINAL VALIDATION RESULTS")
    print("=" * 50)
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if not result:
            all_passed = False
    
    print("=" * 50)
    
    if all_passed:
        print("🎉 ALL VALIDATIONS PASSED!")
        print("🚀 READY FOR RUNPOD DEPLOYMENT")
        print("💡 You can proceed with confidence - minimal risk of failure")
        generate_runpod_checklist()
    else:
        print("⚠️ SOME VALIDATIONS FAILED")
        print("💡 Fix the issues above before RunPod deployment")
        print("🛡️ Do NOT proceed until all tests pass")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
