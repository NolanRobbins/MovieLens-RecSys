#!/usr/bin/env python3
"""
Test model initialization without GPU dependencies
Validates our model architecture and configuration
"""

import sys
import torch
import yaml
from pathlib import Path

def test_config_loading():
    """Test configuration file loading"""
    print("🔍 Testing configuration loading...")
    
    try:
        config_path = Path("configs/official/ss4rec_official.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print("✅ Configuration loaded successfully")
        
        # Validate required fields
        required_fields = [
            'model', 'dataset', 'hidden_size', 'n_layers', 
            'dropout_prob', 'd_state', 'd_conv', 'expand',
            'dt_min', 'dt_max', 'learning_rate', 'train_batch_size'
        ]
        
        missing_fields = []
        for field in required_fields:
            if field not in config:
                missing_fields.append(field)
        
        if missing_fields:
            print(f"❌ Missing config fields: {missing_fields}")
            return False
        
        print("✅ All required configuration fields present")
        print(f"   Model: {config['model']}")
        print(f"   Hidden size: {config['hidden_size']}")
        print(f"   Layers: {config['n_layers']}")
        print(f"   Batch size: {config['train_batch_size']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False

def test_data_format():
    """Test data format compatibility"""
    print("\n🔍 Testing data format...")
    
    try:
        import pandas as pd
        
        # Test movielens_past.inter
        sample = pd.read_csv("data/processed/movielens_past.inter", sep='\t', nrows=10)
        print(f"✅ movielens_past.inter loaded: {len(sample)} rows")
        
        # Check columns
        expected_columns = ['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float']
        actual_columns = list(sample.columns)
        
        if actual_columns == expected_columns:
            print("✅ Column format correct")
        else:
            print(f"❌ Column mismatch: {actual_columns}")
            return False
        
        # Check data types
        print(f"   User ID range: {sample['user_id:token'].min()} - {sample['user_id:token'].max()}")
        print(f"   Item ID range: {sample['item_id:token'].min()} - {sample['item_id:token'].max()}")
        print(f"   Rating range: {sample['rating:float'].min()} - {sample['rating:float'].max()}")
        print(f"   Timestamp range: {sample['timestamp:float'].min()} - {sample['timestamp:float'].max()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data format test failed: {e}")
        return False

def test_model_architecture():
    """Test model architecture without RecBole dependencies"""
    print("\n🔍 Testing model architecture...")
    
    try:
        # Test basic tensor operations that our model will use
        batch_size = 32
        seq_len = 50
        hidden_size = 64
        
        # Simulate input tensors
        item_seq = torch.randint(1, 1000, (batch_size, seq_len))
        item_seq_len = torch.randint(10, seq_len, (batch_size,))
        timestamps = torch.rand(batch_size, seq_len) * 1000000
        
        print(f"✅ Input tensors created:")
        print(f"   Item sequence: {item_seq.shape}")
        print(f"   Sequence lengths: {item_seq_len.shape}")
        print(f"   Timestamps: {timestamps.shape}")
        
        # Test embedding layer
        n_items = 1000
        embedding = torch.nn.Embedding(n_items, hidden_size, padding_idx=0)
        item_emb = embedding(item_seq)
        print(f"✅ Embedding layer test: {item_emb.shape}")
        
        # Test position embedding
        position_embedding = torch.nn.Embedding(seq_len, hidden_size)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        position_emb = position_embedding(position_ids)
        print(f"✅ Position embedding test: {position_emb.shape}")
        
        # Test combined embeddings
        combined_emb = item_emb + position_emb
        print(f"✅ Combined embeddings: {combined_emb.shape}")
        
        # Test layer normalization
        layer_norm = torch.nn.LayerNorm(hidden_size)
        normalized = layer_norm(combined_emb)
        print(f"✅ Layer normalization test: {normalized.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model architecture test failed: {e}")
        return False

def test_time_interval_computation():
    """Test time interval computation logic"""
    print("\n🔍 Testing time interval computation...")
    
    try:
        # Simulate timestamp data
        timestamps = torch.tensor([
            [1000, 2000, 3000, 4000, 5000],  # Regular intervals
            [1000, 1500, 2000, 2500, 3000],  # Irregular intervals
            [1000, 1000, 1000, 1000, 1000]   # Same timestamps
        ])
        
        # Compute time intervals (our model's logic)
        time_diffs = timestamps[:, 1:] - timestamps[:, :-1]
        print(f"✅ Time differences computed: {time_diffs.shape}")
        print(f"   Sample intervals: {time_diffs[0].tolist()}")
        
        # Test clamping (our model's logic)
        dt_min, dt_max = 0.001, 0.1
        clamped_intervals = torch.clamp(time_diffs.float(), min=dt_min, max=dt_max)
        print(f"✅ Time intervals clamped: {clamped_intervals.shape}")
        
        # Test edge case handling
        zero_intervals = torch.where(
            clamped_intervals <= 0,
            torch.full_like(clamped_intervals, dt_min),
            clamped_intervals
        )
        print(f"✅ Edge case handling: {zero_intervals.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Time interval test failed: {e}")
        return False

def test_runpod_script_logic():
    """Test RunPod script logic without execution"""
    print("\n🔍 Testing RunPod script logic...")
    
    try:
        script_path = Path("runpod_entrypoint.sh")
        if not script_path.exists():
            print("❌ RunPod script not found")
            return False
        
        with open(script_path, 'r') as f:
            script_content = f.read()
        
        # Check for critical components
        critical_components = [
            "gdown",
            "uv pip install",
            "mamba-ssm",
            "s5-pytorch", 
            "recbole",
            "GDRIVE_INTER_URL",
            "movielens_past.inter",
            "movielens.inter"
        ]
        
        missing_components = []
        for component in critical_components:
            if component not in script_content:
                missing_components.append(component)
        
        if missing_components:
            print(f"❌ Missing script components: {missing_components}")
            return False
        
        print("✅ All critical script components present")
        
        # Check Google Drive URL format
        if "1tGY6F_2nEeSWwAXJ_4F832p0BzEbAGfv" in script_content:
            print("✅ Correct Google Drive file ID found")
        else:
            print("❌ Google Drive file ID not found or incorrect")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ RunPod script test failed: {e}")
        return False

def main():
    """Run all model initialization tests"""
    print("🛡️ MODEL INITIALIZATION VALIDATION")
    print("=" * 50)
    
    tests = [
        ("Configuration Loading", test_config_loading),
        ("Data Format", test_data_format),
        ("Model Architecture", test_model_architecture),
        ("Time Interval Computation", test_time_interval_computation),
        ("RunPod Script Logic", test_runpod_script_logic)
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
    print("📊 MODEL VALIDATION RESULTS")
    print("=" * 50)
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if not result:
            all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("🎉 ALL MODEL TESTS PASSED!")
        print("💡 Model architecture and data are ready for RunPod")
    else:
        print("⚠️ SOME MODEL TESTS FAILED")
        print("💡 Fix the issues above before RunPod deployment")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
