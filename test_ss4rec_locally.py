#!/usr/bin/env python3
"""
Test SS4Rec model locally before deploying to RunPod
This script validates all the tensor operations without full training
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_imports():
    """Test that all required imports work"""
    print("🔍 Testing imports...")
    try:
        from models.sota_2025.ss4rec import SS4Rec
        from models.sota_2025.components.state_space_models import S5Layer, SSBlock
        print("✅ Core model imports successful")
        
        # Test training imports separately (wandb might be missing)
        try:
            import wandb
            from training.train_ss4rec import MovieLensSequentialDataset
            print("✅ Training imports successful")
        except ImportError as e:
            print(f"⚠️  Training imports failed (this is OK for local testing): {e}")
            
        return True
    except ImportError as e:
        print(f"❌ Core import failed: {e}")
        return False

def test_model_creation():
    """Test SS4Rec model creation with typical parameters"""
    print("\n🔍 Testing SS4Rec model creation...")
    try:
        from models.sota_2025.ss4rec import SS4Rec
        
        model_config = {
            'n_users': 200948,
            'n_items': 84430, 
            'd_model': 64,
            'd_state': 16,
            'n_layers': 4,
            'max_seq_len': 200,
            'dropout': 0.1
        }
        
        model = SS4Rec(**model_config)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ SS4Rec model created successfully")
        print(f"   Total parameters: {total_params:,}")
        return model, model_config
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_forward_pass(model, model_config):
    """Test a forward pass with realistic tensor shapes"""
    print("\n🔍 Testing forward pass with realistic data shapes...")
    try:
        batch_size = 32  # Smaller batch for local testing
        seq_len = model_config['max_seq_len']
        
        # Create sample inputs
        user_ids = torch.randint(0, model_config['n_users'], (batch_size,))
        item_seq = torch.randint(0, model_config['n_items'], (batch_size, seq_len))
        timestamps = torch.randn(batch_size, seq_len)  # Normalized timestamps
        target_items = torch.randint(0, model_config['n_items'], (batch_size,))
        
        print(f"   Input shapes:")
        print(f"     user_ids: {user_ids.shape}")
        print(f"     item_seq: {item_seq.shape}")
        print(f"     timestamps: {timestamps.shape}")
        print(f"     target_items: {target_items.shape}")
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            predictions = model(
                users=user_ids,
                item_seq=item_seq,
                target_items=target_items,
                timestamps=timestamps
            )
        
        print(f"✅ Forward pass successful")
        print(f"   Output shape: {predictions.shape}")
        print(f"   Output range: [{predictions.min():.3f}, {predictions.max():.3f}]")
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_loss_computation():
    """Test loss computation with sample predictions and targets"""
    print("\n🔍 Testing loss computation...")
    try:
        batch_size = 32
        predictions = torch.randn(batch_size, 1)  # Random predictions
        targets = torch.randn(batch_size, 1)      # Random targets
        
        criterion = nn.MSELoss()
        loss = criterion(predictions, targets)
        
        print(f"✅ Loss computation successful")
        print(f"   Loss value: {loss.item():.6f}")
        return True
        
    except Exception as e:
        print(f"❌ Loss computation failed: {e}")
        return False

def test_dataset_creation():
    """Test dataset creation with dummy data"""
    print("\n🔍 Testing dataset creation...")
    try:
        # Create dummy data that matches expected format
        n_samples = 1000
        dummy_data = pd.DataFrame({
            'user_idx': np.random.randint(0, 1000, n_samples),
            'movie_idx': np.random.randint(0, 5000, n_samples), 
            'rating': np.random.uniform(1, 5, n_samples),
            'timestamp': np.random.uniform(0, 1, n_samples)  # Normalized
        })
        
        from training.train_ss4rec import MovieLensSequentialDataset
        dataset = MovieLensSequentialDataset(
            dummy_data, 
            max_seq_len=200, 
            mode='train'
        )
        
        # Test getting a sample
        sample = dataset[0]
        print(f"✅ Dataset creation successful")
        print(f"   Dataset length: {len(dataset)}")
        print(f"   Sample keys: {list(sample.keys())}")
        print(f"   Sample shapes: {[(k, v.shape if hasattr(v, 'shape') else type(v)) for k, v in sample.items()]}")
        return True
        
    except Exception as e:
        print(f"❌ Dataset creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_s5layer_directly():
    """Test S5Layer component directly with the fixed tensor operations"""
    print("\n🔍 Testing S5Layer component...")
    try:
        from models.sota_2025.components.state_space_models import S5Layer
        
        batch_size = 32
        seq_len = 200
        d_model = 64
        d_state = 16
        
        # Create S5Layer
        s5_layer = S5Layer(d_model=d_model, d_state=d_state)
        
        # Create inputs
        x = torch.randn(batch_size, seq_len, d_model)
        time_intervals = torch.randn(batch_size, seq_len - 1)  # One less than seq_len
        
        # Forward pass
        output = s5_layer(x, time_intervals)
        
        print(f"✅ S5Layer test successful")
        print(f"   Input shape: {x.shape}")
        print(f"   Time intervals shape: {time_intervals.shape}")
        print(f"   Output shape: {output.shape}")
        return True
        
    except Exception as e:
        print(f"❌ S5Layer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 SS4Rec Local Testing Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_s5layer_directly,
        test_model_creation,
        test_dataset_creation, 
        test_loss_computation,
    ]
    
    results = []
    model = None
    
    for test_func in tests:
        if test_func.__name__ == 'test_model_creation':
            model, model_config = test_func()
            results.append(model is not None)
        elif test_func.__name__ == 'test_forward_pass':
            if model is not None:
                results.append(test_func(model, model_config))
            else:
                print(f"\n⏭️  Skipping {test_func.__name__} (model creation failed)")
                results.append(False)
        else:
            results.append(test_func())
    
    # Test forward pass if model was created successfully
    if model is not None:
        results.append(test_forward_pass(model, model_config))
    
    # Summary
    print(f"\n📊 Test Summary")
    print("=" * 50)
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! SS4Rec should work on RunPod")
        return True
    else:
        print("❌ Some tests failed. Fix issues before deploying to RunPod")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)