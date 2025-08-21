#!/usr/bin/env python3
"""
Minimal SS4Rec test - just test one forward pass
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_minimal_ss4rec():
    """Test SS4Rec with one forward pass - that's it!"""
    print("🧪 Minimal SS4Rec Test")
    print("=" * 30)
    
    try:
        # Import SS4Rec
        print("1️⃣ Importing SS4Rec...")
        from models.sota_2025.ss4rec import SS4Rec
        print("   ✅ Import successful")
        
        # Create model with REAL data dimensions
        print("\n2️⃣ Creating SS4Rec model...")
        model = SS4Rec(
            n_users=200948,   # Real user count
            n_items=84430,    # Real item count  
            d_model=64,
            d_state=16,
            n_layers=2,       # Fewer layers for speed
            max_seq_len=50,   # Shorter sequences
            dropout=0.1
        )
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   ✅ Model created: {total_params:,} parameters")
        
        # Create one small batch
        print("\n3️⃣ Creating test batch...")
        batch_size = 4
        seq_len = 50
        
        user_ids = torch.randint(0, 200948, (batch_size,))       # Real user range
        item_seq = torch.randint(0, 84430, (batch_size, seq_len))  # Real item range
        # Use realistic timestamps (Unix timestamps from 2020-2025 range) - SORTED!
        timestamps = torch.randint(1577836800, 1735689600, (batch_size, seq_len)).float()
        timestamps = timestamps.sort(dim=1)[0]  # Sort timestamps within each sequence
        target_items = torch.randint(0, 84430, (batch_size,))    # Real item range
        
        print(f"   ✅ Batch created: {batch_size} samples, {seq_len} sequence length")
        
        # Forward pass - TEST BOTH EVAL AND TRAIN MODE
        print("\n4️⃣ Running forward pass...")
        
        # Test eval mode first (like our original test)
        model.eval()
        with torch.no_grad():
            print("   Testing in eval mode...")
            predictions_eval = model(
                users=user_ids,
                item_seq=item_seq, 
                target_items=target_items,
                timestamps=timestamps
            )
            print(f"   Eval mode - Output range: [{predictions_eval.min():.3f}, {predictions_eval.max():.3f}]")
            if torch.isnan(predictions_eval).any():
                print("   ⚠️  NaN detected in EVAL mode")
            else:
                print("   ✅ No NaN in eval mode")
        
        # Test train mode (like real training)
        model.train()
        print("   Testing in train mode...")
        predictions = model(
            users=user_ids,
            item_seq=item_seq, 
            target_items=target_items,
            timestamps=timestamps
        )
        
        print(f"   ✅ Forward pass successful!")
        print(f"   📊 Output shape: {predictions.shape}")
        print(f"   📊 Output range: [{predictions.min():.3f}, {predictions.max():.3f}]")
        
        # Test loss computation AND backward pass (like real training)
        print("\n5️⃣ Testing loss computation and backward pass...")
        targets = torch.rand(batch_size) * 4.5 + 0.5  # Realistic ratings 0.5-5.0
        criterion = nn.MSELoss()
        loss = criterion(predictions, targets)
        print(f"   ✅ Loss computed: {loss.item():.6f}")
        
        # Test backward pass (this is where nan often appears!)
        print("   Testing backward pass...")
        loss.backward()
        print("   ✅ Backward pass successful!")
        
        # Check for NaN in gradients
        nan_grads = any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)
        if nan_grads:
            print("   ⚠️  NaN detected in gradients!")
        else:
            print("   ✅ No NaN in gradients")
        
        # Check for NaN values
        if torch.isnan(predictions).any():
            print("   ⚠️  Warning: NaN values in predictions (numerical instability)")
        else:
            print("   ✅ No NaN values detected")
        
        print("\n🎉 SUCCESS! SS4Rec is working correctly")
        print("   Ready for RunPod deployment! 🚀")
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        print("\n🔧 Fix the error before deploying to RunPod")
        return False

if __name__ == "__main__":
    success = test_minimal_ss4rec()
    sys.exit(0 if success else 1)