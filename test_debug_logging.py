#!/usr/bin/env python3
"""
Quick test to verify debug logging is working in SS4Rec components
"""

import sys
import logging
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def setup_logging():
    """Set up debug logging"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # Enable specific loggers
    logging.getLogger("models.sota_2025.ss4rec").setLevel(logging.DEBUG)
    logging.getLogger("models.sota_2025.components.state_space_models").setLevel(logging.DEBUG)

def test_s5_layer():
    """Test S5Layer with debug logging"""
    print("\n🔍 Testing S5Layer with debug logging...")
    
    from models.sota_2025.components.state_space_models import S5Layer
    
    # Create small S5Layer
    s5_layer = S5Layer(d_model=32, d_state=8)
    
    # Small test input
    batch_size, seq_len = 2, 5
    x = torch.randn(batch_size, seq_len, 32) * 0.1
    time_intervals = torch.ones(batch_size, seq_len - 1) * 0.1
    
    print(f"Input shape: {x.shape}")
    print(f"Time intervals shape: {time_intervals.shape}")
    
    # Forward pass
    try:
        output = s5_layer(x, time_intervals)
        print(f"✅ S5Layer forward pass successful, output shape: {output.shape}")
        print(f"Output range: [{output.min():.6f}, {output.max():.6f}]")
        print(f"Has NaN: {torch.isnan(output).any()}")
        print(f"Has Inf: {torch.isinf(output).any()}")
    except Exception as e:
        print(f"❌ S5Layer forward pass failed: {e}")
        import traceback
        traceback.print_exc()

def test_mamba_layer():
    """Test MambaLayer with debug logging"""
    print("\n🔍 Testing MambaLayer with debug logging...")
    
    from models.sota_2025.components.state_space_models import MambaLayer
    
    # Create small MambaLayer
    mamba_layer = MambaLayer(d_model=32, d_state=8)
    
    # Small test input
    batch_size, seq_len = 2, 5
    x = torch.randn(batch_size, seq_len, 32) * 0.1
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    try:
        output = mamba_layer(x)
        print(f"✅ MambaLayer forward pass successful, output shape: {output.shape}")
        print(f"Output range: [{output.min():.6f}, {output.max():.6f}]")
        print(f"Has NaN: {torch.isnan(output).any()}")
        print(f"Has Inf: {torch.isinf(output).any()}")
    except Exception as e:
        print(f"❌ MambaLayer forward pass failed: {e}")
        import traceback
        traceback.print_exc()

def test_ss4rec_model():
    """Test full SS4Rec model with debug logging"""
    print("\n🔍 Testing SS4Rec model with debug logging...")
    
    from models.sota_2025.ss4rec import create_ss4rec_model
    
    # Create small model
    n_users, n_items = 100, 50
    config = {
        'd_model': 32,
        'n_layers': 1,
        'd_state': 8,
        'max_seq_len': 10
    }
    
    model = create_ss4rec_model(n_users, n_items, config)
    
    # Small test input
    batch_size, seq_len = 2, 5
    users = torch.randint(0, n_users, (batch_size,))
    item_seq = torch.randint(1, n_items, (batch_size, seq_len))  # Start from 1 (0 is padding)
    target_items = torch.randint(1, n_items, (batch_size,))
    timestamps = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).float()
    
    print(f"Users shape: {users.shape}")
    print(f"Item sequence shape: {item_seq.shape}")
    print(f"Target items shape: {target_items.shape}")
    print(f"Timestamps shape: {timestamps.shape}")
    
    # Forward pass
    try:
        output = model(users, item_seq, target_items, timestamps)
        print(f"✅ SS4Rec forward pass successful, output shape: {output.shape}")
        print(f"Output range: [{output.min():.6f}, {output.max():.6f}]")
        print(f"Has NaN: {torch.isnan(output).any()}")
        print(f"Has Inf: {torch.isinf(output).any()}")
    except Exception as e:
        print(f"❌ SS4Rec forward pass failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all tests"""
    print("🚀 Testing SS4Rec Debug Logging")
    print("=" * 50)
    
    setup_logging()
    
    test_s5_layer()
    test_mamba_layer() 
    test_ss4rec_model()
    
    print("\n" + "=" * 50)
    print("🏁 Debug logging tests completed")

if __name__ == "__main__":
    main()