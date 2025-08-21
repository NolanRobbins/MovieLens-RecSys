#!/usr/bin/env python3
"""
Debug NaN values in SS4Rec
"""

import sys
import os
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def debug_nan():
    print("🔍 Debugging NaN values in SS4Rec")
    
    try:
        from models.sota_2025.components.state_space_models import S5Layer
        
        # Create small S5Layer
        s5_layer = S5Layer(d_model=64, d_state=16)
        
        # Small batch
        batch_size = 2
        seq_len = 5
        x = torch.randn(batch_size, seq_len, 64) * 0.1  # Small values
        time_intervals = torch.ones(batch_size, seq_len - 1) * 0.1  # Small, positive values
        
        print(f"Input x range: [{x.min():.3f}, {x.max():.3f}]")
        print(f"Time intervals range: [{time_intervals.min():.3f}, {time_intervals.max():.3f}]")
        
        # Forward pass
        output = s5_layer(x, time_intervals)
        
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
        print(f"Has NaN: {torch.isnan(output).any()}")
        print(f"Has Inf: {torch.isinf(output).any()}")
        
        # Check individual components
        with torch.no_grad():
            dt = s5_layer.dt_proj(x)
            dt = torch.sigmoid(dt) * (s5_layer.dt_max - s5_layer.dt_min) + s5_layer.dt_min
            print(f"dt range: [{dt.min():.6f}, {dt.max():.6f}]")
            
            A = -torch.exp(s5_layer.A_log)
            print(f"A range: [{A.min():.6f}, {A.max():.6f}]")
            
            # Check for problematic values
            print(f"A_log range: [{s5_layer.A_log.min():.3f}, {s5_layer.A_log.max():.3f}]")
            if s5_layer.A_log.max() > 10:
                print("⚠️  A_log values are too large!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_nan()