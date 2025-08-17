#!/usr/bin/env python3
"""
Setup Model Versioning and Download Current Model
"""

import os
import json
import requests
from pathlib import Path
import sys

# Add src to path
sys.path.append('src')

try:
    from versioning.model_version_manager import ModelVersionManager, ModelType, register_current_model, create_next_experiment
except ImportError:
    print("❌ Could not import version manager. Make sure you're in the project root.")
    sys.exit(1)

def create_directories():
    """Create necessary directories"""
    dirs = [
        "models/downloaded",
        "models/versions", 
        "models/current",
        "models/archive"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✅ Created model directories")

def create_mock_current_model():
    """Create a mock model file for the current experiment"""
    # Since we can't easily download from W&B without auth, 
    # let's create a placeholder for now
    
    models_dir = Path("models/downloaded")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Create mock model file (you'll replace this with the real download)
    mock_model_path = models_dir / "current_model_rmse_0.806.pt"
    
    # Create a minimal PyTorch model file structure
    import torch
    
    # Mock model state dict (minimal structure)
    mock_state = {
        'experiment_info': {
            'rmse': 0.806,
            'experiment_id': 'a100-experiment-0815-1805-rmse0.806',
            'wandb_run_id': '2uw7m29i',
            'timestamp': '2025-08-15T18:05:01'
        },
        'model_state_dict': {
            # Mock parameters (replace with real model when downloaded)
            'user_embedding.weight': torch.randn(200948, 150),
            'movie_embedding.weight': torch.randn(84432, 150)
        },
        'hyperparameters': {
            'n_factors': 150,
            'hidden_dims': [512, 256],
            'latent_dim': 64,
            'dropout_rate': 0.3,
            'batch_size': 1024,
            'lr': 5e-4
        }
    }
    
    # Save mock model
    torch.save(mock_state, mock_model_path)
    
    print(f"📄 Created mock model file: {mock_model_path}")
    print("⚠️  Note: This is a placeholder. Download the real model using:")
    print("   - W&B: wandb artifact download nolanrobbins/movielens-hybrid-vae-a100/model:latest")
    print("   - Or use the RunPod download script")
    
    return str(mock_model_path)

def setup_model_versioning():
    """Setup complete model versioning system"""
    print("🔬 Setting up Model Versioning System")
    print("=" * 50)
    
    # Create directories
    create_directories()
    
    # Create mock current model (replace with real download)
    mock_model_path = create_mock_current_model()
    
    # Register current model in version system
    print("\n📝 Registering current model...")
    current_exp_id = register_current_model()
    
    # Create next experiment
    print("\n🚀 Creating next experiment...")
    next_exp_id = create_next_experiment()
    
    # Show experiment summary
    print("\n📊 Experiment Summary:")
    manager = ModelVersionManager()
    summary = manager.get_experiment_summary()
    print(summary.to_string(index=False))
    
    # Set current model as best (for now)
    try:
        manager.set_current_best(current_exp_id)
    except Exception as e:
        print(f"⚠️  Could not set current best: {e}")
    
    print(f"\n✅ Versioning system setup complete!")
    print(f"📁 Models directory: {Path('models').absolute()}")
    print(f"🔄 Current experiment: {current_exp_id}")
    print(f"🚀 Next experiment ready: {next_exp_id}")
    
    return current_exp_id, next_exp_id

def show_download_instructions():
    """Show instructions for downloading the real model"""
    print("\n" + "=" * 60)
    print("📥 NEXT STEPS: Download Your Real Model")
    print("=" * 60)
    
    print("""
Choose one of these methods to download your real trained model:

1. WEIGHTS & BIASES (Recommended):
   ```bash
   pip install wandb
   wandb login
   wandb artifact download nolanrobbins/movielens-hybrid-vae-a100/model:latest
   # Move downloaded files to models/downloaded/
   ```

2. RUNPOD SSH:
   ```bash
   scp root@YOUR_RUNPOD_IP:/workspace/models/hybrid_vae_best.pt models/downloaded/current_model_rmse_0.806.pt
   ```

3. MANUAL DOWNLOAD:
   - Open RunPod web interface
   - Navigate to /workspace/models/
   - Download hybrid_vae_best.pt
   - Save as models/downloaded/current_model_rmse_0.806.pt

After downloading the real model, the versioning system will automatically track it!
""")

if __name__ == "__main__":
    try:
        current_exp_id, next_exp_id = setup_model_versioning()
        show_download_instructions()
        
        print(f"\n🎯 Ready for next experiment!")
        print(f"Run: python src/models/enhanced_training_v2.py --experiment-id {next_exp_id}")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()