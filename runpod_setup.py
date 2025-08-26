"""
RunPod Setup Script for Neural CF Training
Automated environment setup for A6000 GPU training
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
import yaml
import torch

def setup_logging():
    """Setup logging for setup process"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

def install_requirements():
    """Install required packages with uv for speed"""
    logger = logging.getLogger(__name__)
    
    try:
        # Check if uv is available, install if not
        subprocess.run(['uv', '--version'], check=True, capture_output=True)
        logger.info("✅ uv package manager found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.info("📦 Installing uv package manager...")
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', 'uv'
        ], check=True)
    
    # Create virtual environment
    logger.info("🐍 Creating virtual environment...")
    subprocess.run(['uv', 'venv'], check=True)
    
    # Activate and install requirements
    logger.info("📦 Installing dependencies...")
    
    # Base requirements
    base_reqs = [
        'torch>=2.0.0',
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scikit-learn>=1.0.0',
        'pyyaml>=6.0',
        'tqdm>=4.60.0',
        'wandb>=0.15.0',
        'plotly>=5.0.0',
        'requests>=2.25.0',
        'sqlalchemy>=1.4.0',
        'diskcache>=5.4.0'
    ]
    
    # Install with uv
    for req in base_reqs:
        try:
            subprocess.run(['uv', 'pip', 'install', req], check=True)
            logger.info(f"✅ Installed {req}")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install {req}: {e}")
            raise

def verify_gpu_setup():
    """Verify GPU availability and configuration"""
    logger = logging.getLogger(__name__)
    
    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available! This setup requires GPU.")
        return False
    
    gpu_count = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    logger.info(f"🚀 GPU Setup:")
    logger.info(f"   Device Count: {gpu_count}")
    logger.info(f"   Device Name: {gpu_name}")
    logger.info(f"   Memory: {gpu_memory:.1f} GB")
    
    # Verify A6000 optimization compatibility
    if "A6000" in gpu_name or gpu_memory > 40:
        logger.info("✅ A6000 optimizations enabled")
        return True
    else:
        logger.warning("⚠️  Non-A6000 GPU detected. Performance may vary.")
        return True

def setup_data_directories():
    """Create necessary data directories"""
    logger = logging.getLogger(__name__)
    
    directories = [
        'data/processed',
        'results/ncf_baseline',
        'logs/ncf_baseline',
        'models/checkpoints'
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Created directory: {dir_path}")

def setup_wandb():
    """Setup Weights & Biases integration"""
    logger = logging.getLogger(__name__)
    
    # Check if WANDB_API_KEY is set
    if 'WANDB_API_KEY' not in os.environ:
        logger.warning("⚠️  WANDB_API_KEY not set in environment")
        logger.info("Set your W&B API key: export WANDB_API_KEY=<your-key>")
        return False
    
    try:
        import wandb
        wandb.login()
        logger.info("✅ W&B authentication successful")
        return True
    except Exception as e:
        logger.error(f"❌ W&B setup failed: {e}")
        return False

def create_runpod_config():
    """Create RunPod-optimized configuration"""
    logger = logging.getLogger(__name__)
    
    runpod_config = {
        'model': {
            'type': 'neural_cf',
            'mf_dim': 64,
            'mlp_dims': [256, 128, 64],  # Increased for A6000
            'dropout_rate': 0.2
        },
        'training': {
            'batch_size': 4096,  # Optimized for A6000 48GB VRAM
            'learning_rate': 0.001,
            'epochs': 150,
            'early_stopping': {
                'patience': 20,
                'min_delta': 0.0005
            }
        },
        'optimizer': {
            'type': 'adamw',  # Better for large models
            'weight_decay': 1e-4,
            'betas': [0.9, 0.999]
        },
        'scheduler': {
            'type': 'cosine_annealing',
            'T_max': 150,
            'eta_min': 1e-7
        },
        'data': {
            'dataset': 'movielens-25m',
            'train_ratio': 0.8,
            'val_ratio': 0.1,
            'test_ratio': 0.1,
            'min_interactions': 20
        },
        'evaluation': {
            'metrics': ['rmse', 'mae', 'hr@10', 'ndcg@10', 'mrr@10'],
            'protocol': 'leave_one_out'
        },
        'gpu': {
            'device': 'cuda',
            'mixed_precision': True,
            'compile': True,
            'pin_memory': True,
            'non_blocking': True
        },
        'logging': {
            'log_level': 'INFO',
            'save_model': True,
            'checkpoint_freq': 5,
            'wandb_project': 'movielens-ncf-runpod'
        },
        'paths': {
            'data_dir': 'data/processed',
            'model_dir': 'results/ncf_runpod',
            'logs_dir': 'logs/ncf_runpod'
        }
    }
    
    # Save RunPod-specific config
    config_path = Path('configs/ncf_runpod.yaml')
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(runpod_config, f, default_flow_style=False, indent=2)
    
    logger.info(f"✅ Created RunPod config: {config_path}")
    return config_path

def main():
    """Main setup function"""
    logger = setup_logging()
    
    logger.info("🚀 Starting RunPod Neural CF Setup")
    logger.info("=" * 50)
    
    try:
        # 1. Install requirements
        logger.info("📦 Step 1: Installing requirements...")
        install_requirements()
        
        # 2. Verify GPU
        logger.info("🔍 Step 2: Verifying GPU setup...")
        if not verify_gpu_setup():
            sys.exit(1)
        
        # 3. Setup directories
        logger.info("📁 Step 3: Creating directories...")
        setup_data_directories()
        
        # 4. Setup W&B
        logger.info("📊 Step 4: Setting up W&B...")
        wandb_ready = setup_wandb()
        
        # 5. Create config
        logger.info("⚙️  Step 5: Creating RunPod config...")
        config_path = create_runpod_config()
        
        # 6. Setup complete
        logger.info("=" * 50)
        logger.info("✅ RunPod setup complete!")
        logger.info(f"🎯 Config file: {config_path}")
        logger.info("🏃 Ready to train Neural CF model")
        
        if wandb_ready:
            logger.info("📊 W&B logging enabled")
        else:
            logger.info("⚠️  W&B disabled - set WANDB_API_KEY to enable")
        
        logger.info("\n🚀 To start training:")
        logger.info(f"python runpod_training_wandb.py --model ncf --config {config_path}")
        
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()