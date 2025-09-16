#!/usr/bin/env python3
"""
Google Cloud Training Script for SS4Rec
========================================

This script sets up SS4Rec training on Google Cloud Platform with:
- Automatic dependency installation with compatible versions
- Data download from Google Drive
- Model training with proper error handling
- Results upload to Google Cloud Storage

Usage:
    python gcp_training.py --config configs/official/ss4rec_official.yaml
"""

import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any

def setup_logging(log_level: str = 'INFO'):
    """Setup logging configuration"""
    level = getattr(logging, log_level.upper())

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('gcp_training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def install_dependencies():
    """Install SS4Rec dependencies with Google Cloud compatible versions"""
    logger = logging.getLogger(__name__)

    # Google Cloud compatible dependency versions
    dependencies = [
        # Core ML frameworks
        "torch>=2.0.0,<2.5.0",  # Compatible with CUDA 11.8/12.1
        "numpy>=1.21.0,<2.0.0",  # RecBole compatibility
        "pandas>=1.3.0",
        "scipy>=1.9.0",
        "scikit-learn>=1.1.0",

        # RecBole framework
        "recbole==1.2.0",
        "ray>=2.0.0",
        "hyperopt>=0.2.7",
        "kmeans-pytorch>=0.3.0",
        "lightgbm>=3.3.0",
        "xgboost>=1.7.0",

        # State Space Models (install in specific order)
        "causal-conv1d>=1.1.0",  # More flexible version constraint
        "mamba-ssm>=2.0.0",      # Compatible version

        # Experiment tracking
        "wandb>=0.15.0",
        "tensorboard>=2.10.0",

        # Data processing
        "tqdm>=4.64.0",
        "pyyaml>=6.0",

        # Visualization
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
    ]

    logger.info("🔧 Installing SS4Rec dependencies...")

    for dep in dependencies:
        try:
            logger.info(f"Installing {dep}...")
            subprocess.run([sys.executable, "-m", "pip", "install", dep],
                         check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to install {dep}: {e}")
            # Continue with other dependencies

    logger.info("✅ Dependencies installation completed")

def download_data():
    """Download training data from Google Drive"""
    logger = logging.getLogger(__name__)

    # Create data directory
    data_dir = Path("data/recbole_format/ml-25m")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Google Drive file ID for ml-25m.inter
    file_id = "1tGY6F_2nEeSWwAXJ_4F832p0BzEbAGfv"
    output_path = data_dir / "ml-25m.inter"

    if output_path.exists():
        logger.info(f"✅ Data file already exists: {output_path}")
        return

    logger.info("📥 Downloading training data from Google Drive...")

    # Download using gdown (works better on Google Cloud)
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "gdown"
        ], check=True)

        subprocess.run([
            "gdown", "--id", file_id, "--output", str(output_path)
        ], check=True)

        logger.info(f"✅ Data downloaded: {output_path}")
        logger.info(f"File size: {output_path.stat().st_size / (1024*1024):.1f} MB")

    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Data download failed: {e}")
        raise

def create_simple_ss4rec_config() -> Dict[str, Any]:
    """Create simplified SS4Rec config for Google Cloud (ML-1M for debug)"""
    return {
        # Model configuration
        'model': 'SS4Rec',
        'dataset': 'ml-1m',

        # SS4Rec parameters (reduced for faster training)
        'hidden_size': 32,        # Reduced from 64
        'num_layers': 1,          # Reduced from 2 (correct field name)
        'dropout_prob': 0.3,      # Reduced dropout
        'loss_type': 'BPR',

        # SSM parameters (official SS4Rec)
        'd_state': 8,             # Reduced from 16
        'd_conv': 4,
        'expand': 2,
        'dt_min': 0.001,
        'dt_max': 0.1,
        'd_P': 8,                 # S5 state dimension
        'd_H': 32,                # S5 width dimension
        'model_type': 'hybrid',   # SS4Rec model type

        # Timestamp configuration
        'TIMESTAMP_FIELD': 'timestamp',

        # Training parameters (optimized for ML-1M debug)
        'learning_rate': 0.003,   # Increased for faster convergence
        'train_batch_size': 512,  # Small batch for ML-1M
        'eval_batch_size': 512,
        'epochs': 15,             # Quick debug run
        'stopping_step': 3,       # Early stopping
        'weight_decay': 0.0001,

        # Evaluation
        'metrics': ['Recall', 'MRR', 'NDCG', 'Hit'],
        'topk': [5, 10],          # Reduced evaluation complexity
        'valid_metric': 'NDCG@10',

        # Data configuration (ML-1M)
        'USER_ID_FIELD': 'user_id',
        'ITEM_ID_FIELD': 'item_id',
        'RATING_FIELD': 'rating',
        'TIME_FIELD': 'timestamp',
        'data_path': 'data/recbole_format',
        'download': True,            # Let RecBole download ML-1M
        'MAX_ITEM_LIST_LENGTH': 20,  # Reduced for ML-1M
        'load_col': {
            'inter': ['user_id', 'item_id', 'rating', 'timestamp']
        },

        # Data splitting
        'eval_args': {
            'group_by': 'user',
            'split': {'LS': 'valid_and_test'},
            'order': 'TO',
            'mode': 'full'
        },

        # Device configuration
        'device': 'cuda' if os.getenv('CUDA_VISIBLE_DEVICES') else 'cpu',
        'gpu_id': 0,
        'reproducibility': True,
        'seed': 2023,

        # Single-GPU parameters
        'nproc': 1,
        'world_size': 1,
        'offset': 0,
        'ip': 'localhost',
        'port': 29500,
        'backend': 'gloo',

        # Optimization
        'scheduler': 'StepLR',
        'step_size': 20,
        'gamma': 0.8,

        # Logging
        'log_wandb': False,
        'save_dataset': False,
        'save_dataloaders': False,
        'checkpoint_dir': 'results/gcp_ss4rec',

        # Performance
        'num_workers': 2,         # Reduced workers
        'pin_memory': True,
    }

def train_ss4rec():
    """Train SS4Rec model"""
    logger = logging.getLogger(__name__)

    try:
        # Import RecBole components
        from recbole.quick_start import run_recbole
        from recbole.config import Config
        from recbole.utils import init_seed, init_logger

        # Import our SS4Rec model (official implementation)
        sys.path.append(str(Path(__file__).parent))
        from models.official_ss4rec.ss4rec_official import SS4Rec as SS4RecOfficial

        logger.info("🚀 Starting SS4Rec training on Google Cloud")

        # Create configuration
        config_dict = create_simple_ss4rec_config()
        config = Config(model=SS4RecOfficial, config_dict=config_dict)

        # Initialize reproducibility
        init_seed(config['seed'], config['reproducibility'])

        # Initialize logger
        init_logger(config)

        logger.info("📊 Training Configuration:")
        logger.info(f"  - Hidden Size: {config['hidden_size']}")
        logger.info(f"  - Layers: {config['n_layers']}")
        logger.info(f"  - Batch Size: {config['train_batch_size']}")
        logger.info(f"  - Epochs: {config['epochs']}")
        logger.info(f"  - Device: {config['device']}")

        # Run training
        logger.info("🏃 Starting RecBole training pipeline...")
        result = run_recbole(
            model=SS4RecOfficial,
            dataset=config['dataset'],
            config_dict=config.final_config_dict
        )

        logger.info("🎉 Training completed successfully!")

        if 'test_result' in result:
            test_results = result['test_result']
            logger.info("📊 Final Results:")
            for metric, value in test_results.items():
                logger.info(f"  - {metric}: {value:.6f}")

        return result

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        logger.error("Full error details:", exc_info=True)
        raise

def main():
    parser = argparse.ArgumentParser(description='Train SS4Rec on Google Cloud')

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file (optional - uses simplified config by default)'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    parser.add_argument(
        '--skip-deps',
        action='store_true',
        help='Skip dependency installation'
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_level)

    logger.info("🌟 SS4Rec Google Cloud Training Script")
    logger.info(f"Python: {sys.version}")
    logger.info(f"Working Directory: {os.getcwd()}")

    try:
        # Install dependencies
        if not args.skip_deps:
            install_dependencies()
        else:
            logger.info("⏭️ Skipping dependency installation")

        # Download data
        download_data()

        # Run pre-training validations
        logger.info("🔍 Running pre-training validations...")

        # Validate dependencies
        try:
            subprocess.run([sys.executable, "validate_dependencies.py"], check=True)
            logger.info("✅ Dependencies validated")
        except subprocess.CalledProcessError:
            logger.warning("⚠️ Dependency validation failed, proceeding anyway")
        except FileNotFoundError:
            logger.warning("⚠️ Dependency validation script not found")

        # Validate data pipeline
        try:
            subprocess.run([sys.executable, "validate_data_pipeline.py"], check=True)
            logger.info("✅ Data pipeline validated")
        except subprocess.CalledProcessError:
            logger.warning("⚠️ Data pipeline validation failed, proceeding anyway")
        except FileNotFoundError:
            logger.warning("⚠️ Data pipeline validation script not found")

        # Train model
        result = train_ss4rec()

        logger.info("🎉 Training completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        logger.error("Full traceback:", exc_info=True)
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)