#!/usr/bin/env python3
"""
Official SS4Rec Training Script using RecBole Framework

This script trains SS4Rec using the official implementation with:
- RecBole 1.0 for standard sequential recommendation evaluation
- Official mamba-ssm and s5-pytorch libraries for numerical stability
- Standard RecSys metrics: HR@K, NDCG@K, MRR@K
- BPR loss for ranking task (as per paper)

Usage:
    python training/official/train_ss4rec_official.py --config configs/official/ss4rec_official.yaml
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Defer RecBole imports until after dependency installation
RECBOLE_AVAILABLE = False
SS4RecOfficial = None
create_ss4rec_config = None

def check_recbole_imports():
    """Check if RecBole imports are available"""
    global RECBOLE_AVAILABLE, SS4RecOfficial, create_ss4rec_config
    try:
        from recbole.quick_start import run_recbole
        from recbole.config import Config
        from recbole.data import create_dataset, data_preparation
        from recbole.trainer import Trainer
        from recbole.utils import init_seed, init_logger
        
        # Import our custom model
        from models.official_ss4rec.ss4rec_official import SS4RecOfficial as SS4RecOfficialClass, create_ss4rec_config as create_config
        SS4RecOfficial = SS4RecOfficialClass
        create_ss4rec_config = create_config
        RECBOLE_AVAILABLE = True
        return True
    except ImportError as e:
        print(f"❌ RecBole or SS4Rec import failed: {e}")
        print("Install with: uv pip install recbole==1.2.0")
        return False


def setup_logging(log_level: str = 'INFO', log_file: str = None):
    """Setup logging configuration"""
    level = getattr(logging, log_level.upper())
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def train_ss4rec_official(config_file: str, dataset_path: str = None, output_dir: str = None):
    """
    Train SS4Rec using official RecBole framework
    
    Args:
        config_file: Path to YAML configuration file
        dataset_path: Path to dataset directory (optional)
        output_dir: Path to output directory for results
    """
    # Check if RecBole imports are available now
    if not check_recbole_imports():
        raise ImportError("RecBole is required. Install with: uv pip install recbole==1.2.0")
    
    # Import RecBole modules now that we know they're available
    from recbole.quick_start import run_recbole
    from recbole.config import Config
    from recbole.utils import init_seed, init_logger
    
    logging.info("🚀 Starting Official SS4Rec Training")
    
    # Load configuration
    if os.path.exists(config_file):
        logging.info(f"📄 Loading config from: {config_file}")
        config = Config(model=SS4RecOfficial, config_file_list=[config_file])
    else:
        logging.warning(f"⚠️ Config file not found: {config_file}")
        logging.info("📄 Using default configuration")
        config = Config(model=SS4RecOfficial, config_dict=create_ss4rec_config())
    
    # Override paths if provided
    if dataset_path:
        config['data_path'] = dataset_path
    if output_dir:
        config['checkpoint_dir'] = output_dir
        
    # Initialize reproducibility
    init_seed(config['seed'], config['reproducibility'])
    
    # Initialize logger
    init_logger(config)
    
    logging.info("📊 Configuration Summary:")
    logging.info(f"  - Model: {config['model']}")
    logging.info(f"  - Dataset: {config['dataset']}")
    logging.info(f"  - Hidden Size: {config['hidden_size']}")
    logging.info(f"  - Layers: {config['n_layers']}")
    logging.info(f"  - Batch Size: {config['train_batch_size']}")
    logging.info(f"  - Learning Rate: {config['learning_rate']}")
    logging.info(f"  - Epochs: {config['epochs']}")
    logging.info(f"  - Device: {config['device']}")
    
    try:
        # Run training using RecBole's standard pipeline
        logging.info("🏃 Starting RecBole training pipeline...")
        result = run_recbole(
            model=SS4RecOfficial,
            dataset=config['dataset'],
            config_dict=config.final_config_dict
        )
        
        logging.info("🎉 Training completed successfully!")
        logging.info("📊 Final Results:")
        
        if 'test_result' in result:
            test_results = result['test_result']
            for metric, value in test_results.items():
                logging.info(f"  - {metric}: {value:.6f}")
                
            # Check if we achieved paper benchmarks
            hr_10 = test_results.get('hit@10', 0)
            ndcg_10 = test_results.get('ndcg@10', 0)
            
            logging.info("\n🎯 Performance vs Paper Benchmarks:")
            logging.info(f"  - HR@10: {hr_10:.6f} (Target: >0.30)")
            logging.info(f"  - NDCG@10: {ndcg_10:.6f} (Target: >0.25)")
            
            if hr_10 > 0.30:
                logging.info("✅ HR@10 benchmark achieved!")
            if ndcg_10 > 0.25:
                logging.info("✅ NDCG@10 benchmark achieved!")
        
        return result
        
    except Exception as e:
        logging.error(f"❌ Training failed: {e}")
        logging.error("Full error details:", exc_info=True)
        raise


def main():
    parser = argparse.ArgumentParser(description='Train Official SS4Rec Model')
    
    parser.add_argument(
        '--config', 
        type=str,
        default='configs/official/ss4rec_official.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        default=None,
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str, 
        default='results/official_ss4rec',
        help='Output directory for results'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = output_dir / f"ss4rec_official_training.log"
    setup_logging(args.log_level, str(log_file))
    
    logging.info("🚀 Official SS4Rec Training Script")
    logging.info(f"📄 Config: {args.config}")
    logging.info(f"📂 Output: {args.output_dir}")
    logging.info(f"📝 Log: {log_file}")
    
    # Check dependencies - try to import RecBole now
    if not check_recbole_imports():
        logging.error("❌ RecBole not available")
        logging.error("Install with: uv pip install recbole==1.2.0")
        return 1
    
    try:
        # Import test to check other dependencies
        from mamba_ssm import Mamba
        from s5 import S5
        logging.info("✅ All dependencies available")
    except ImportError as e:
        logging.error(f"❌ Missing dependency: {e}")
        logging.error("Install with: uv pip install mamba-ssm==2.2.2 s5-pytorch==0.2.1")
        return 1
    
    try:
        # Train model
        result = train_ss4rec_official(
            config_file=args.config,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir
        )
        
        logging.info("🎉 Training completed successfully!")
        return 0
        
    except Exception as e:
        logging.error(f"❌ Training failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)