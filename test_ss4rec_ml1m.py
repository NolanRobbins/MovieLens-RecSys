#!/usr/bin/env python3
"""
Test SS4Rec Model on ML-1M Dataset
===================================

This script tests the SS4Rec model on the standard RecBole ml-1m dataset
before attempting to use the custom ml-25m dataset. This helps validate
that the model implementation works correctly with a known dataset.

Usage:
    python test_ss4rec_ml1m.py [--config CONFIG_FILE] [--debug]

Example:
    python test_ss4rec_ml1m.py --config configs/official/ss4rec_ml1m_test.yaml --debug
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import time
import traceback

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_logging(debug=False):
    """Set up logging configuration"""
    log_level = logging.DEBUG if debug else logging.INFO
    log_format = '%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s'
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_dir / "ss4rec_ml1m_test.log", mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set console handler to show only important messages
    console_handler = logging.getLogger().handlers[1]
    if not debug:
        console_handler.setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info("🔧 Logging configured for SS4Rec ML-1M test")
    return logger

def check_dependencies():
    """Check if required dependencies are available"""
    logger = logging.getLogger(__name__)
    
    required_packages = [
        'torch',
        'recbole',
        'numpy',
        'pandas',
        'yaml',
        'kmeans_pytorch'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package} available")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"❌ {package} not available")
    
    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        logger.error("Please install requirements: pip install -r requirements_ss4rec.txt")
        return False
    
    # Check for SS4Rec specific dependencies
    try:
        import mamba_ssm
        logger.info(f"✅ mamba-ssm available (version: {mamba_ssm.__version__})")
    except ImportError:
        logger.warning("⚠️ mamba-ssm not available - SS4Rec may not work properly")
    
    try:
        import s5
        logger.info("✅ s5-pytorch available")
    except ImportError:
        logger.warning("⚠️ s5-pytorch not available - SS4Rec may not work properly")
    
    return True

def test_recbole_ml1m_download():
    """Test if RecBole can download and load ml-1m dataset"""
    logger = logging.getLogger(__name__)
    
    try:
        from recbole.config import Config
        from recbole.data import create_dataset, data_preparation
        from recbole.utils import init_seed
        import torch.distributed as dist
        import os
        
        # Initialize single-GPU distributed environment
        if not dist.is_initialized():
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '29500'
            os.environ['RANK'] = '0'
            os.environ['WORLD_SIZE'] = '1'
            dist.init_process_group(backend='gloo', rank=0, world_size=1)
            logger.info("✅ Initialized single-GPU distributed environment")
        
        logger.info("🧪 Testing RecBole ml-1m dataset download...")
        
        # Create a minimal config for testing
        config_dict = {
            'model': 'BPR',
            'dataset': 'ml-1m',
            'data_path': 'data/recbole_format',
            'download': True,
            'USER_ID_FIELD': 'user_id',
            'ITEM_ID_FIELD': 'item_id',
            'RATING_FIELD': 'rating',
            'TIME_FIELD': 'timestamp',
            'load_col': {
                'inter': ['user_id', 'item_id', 'rating', 'timestamp']
            },
            'eval_args': {
                'group_by': 'user',
                'split': {'RS': [0.8, 0.1, 0.1]},
                'order': 'RO',
                'mode': 'full'
            },
            'metrics': ['Recall', 'MRR', 'NDCG', 'Hit'],
            'topk': [10],
            'valid_metric': 'MRR@10',
            'train_batch_size': 1024,
            'eval_batch_size': 1024,
            'epochs': 1,  # Just test loading
            'device': 'cpu',  # Use CPU for testing
            'reproducibility': True,
            'seed': 2023,
            # Single-GPU distributed training parameters (required by RecBole)
            'nproc': 1,
            'world_size': 1,
            'offset': 0,
            'ip': 'localhost',
            'port': 29500,
            'backend': 'gloo'
        }
        
        config = Config(model='BPR', dataset='ml-1m', config_dict=config_dict)
        init_seed(config['seed'], config['reproducibility'])
        
        # Test dataset creation
        logger.info("📊 Creating dataset...")
        dataset = create_dataset(config)
        logger.info(f"✅ Dataset created successfully")
        logger.info(f"   Users: {dataset.user_num}")
        logger.info(f"   Items: {dataset.item_num}")
        logger.info(f"   Interactions: {dataset.inter_num}")
        
        # Test data preparation
        logger.info("🔄 Preparing data...")
        train_data, valid_data, test_data = data_preparation(config, dataset)
        logger.info("✅ Data preparation successful")
        logger.info(f"   Train batches: {len(train_data)}")
        logger.info(f"   Valid batches: {len(valid_data)}")
        logger.info(f"   Test batches: {len(test_data)}")
        
        # Cleanup distributed environment
        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info("✅ Cleaned up distributed environment")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ RecBole ml-1m test failed: {e}")
        logger.error(traceback.format_exc())
        
        # Cleanup distributed environment on error
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
                logger.info("✅ Cleaned up distributed environment after error")
        except:
            pass  # Ignore cleanup errors
        
        return False

def test_ss4rec_model_loading():
    """Test if SS4Rec model can be loaded"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("🧠 Testing SS4Rec model loading...")
        
        # Try to import the official SS4Rec model
        from models.official_ss4rec.ss4rec_official import SS4RecOfficial
        logger.info("✅ SS4RecOfficial model imported successfully")
        
        # Initialize single-GPU distributed environment
        import torch.distributed as dist
        import os
        
        if not dist.is_initialized():
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '29500'
            os.environ['RANK'] = '0'
            os.environ['WORLD_SIZE'] = '1'
            dist.init_process_group(backend='gloo', rank=0, world_size=1)
            logger.info("✅ Initialized single-GPU distributed environment for model test")
        
        # Create a minimal config for model testing
        from recbole.config import Config
        
        config_dict = {
            'model': 'SS4RecOfficial',
            'dataset': 'ml-1m',
            'data_path': 'data/recbole_format',
            'USER_ID_FIELD': 'user_id',
            'ITEM_ID_FIELD': 'item_id',
            'RATING_FIELD': 'rating',
            'TIME_FIELD': 'timestamp',
            'load_col': {
                'inter': ['user_id', 'item_id', 'rating', 'timestamp']
            },
            'hidden_size': 64,
            'n_layers': 2,
            'dropout_prob': 0.5,
            'loss_type': 'BPR',
            'd_state': 16,
            'd_conv': 4,
            'expand': 2,
            'dt_min': 0.001,
            'dt_max': 0.1,
            'learning_rate': 0.001,
            'train_batch_size': 1024,
            'eval_batch_size': 1024,
            'epochs': 1,
            'device': 'cpu',
            'reproducibility': True,
            'seed': 2023,
            'nproc': 1,
            'world_size': 1,
            'offset': 0,
            'ip': 'localhost',
            'port': 29500,
            'backend': 'gloo'
        }
        
        config = Config(model='SS4RecOfficial', dataset='ml-1m', config_dict=config_dict)
        logger.info("✅ Config created for SS4Rec model testing")
        
        # Test model initialization with proper config
        model = SS4RecOfficial(config=config, dataset=None)
        logger.info("✅ SS4RecOfficial model initialized successfully")
        
        # Cleanup distributed environment
        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info("✅ Cleaned up distributed environment")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ SS4Rec model loading failed: {e}")
        logger.error(traceback.format_exc())
        
        # Cleanup distributed environment on error
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
                logger.info("✅ Cleaned up distributed environment after error")
        except:
            pass  # Ignore cleanup errors
        
        return False

def run_ss4rec_ml1m_test(config_file, debug=False):
    """Run the actual SS4Rec test on ml-1m dataset"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("🚀 Starting SS4Rec ML-1M test...")
        logger.info(f"📄 Config file: {config_file}")
        
        # Import RecBole components
        from recbole.config import Config
        from recbole.data import create_dataset, data_preparation
        from recbole.utils import init_seed, get_model, get_trainer
        from recbole.utils.logger import init_logger
        import torch.distributed as dist
        import os
        
        # Initialize single-GPU distributed environment
        if not dist.is_initialized():
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '29500'
            os.environ['RANK'] = '0'
            os.environ['WORLD_SIZE'] = '1'
            dist.init_process_group(backend='gloo', rank=0, world_size=1)
            logger.info("✅ Initialized single-GPU distributed environment for SS4Rec test")
        
        # Load configuration
        config = Config(model='SS4RecOfficial', dataset='ml-1m', config_file_list=[config_file])
        init_seed(config['seed'], config['reproducibility'])
        
        # Initialize logger
        init_logger(config)
        
        logger.info("📊 Creating dataset...")
        dataset = create_dataset(config)
        logger.info(f"✅ Dataset: {dataset.user_num} users, {dataset.item_num} items, {dataset.inter_num} interactions")
        
        logger.info("🔄 Preparing data...")
        train_data, valid_data, test_data = data_preparation(config, dataset)
        
        logger.info("🧠 Initializing SS4Rec model...")
        model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
        logger.info(f"✅ Model initialized on {config['device']}")
        
        logger.info("🏃 Initializing trainer...")
        trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
        
        logger.info("🎯 Starting training (limited epochs for testing)...")
        start_time = time.time()
        
        # Run training for a few epochs
        best_valid_score, best_valid_result = trainer.fit(
            train_data, valid_data, verbose=True, saved=True, show_progress=True
        )
        
        training_time = time.time() - start_time
        logger.info(f"✅ Training completed in {training_time:.2f} seconds")
        logger.info(f"📊 Best validation score: {best_valid_score}")
        logger.info(f"📊 Best validation result: {best_valid_result}")
        
        logger.info("🧪 Running evaluation...")
        test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=True)
        logger.info(f"📊 Test results: {test_result}")
        
        # Cleanup distributed environment
        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info("✅ Cleaned up distributed environment")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ SS4Rec ML-1M test failed: {e}")
        logger.error(traceback.format_exc())
        
        # Cleanup distributed environment on error
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
                logger.info("✅ Cleaned up distributed environment after error")
        except:
            pass  # Ignore cleanup errors
        
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Test SS4Rec on ML-1M dataset')
    parser.add_argument('--config', default='configs/official/ss4rec_ml1m_test.yaml',
                       help='Configuration file path')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    parser.add_argument('--skip-deps', action='store_true',
                       help='Skip dependency checks')
    parser.add_argument('--skip-recbole-test', action='store_true',
                       help='Skip RecBole ml-1m download test')
    parser.add_argument('--skip-model-test', action='store_true',
                       help='Skip SS4Rec model loading test')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.debug)
    
    logger.info("🎬 SS4Rec ML-1M Test Starting")
    logger.info("=" * 50)
    logger.info(f"📄 Config: {args.config}")
    logger.info(f"🔍 Debug: {args.debug}")
    logger.info("=" * 50)
    
    # Check if config file exists
    if not Path(args.config).exists():
        logger.error(f"❌ Config file not found: {args.config}")
        return 1
    
    # Check dependencies
    if not args.skip_deps:
        logger.info("🔍 Checking dependencies...")
        if not check_dependencies():
            logger.error("❌ Dependency check failed")
            return 1
        logger.info("✅ All dependencies available")
    
    # Test RecBole ml-1m download
    if not args.skip_recbole_test:
        logger.info("🧪 Testing RecBole ml-1m dataset...")
        if not test_recbole_ml1m_download():
            logger.error("❌ RecBole ml-1m test failed")
            return 1
        logger.info("✅ RecBole ml-1m test passed")
    
    # Test SS4Rec model loading
    if not args.skip_model_test:
        logger.info("🧠 Testing SS4Rec model...")
        if not test_ss4rec_model_loading():
            logger.error("❌ SS4Rec model test failed")
            return 1
        logger.info("✅ SS4Rec model test passed")
    
    # Run the actual test
    logger.info("🚀 Running SS4Rec ML-1M test...")
    if run_ss4rec_ml1m_test(args.config, args.debug):
        logger.info("🎉 SS4Rec ML-1M test completed successfully!")
        logger.info("✅ Model is ready for ml-25m dataset")
        return 0
    else:
        logger.error("❌ SS4Rec ML-1M test failed")
        return 1

if __name__ == "__main__":
    exit(main())
