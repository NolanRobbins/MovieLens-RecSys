#!/usr/bin/env python3
"""
RunPod Training Script for Official SS4Rec Implementation

This script provides RunPod-optimized training for official SS4Rec using:
- RecBole 1.0 framework for standard evaluation
- Official mamba-ssm and s5-pytorch libraries  
- A6000 GPU optimization
- Discord notifications and W&B logging
- Automatic dependency installation

Usage on RunPod:
    python training/official/runpod_train_ss4rec_official.py --config configs/official/ss4rec_official.yaml
"""

import os
import sys
import argparse
import subprocess
import logging
import time
import requests
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import shutil  # Added for file moving
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


# Inject our SequentialDataset override before any RecBole import
def inject_sequential_dataset_override() -> None:
    try:
        import importlib.util
        override_path = project_root / 'models' / 'official_ss4rec' / 'sequential_dataset_official.py'
        if override_path.exists():
            spec = importlib.util.spec_from_file_location(
                'recbole.data.sequential_dataset', str(override_path)
            )
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            sys.modules['recbole.data.sequential_dataset'] = module
            logging.info('✅ Injected SequentialDataset override for RecBole')
        else:
            logging.warning(f'SequentialDataset override not found: {override_path}')
    except Exception as e:
        logging.warning(f'Failed to inject SequentialDataset override: {e}')


def setup_logging(log_file: Optional[str] = None):
    """Setup logging configuration"""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def send_discord_notification(message: str, webhook_url: str, color: int = 5763719) -> bool:
    """Send Discord notification"""
    try:
        payload = {
            "embeds": [{
                "title": "🎬 Official SS4Rec Training Update",
                "description": message,
                "color": color,
                "timestamp": datetime.utcnow().isoformat(),
                "footer": {"text": "RunPod A6000 - Official Implementation"}
            }]
        }
        response = requests.post(webhook_url, json=payload, timeout=10)
        return response.status_code == 204
    except Exception as e:
        logging.warning(f"Discord notification failed: {e}")
        return False


def check_dependencies() -> bool:
    """Check if all required dependencies are installed"""
    # Ensure dataset override is active before importing RecBole
    inject_sequential_dataset_override()
    required_packages = [
        ('torch', 'PyTorch'),
        ('mamba_ssm', 'Mamba SSM'),
        ('causal_conv1d', 'Causal Conv1D'),
        ('wandb', 'Weights & Biases'),
        ('ray', 'Ray'),
        ('hyperopt', 'Hyperopt')
    ]
    
    missing_packages = []
    for package, name in required_packages:
        try:
            __import__(package)
            logging.info(f"✅ {name} available")
        except ImportError:
            missing_packages.append((package, name))
            logging.error(f"❌ {name} not available")
    
    # Check RecBole separately - it needs special handling
    try:
        import recbole
        logging.info("✅ RecBole available")
    except ImportError:
        missing_packages.append(('recbole', 'RecBole'))
        logging.error("❌ RecBole not available")
    
    return len(missing_packages) == 0


def install_dependencies():
    """Install required dependencies for official SS4Rec"""
    logging.info("📦 Installing official SS4Rec dependencies...")
    
    try:
        # Check if uv is available (preferred method)
        uv_available = subprocess.run(['uv', '--version'], capture_output=True).returncode == 0
        
        requirements_file = project_root / 'requirements_ss4rec.txt'
        if not requirements_file.exists():
            logging.error(f"❌ Requirements file not found: {requirements_file}")
            return False
        
        if uv_available:
            # Use uv pip for faster installation
            logging.info("Using uv for dependency installation...")
            cmd = ['uv', 'pip', 'install', '-r', str(requirements_file)]
        else:
            # Fallback to regular pip, but first ensure pip is available
            try:
                # Try to import pip to check if it's available
                import pip
                cmd = [sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)]
            except ImportError:
                # If pip is not available, try to bootstrap it
                logging.warning("pip not available, attempting to bootstrap...")
                subprocess.run([sys.executable, '-m', 'ensurepip', '--default-pip'], 
                              capture_output=True, check=True)
                cmd = [sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)]
        
        # Run installation command
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logging.info("✅ Dependencies installed successfully")
        return True
            
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ Failed to install dependencies: {e}")
        logging.error(f"Stdout: {e.stdout}")
        logging.error(f"Stderr: {e.stderr}")
        return False
    except Exception as e:
        logging.error(f"❌ Unexpected error during dependency installation: {e}")
        return False


def prepare_data():
    """Prepare data in RecBole format"""
    logging.info("📊 Preparing data in RecBole format...")
    
    try:
        # Define the target path in RecBole-required structure: {data_path}/{dataset}/{dataset}.inter
        dataset_dir = project_root / 'data' / 'recbole_format' / 'movielens'
        recbole_data = dataset_dir / 'movielens.inter'
        
        if recbole_data.exists():
            logging.info("✅ RecBole format data already exists in correct structure")
            return True
        
        # Create directories if needed
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Run data conversion (assumes converter creates movielens.inter in data/recbole_format)
        converter_script = project_root / 'data' / 'recbole_format' / 'movielens_adapter.py'
        if not converter_script.exists():
            logging.error(f"❌ Converter script not found: {converter_script}")
            return False
            
        cmd = [sys.executable, str(converter_script)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # After conversion, move the generated file to the dataset subdirectory
        generated_file = project_root / 'data' / 'recbole_format' / 'movielens.inter'
        if generated_file.exists():
            shutil.move(str(generated_file), str(recbole_data))
            logging.info(f"✅ Moved {generated_file.name} to {dataset_dir}")
        else:
            logging.error(f"❌ Converted file not found: {generated_file}")
            return False
        
        logging.info("✅ Data preparation completed")
        return True
        
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ Data preparation failed: {e}")
        logging.error(f"Stdout: {e.stdout}")
        logging.error(f"Stderr: {e.stderr}")
        return False


def parse_dataset_from_config(config_path: str) -> Dict[str, Any]:
    """Parse dataset name and download flag from YAML config."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
        dataset = str(cfg.get('dataset', '')).strip()
        download = bool(cfg.get('download', False))
        data_path = str(cfg.get('data_path', '')).strip()
        return {'dataset': dataset, 'download': download, 'data_path': data_path}
    except Exception as e:
        logging.warning(f"Failed to parse config {config_path}: {e}")
        return {'dataset': '', 'download': False, 'data_path': ''}


def run_official_ss4rec_training(config_file: str, output_dir: str) -> Dict[str, Any]:
    """
    Run official SS4Rec training using RecBole
    
    Args:
        config_file: Path to configuration file
        output_dir: Output directory for results
        
    Returns:
        results: Training results dictionary
    """
    logging.info("🚀 Starting official SS4Rec training...")
    
    try:
        # Ensure dataset override is active during training
        inject_sequential_dataset_override()
        # Import training function
        from training.official.train_ss4rec_official import train_ss4rec_official
        
        # Run training
        results = train_ss4rec_official(
            config_file=config_file,
            output_dir=output_dir
        )
        
        logging.info("✅ Training completed successfully")
        return results
        
    except Exception as e:
        logging.error(f"❌ Training failed: {e}")
        logging.error("Full error details:", exc_info=True)
        raise


def main():
    parser = argparse.ArgumentParser(description='RunPod Official SS4Rec Training')
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/official/ss4rec_official.yaml',
        help='Configuration file path'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/official_ss4rec',
        help='Output directory for results'
    )
    parser.add_argument(
        '--install-deps',
        action='store_true',
        help='Install dependencies before training'
    )
    parser.add_argument(
        '--prepare-data',
        action='store_true',
        help='Prepare data in RecBole format'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = output_dir / f"runpod_ss4rec_official_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(str(log_file))
    
    # Get Discord webhook
    discord_webhook = os.environ.get('DISCORD_WEBHOOK_URL', '')
    
    logging.info("🎬 RunPod Official SS4Rec Training")
    logging.info(f"📄 Config: {args.config}")
    logging.info(f"📂 Output: {args.output_dir}")
    logging.info(f"📝 Log: {log_file}")
    
    start_time = time.time()
    
    try:
        # Ensure dataset override is injected at startup
        inject_sequential_dataset_override()
        # Send start notification
        if discord_webhook:
            start_message = f"""
🚀 **Official SS4Rec Training Started**

📊 **Implementation**: RecBole + Official Libraries
🖥️ **System**: RunPod A6000 GPU
⏰ **Started**: {datetime.now().strftime('%H:%M:%S UTC')}

🎯 **Expected Performance**:
- HR@10: > 0.30 (paper benchmark)
- NDCG@10: > 0.25 (paper benchmark)
- Training Time: 4-8 hours

🔧 **Architecture**: Time-Aware SSM + Relation-Aware SSM
            """.strip()
            
            send_discord_notification(start_message, discord_webhook, color=3447003)
        
        # Install dependencies if requested
        if args.install_deps:
            if not install_dependencies():
                raise RuntimeError("Failed to install dependencies")
        
        # Check dependencies
        if not check_dependencies():
            logging.error("❌ Missing required dependencies")
            logging.info("💡 Run with --install-deps to install automatically")
            return 1
        
        # Prepare data if requested  
        if args.prepare_data:
            if not prepare_data():
                raise RuntimeError("Failed to prepare data")
        
        # Check if data exists (updated to check the subdirectory structure)
        dataset_dir = project_root / 'data' / 'recbole_format' / 'movielens'
        recbole_data = dataset_dir / 'movielens.inter'
        if not recbole_data.exists():
            logging.error("❌ RecBole format data not found in expected structure")
            logging.info("💡 Run with --prepare-data to convert data automatically")
            return 1
        
        # Run training
        results = run_official_ss4rec_training(args.config, args.output_dir)
        
        # Calculate training time
        duration_hours = (time.time() - start_time) / 3600
        
        # Send success notification
        if discord_webhook:
            hr_10 = results.get('test_result', {}).get('hit@10', 0)
            ndcg_10 = results.get('test_result', {}).get('ndcg@10', 0)
            
            success_message = f"""
🎉 **Official SS4Rec Training Complete!**

⏱️ **Duration**: {duration_hours:.1f} hours
🎯 **HR@10**: {hr_10:.6f} {'✅ (Benchmark achieved!)' if hr_10 > 0.30 else '(Good progress)'}
🎯 **NDCG@10**: {ndcg_10:.6f} {'✅ (Benchmark achieved!)' if ndcg_10 > 0.25 else '(Good progress)'}

🏆 **Status**: {'SOTA Performance Achieved!' if hr_10 > 0.30 and ndcg_10 > 0.25 else 'Good Results!'}
💾 **Results**: `{args.output_dir}`

✅ **Instance can now be safely terminated.**
            """.strip()
            
            color = 5763719 if hr_10 > 0.30 and ndcg_10 > 0.25 else 16776960
            send_discord_notification(success_message, discord_webhook, color)
        
        logging.info("🎉 Official SS4Rec training completed successfully!")
        return 0
        
    except Exception as e:
        # Calculate failed duration
        duration_hours = (time.time() - start_time) / 3600
        
        # Send failure notification
        if discord_webhook:
            failure_message = f"""
❌ **Official SS4Rec Training Failed**

⏱️ **Duration**: {duration_hours:.1f} hours
❌ **Error**: {str(e)[:200]}{'...' if len(str(e)) > 200 else ''}

📝 **Check log**: {log_file.name}
🔧 **Common fixes**:
• Check dependencies: --install-deps
• Check data: --prepare-data
• Verify GPU memory availability

🔄 **Ready for restart with fixes applied**
            """.strip()
            
            send_discord_notification(failure_message, discord_webhook, color=15158332)
        
        logging.error(f"❌ Training failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)