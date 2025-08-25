"""
Unified RunPod Training Script with W&B Integration
Compatible with auto_train_with_notification.py Discord integration

Supports both Neural CF and SS4Rec training with automatic model selection
"""

import os
import sys
import argparse
import yaml
import torch
import wandb
from pathlib import Path
import logging
import time
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import our training modules
from training.train_ncf import NCFTrainer


def setup_wandb(config: dict, model_type: str) -> None:
    """Initialize Weights & Biases logging"""
    wandb.init(
        project=f"movielens-{model_type}",
        name=f"{model_type}-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config=config,
        tags=[model_type, "runpod", "a6000"]
    )


def main():
    parser = argparse.ArgumentParser(description="RunPod Training with W&B")
    parser.add_argument('--model', type=str, choices=['ncf', 'ss4rec'], 
                       default='ncf', help='Model to train')
    parser.add_argument('--config', type=str, default=None,
                       help='Custom config file (optional)')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable W&B logging')
    parser.add_argument('--debug', action='store_true',
                       help='Enable comprehensive debug logging for NaN detection')
    
    args = parser.parse_args()
    
    print(f"""
🎬 MovieLens RecSys Training Started
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Model: {args.model.upper()}
🖥️  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}
🔍 Debug Mode: {'ENABLED - NaN detection active' if args.debug else 'Disabled'}
📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
    
    # Determine config file
    if args.config:
        config_path = args.config
    else:
        config_path = f"configs/{args.model}_baseline.yaml" if args.model == 'ncf' else f"configs/{args.model}.yaml"
    
    # Check if config exists
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        print("Available configs:")
        for config_file in Path("configs").glob("*.yaml"):
            print(f"  - {config_file}")
        sys.exit(1)
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add debug mode to config
    config['debug_mode'] = args.debug
    if args.debug:
        print("🔍 Debug mode enabled - comprehensive logging and NaN detection active")
    
    # Initialize W&B if enabled (NCF only - SS4Rec handles its own W&B)
    if not args.no_wandb and args.model == 'ncf':
        try:
            setup_wandb(config, args.model)
            print("✅ W&B initialized successfully")
        except Exception as e:
            print(f"⚠️  W&B initialization failed: {e}")
            print("Continuing without W&B logging...")
    elif args.model == 'ss4rec':
        print("📊 W&B will be handled by SS4Rec training script")
    
    # Train based on model type
    try:
        if args.model == 'ncf':
            print("🚀 Starting Neural CF training...")
            trainer = NCFTrainer(config_path)
            trainer.prepare_data()
            trainer.create_model()
            trainer.setup_training()
            
            # Add W&B logging to trainer if available
            if not args.no_wandb and wandb.run:
                trainer.wandb_run = wandb.run
            
            best_rmse = trainer.train()
            
            print(f"""
🎉 Neural CF Training Complete!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 Best RMSE: {best_rmse:.4f}
📊 Model saved to: {config['paths']['model_dir']}
⏱️  Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
            
            # Log to W&B
            if not args.no_wandb and wandb.run:
                wandb.log({
                    "final/best_val_rmse": best_rmse,
                    "final/model_type": "neural_cf",
                    "final/training_complete": True
                })
                wandb.finish()
        
        elif args.model == 'ss4rec':
            # Use standalone SS4Rec training script
            print("🚀 Starting SS4Rec training...")
            
            # Import subprocess to run the training script
            import subprocess
            
            # Build command for SS4Rec training
            cmd = [
                sys.executable, 'training/train_ss4rec.py',
                '--batch-size', str(config.get('training', {}).get('batch_size', 1024)),
                '--epochs', str(config.get('training', {}).get('num_epochs', 100)),
                '--lr', str(config.get('training', {}).get('learning_rate', 0.001)),
                '--max-seq-len', str(config.get('model', {}).get('max_seq_len', 200)),
                '--output-dir', 'results/ss4rec'
            ]
            
            if args.no_wandb:
                # SS4Rec script doesn't support --no-wandb, skip W&B setup in env
                os.environ['WANDB_MODE'] = 'disabled'
            
            if args.debug:
                cmd.extend(['--debug'])
                print("🔍 Debug mode enabled for SS4Rec training")
            
            print(f"📋 Running: {' '.join(cmd)}")
            
            try:
                # Run SS4Rec training
                result = subprocess.run(cmd, check=True, capture_output=False, text=True)
                
                # Extract RMSE from results file if available
                results_file = Path('results/ss4rec/results.json')
                if results_file.exists():
                    import json
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    best_rmse = results.get('best_val_rmse', 0.0)
                else:
                    best_rmse = 0.0  # Default if no results file
                
                print(f"""
🎉 SS4Rec Training Complete!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 Best RMSE: {best_rmse:.4f}
📊 Model saved to: results/ss4rec/
⏱️  Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🔬 SOTA Performance: {'✅' if best_rmse < 0.70 else '❌'} Target < 0.70
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
                
                # Log to W&B if enabled
                if not args.no_wandb and wandb.run:
                    wandb.log({
                        "final/best_val_rmse": best_rmse,
                        "final/model_type": "ss4rec",
                        "final/training_complete": True,
                        "final/sota_target_achieved": best_rmse < 0.70
                    })
                    wandb.finish()
                    
            except subprocess.CalledProcessError as e:
                print(f"❌ SS4Rec training failed: {e}")
                sys.exit(1)
            except FileNotFoundError:
                print("❌ SS4Rec training script not found!")
                print("training/train_ss4rec.py is missing.")
                sys.exit(1)
        
        # Success - compatible with auto_train.py Discord integration
        print("\n✅ Training completed successfully!")
        print("🔔 Discord notification should be sent automatically")
        
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
        if not args.no_wandb and wandb.run:
            wandb.finish()
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        if not args.no_wandb and wandb.run:
            wandb.log({"error": str(e)})
            wandb.finish()
        raise  # Re-raise for auto_train.py to catch


if __name__ == "__main__":
    main()