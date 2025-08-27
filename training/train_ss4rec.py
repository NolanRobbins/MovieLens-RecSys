#!/usr/bin/env python3
"""
SS4Rec Training Script for MovieLens Dataset
Based on paper: "SS4Rec: Continuous-Time Sequential Recommendation with State Space Models"

Implements proper sequential recommendation methodology adapted for rating prediction.
"""

import os
import sys
import logging
import argparse
import time
import json
import pickle
import subprocess
import requests
import signal
import atexit
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import wandb
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Global variables for signal handling
training_interrupted = False
current_model = None
current_optimizer = None
current_epoch = 0
current_best_rmse = float('inf')
output_directory = None
discord_webhook = None


def send_discord_notification(message: str, webhook_url: str, color: int = 5763719) -> bool:
    """Send completion notification via Discord webhook"""
    try:
        payload = {
            "embeds": [{
                "title": "🎬 MovieLens RecSys Training Update",
                "description": message,
                "color": color,
                "timestamp": datetime.utcnow().isoformat(),
                "footer": {"text": "RunPod A6000 Auto-Trainer"}
            }]
        }
        response = requests.post(webhook_url, json=payload, timeout=10)
        return response.status_code == 204
    except Exception as e:
        logging.warning(f"Discord notification failed: {e}")
        return False


def handle_training_interruption(signum, frame):
    """Handle training interruption gracefully with Discord notification"""
    global training_interrupted, current_model, current_optimizer, current_epoch
    global current_best_rmse, output_directory, discord_webhook
    
    training_interrupted = True
    signal_name = signal.Signals(signum).name
    
    logging.warning(f"🛑 Training interrupted by signal {signal_name} ({signum})")
    
    # Save current state if possible
    if current_model is not None and output_directory is not None:
        try:
            interrupt_file = output_directory / f'interrupted_epoch_{current_epoch}.pth'
            torch.save({
                'epoch': current_epoch,
                'model_state_dict': current_model.state_dict(),
                'optimizer_state_dict': current_optimizer.state_dict() if current_optimizer else None,
                'best_rmse': current_best_rmse,
                'interrupted_by': signal_name,
                'interrupted_at': datetime.utcnow().isoformat()
            }, interrupt_file)
            logging.info(f"💾 Interrupted state saved to: {interrupt_file}")
            
            # Send Discord notification about interruption
            if discord_webhook:
                duration_mins = time.time() - getattr(handle_training_interruption, 'start_time', time.time())
                duration_mins = duration_mins / 60
                
                interrupt_message = f"""
🛑 **Training Interrupted - SS4Rec**

**Status**: Stopped by signal {signal_name}
**Epoch**: {current_epoch} (in progress)
**Best RMSE**: {current_best_rmse:.6f}
**Duration**: {duration_mins:.1f} minutes
**Saved**: {interrupt_file.name}

🔄 **To Resume**: Use `--resume {interrupt_file}`
📱 **RunPod**: Process can be safely restarted
                """.strip()
                
                send_discord_notification(interrupt_message, discord_webhook, color=16776960)  # Yellow for warning
                
        except Exception as e:
            logging.error(f"❌ Failed to save interrupted state: {e}")
    
    # Exit gracefully
    logging.info("🛑 Training stopped. Use --resume to continue from checkpoint.")
    sys.exit(1)


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""
    signal.signal(signal.SIGINT, handle_training_interruption)   # Ctrl+C
    signal.signal(signal.SIGTERM, handle_training_interruption)  # Kill command
    # Store start time for duration calculation
    handle_training_interruption.start_time = time.time()


def get_system_info() -> str:
    """Get system information"""
    try:
        hostname = subprocess.check_output(['hostname'], text=True).strip()
        return f"Instance: {hostname} | GPU: NVIDIA A6000"
    except:
        return "Instance: unknown | GPU: NVIDIA A6000"

from models.sota_2025.ss4rec import SS4Rec, create_ss4rec_model
from src.data.feature_pipeline import EnhancedFeaturePipeline as FeaturePipeline
from src.evaluation.evaluate_model import ModelEvaluator


class MovieLensSequentialDataset(Dataset):
    """
    Sequential dataset for SS4Rec training
    
    Creates user sequences with timestamps for temporal modeling
    """
    
    def __init__(self, 
                 data: pd.DataFrame,
                 user_col: str = 'user_idx',
                 item_col: str = 'movie_idx', 
                 rating_col: str = 'rating',
                 timestamp_col: str = 'timestamp',
                 max_seq_len: int = 200,
                 min_seq_len: int = 5,
                 max_sequences_per_user: int = 10,
                 mode: str = 'train'):
        """
        Initialize sequential dataset
        
        Args:
            data: MovieLens data with user interactions
            user_col: User ID column name
            item_col: Item ID column name  
            rating_col: Rating column name
            timestamp_col: Timestamp column name
            max_seq_len: Maximum sequence length
            min_seq_len: Minimum sequence length
            max_sequences_per_user: Maximum sequences per user (prevents dataset explosion)
            mode: Dataset mode ('train', 'val', 'test')
        """
        self.data = data.copy()
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.timestamp_col = timestamp_col
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.max_sequences_per_user = max_sequences_per_user
        self.mode = mode
        
        # Sort by user and timestamp
        self.data = self.data.sort_values([user_col, timestamp_col])
        
        # Create user sequences
        self.sequences = self._create_sequences()
        
        logging.info(f"Created {len(self.sequences)} sequences in {mode} mode")
    
    def _create_sequences(self) -> List[Dict[str, Any]]:
        """Create user sequences for training"""
        sequences = []
        
        for user_id, group in self.data.groupby(self.user_col):
            user_data = group.sort_values(self.timestamp_col)
            
            if len(user_data) < self.min_seq_len:
                continue
            
            # Extract sequences
            items = user_data[self.item_col].values
            ratings = user_data[self.rating_col].values
            timestamps = user_data[self.timestamp_col].values
            
            # CRITICAL FIX: Limit sequences per user to prevent dataset explosion
            # Create training sequences with controlled sampling
            total_possible_sequences = len(items) - self.min_seq_len + 1
            
            if total_possible_sequences <= self.max_sequences_per_user:
                # Use all possible sequences if under limit
                sequence_indices = list(range(self.min_seq_len, len(items) + 1))
            else:
                # Sample evenly distributed sequences across user's history
                step = total_possible_sequences // self.max_sequences_per_user
                sequence_indices = []
                for seq_idx in range(self.max_sequences_per_user):
                    idx = self.min_seq_len + (seq_idx * step)
                    sequence_indices.append(min(idx, len(items)))
                
                # Always include the final sequence (most recent)
                if len(items) not in sequence_indices:
                    sequence_indices[-1] = len(items)
            
            for i in sequence_indices:
                if i > self.max_seq_len:
                    # Use last max_seq_len items
                    start_idx = i - self.max_seq_len
                    seq_items = items[start_idx:i-1]
                    seq_timestamps = timestamps[start_idx:i-1]
                    target_item = items[i-1]
                    target_rating = ratings[i-1]
                    target_timestamp = timestamps[i-1]
                else:
                    # Use all available items
                    seq_items = items[:i-1]
                    seq_timestamps = timestamps[:i-1]
                    target_item = items[i-1]
                    target_rating = ratings[i-1]
                    target_timestamp = timestamps[i-1]
                
                if len(seq_items) >= 1:  # Need at least 1 item in sequence
                    sequences.append({
                        'user_id': user_id,
                        'item_sequence': seq_items,
                        'timestamp_sequence': seq_timestamps,
                        'target_item': target_item,
                        'target_rating': target_rating,
                        'target_timestamp': target_timestamp
                    })
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq_data = self.sequences[idx]
        
        # Pad sequences to max_seq_len
        item_seq = np.zeros(self.max_seq_len, dtype=np.int64)
        timestamp_seq = np.zeros(self.max_seq_len, dtype=np.float32)
        
        seq_len = len(seq_data['item_sequence'])
        item_seq[:seq_len] = seq_data['item_sequence']
        timestamp_seq[:seq_len] = seq_data['timestamp_sequence']
        
        # CRITICAL FIX: Use zero padding for SSM stability - State Space Models expect proper padding
        # Using last_timestamp breaks mathematical assumptions and causes gradient instability
        if seq_len < self.max_seq_len and seq_len > 0:
            timestamp_seq[seq_len:] = 0.0  # Zero padding for SSM stability
        
        return {
            'user_id': torch.tensor(seq_data['user_id'], dtype=torch.long),
            'item_sequence': torch.tensor(item_seq, dtype=torch.long),
            'timestamp_sequence': torch.tensor(timestamp_seq, dtype=torch.float32),
            'target_item': torch.tensor(seq_data['target_item'], dtype=torch.long),
            'target_rating': torch.tensor(seq_data['target_rating'], dtype=torch.float32),
            'seq_len': torch.tensor(seq_len, dtype=torch.long)
        }


def create_data_loaders(train_data: pd.DataFrame,
                       val_data: pd.DataFrame,
                       test_data: pd.DataFrame,
                       batch_size: int = 1024,
                       max_seq_len: int = 200,
                       max_sequences_per_user: int = 10,
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for SS4Rec training"""
    
    train_dataset = MovieLensSequentialDataset(
        train_data, max_seq_len=max_seq_len, max_sequences_per_user=max_sequences_per_user, mode='train'
    )
    val_dataset = MovieLensSequentialDataset(
        val_data, max_seq_len=max_seq_len, max_sequences_per_user=max_sequences_per_user, mode='val'
    )
    test_dataset = MovieLensSequentialDataset(
        test_data, max_seq_len=max_seq_len, max_sequences_per_user=max_sequences_per_user, mode='test'
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def train_epoch(model: SS4Rec,
               loader: DataLoader,
               optimizer: optim.Optimizer,
               criterion: nn.Module,
               device: torch.device,
               epoch: int,
               scaler = None,
               log_interval: int = 100) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(loader):
        # Move to device
        user_ids = batch['user_id'].to(device)
        item_seq = batch['item_sequence'].to(device)
        timestamp_seq = batch['timestamp_sequence'].to(device)
        target_items = batch['target_item'].to(device)
        target_ratings = batch['target_rating'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            # Forward pass
            predictions = model(
                users=user_ids,
                item_seq=item_seq,
                target_items=target_items,
                timestamps=timestamp_seq
            )
            
            # Compute loss
            loss = criterion(predictions, target_ratings)
        
        # Debug nan loss
        if torch.isnan(loss):
            print(f"NaN loss detected!")
            print(f"Predictions stats: min={predictions.min()}, max={predictions.max()}, nan_count={torch.isnan(predictions).sum()}")
            print(f"Target ratings stats: min={target_ratings.min()}, max={target_ratings.max()}")
            print(f"Model has nan params: {any(torch.isnan(p).any() for p in model.parameters())}")
            raise ValueError("NaN loss - stopping training")
        
        # Backward pass with mixed precision
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # Unscale for gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % log_interval == 0:
            current_loss = loss.item()
            progress_pct = (batch_idx / len(loader)) * 100
            logging.info(f'Epoch {epoch}, Batch {batch_idx}/{len(loader)} ({progress_pct:.1f}%), Loss: {current_loss:.6f}')
            
            # Log batch-level progress to W&B with proper step management
            global_step = (epoch - 1) * len(loader) + batch_idx + 1  # +1 to avoid step 0
            try:
                wandb.log({
                    'batch_loss': current_loss,
                    'batch_progress_pct': progress_pct,
                    'epoch_num': epoch,
                    'batch_num': batch_idx
                }, step=global_step)
            except Exception as e:
                logging.warning(f"Failed to log to wandb: {e}")
    
    return total_loss / num_batches


def evaluate_model(model: SS4Rec,
                  loader: DataLoader,
                  criterion: nn.Module,
                  device: torch.device) -> Tuple[float, float, float]:
    """Evaluate model on validation/test set"""
    model.eval()
    total_loss = 0.0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in loader:
            # Move to device
            user_ids = batch['user_id'].to(device)
            item_seq = batch['item_sequence'].to(device)
            timestamp_seq = batch['timestamp_sequence'].to(device)
            target_items = batch['target_item'].to(device)
            target_ratings = batch['target_rating'].to(device)
            
            # Forward pass
            preds = model(
                users=user_ids,
                item_seq=item_seq,
                target_items=target_items,
                timestamps=timestamp_seq
            )
            
            # Compute loss
            loss = criterion(preds, target_ratings)
            total_loss += loss.item()
            
            # Store predictions and targets
            predictions.extend(preds.cpu().numpy())
            targets.extend(target_ratings.cpu().numpy())
    
    # Calculate metrics
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    avg_loss = total_loss / len(loader)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    
    return avg_loss, rmse, mae


def load_config_with_fallback(config_path: str, args: argparse.Namespace) -> argparse.Namespace:
    """
    Load YAML config and merge with command line arguments
    CRITICAL FIX: Properly map nested YAML structure to flat arguments
    """
    if not os.path.exists(config_path):
        logging.warning(f"Config file {config_path} not found, using command line args only")
        return args
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Map nested config to flat arguments (prioritize command line args)
        training_config = config.get('training', {})
        model_config = config.get('model', {})
        data_config = config.get('data', {})
        wandb_config = config.get('wandb', {})
        
        # Update args only if not explicitly set via command line
        # (argparse defaults vs explicit values can be distinguished by checking against defaults)
        if args.batch_size == 1024:  # default value
            args.batch_size = training_config.get('batch_size', args.batch_size)
        
        if args.epochs == 100:  # default value
            args.epochs = training_config.get('num_epochs', args.epochs)
        
        if args.lr == 0.001:  # default value
            args.lr = training_config.get('learning_rate', args.lr)
            
        if args.max_seq_len == 200:  # default value
            args.max_seq_len = model_config.get('max_seq_len', args.max_seq_len)
            
        if args.early_stopping == 10:  # default value
            args.early_stopping = training_config.get('early_stopping', args.early_stopping)
            
        if args.seed == 42:  # default value
            args.seed = data_config.get('seed', args.seed)
            
        if args.wandb_project == 'movielens-ss4rec':  # default value
            args.wandb_project = wandb_config.get('project', args.wandb_project)
        
        # Add max_sequences_per_user parameter  
        args.max_sequences_per_user = data_config.get('max_sequences_per_user', 10)
        
        # Add mixed precision support
        args.mixed_precision = training_config.get('mixed_precision', False)
        
        # Store full config for model initialization
        args.full_config = config
        
        logging.info(f"✅ Loaded config from {config_path}")
        logging.info(f"📊 Training: {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")
        
    except Exception as e:
        logging.error(f"❌ Failed to load config {config_path}: {e}")
        logging.info("🔄 Continuing with command line arguments only")
    
    return args


def main():
    parser = argparse.ArgumentParser(description='Train SS4Rec model')
    parser.add_argument('--config', type=str, default='configs/ss4rec.yaml',
                       help='Config file path')
    parser.add_argument('--batch-size', type=int, default=1024,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cuda/cpu/auto)')
    parser.add_argument('--output-dir', type=str, default='results/ss4rec',
                       help='Output directory')
    parser.add_argument('--wandb-project', type=str, default='movielens-ss4rec',
                       help='W&B project name')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--max-seq-len', type=int, default=200,
                       help='Maximum sequence length')
    parser.add_argument('--early-stopping', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--debug', action='store_true',
                       help='Enable comprehensive debug logging for NaN detection')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume training from checkpoint path (e.g., results/ss4rec/best_model.pth)')
    parser.add_argument('--save-every', type=int, default=10,
                       help='Save checkpoint every N epochs (0 = only save best)')
    
    args = parser.parse_args()
    
    # CRITICAL FIX: Load and merge config with command line arguments
    args = load_config_with_fallback(args.config, args)
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logging.info(f"Using device: {device}")
    
    # Create output directory
    global output_directory  # Declare global before assignment
    output_dir = Path(args.output_dir)
    output_directory = output_dir  # Set global for signal handler
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = output_dir / f"ss4rec_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_level = logging.DEBUG if args.debug else logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    if args.debug:
        logging.info("🔍 DEBUG MODE ENABLED - Comprehensive logging and NaN detection active")
        logging.info("⚠️  Training performance will be significantly slower")
        # Set environment variable for SS4Rec model to use debug mode
        os.environ['SS4REC_DEBUG_LOGGING'] = '1'
        # Enable PyTorch anomaly detection for debugging
        torch.autograd.set_detect_anomaly(True)
        logging.info("🚨 PyTorch anomaly detection enabled")
    else:
        # Ensure debug mode is disabled
        os.environ['SS4REC_DEBUG_LOGGING'] = '0'
        logging.info("🚀 PRODUCTION MODE - Optimized performance")
    
    # Initialize W&B
    wandb.login()  # Authenticate with wandb
    wandb.init(
        project=args.wandb_project,
        name=f"ss4rec_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config=vars(args),
        tags=['ss4rec', 'sequential', 'movielens']
    )
    
    # Send Discord start notification
    webhook_url = os.environ.get('DISCORD_WEBHOOK_URL', '')
    system_info = get_system_info()  # Get this regardless for potential use later
    
    if webhook_url:
        start_message = f"""
**🚀 Training Started - SS4REC**
{'🔍 **DEBUG MODE ENABLED** - NaN detection active' if args.debug else ''}

📊 **Model:** SS4Rec (SOTA 2025)
⏰ **Started:** {datetime.now().strftime('%H:%M:%S UTC')}
🖥️ **System:** {system_info}
🎯 **Target:** Validation RMSE < 0.70 (SOTA)
📊 **Epochs:** {args.epochs}

⏳ Training in progress... notification will be sent when complete.
        """.strip()
        
        send_discord_notification(start_message, webhook_url, color=3447003)  # Blue for start
        logging.info("📱 Discord start notification sent")
    else:
        logging.info("⚠️ Discord webhook not configured - notifications disabled")
    
    # Setup signal handlers for graceful shutdown
    global discord_webhook
    discord_webhook = webhook_url
    setup_signal_handlers()
    logging.info("🛡️ Signal handlers configured for graceful shutdown")
    
    training_start_time = time.time()
    
    try:
        # Load data
        logging.info("Loading MovieLens data...")
        data_dir = Path('data/processed')
        
        train_data = pd.read_csv(data_dir / 'train_data.csv')
        
        # Check if val_data.csv exists, if not create from train split
        val_data_path = data_dir / 'val_data.csv'
        if val_data_path.exists():
            val_data = pd.read_csv(val_data_path)
        else:
            logging.warning("val_data.csv not found, creating validation split from train data")
            # Take last 20% of train data as validation
            split_idx = int(len(train_data) * 0.8)
            val_data = train_data.iloc[split_idx:].copy().reset_index(drop=True)
            train_data = train_data.iloc[:split_idx].copy().reset_index(drop=True)
            logging.info(f"Created validation split: Train={len(train_data)}, Val={len(val_data)}")
        
        # Test data should not be used during training - create empty dataframe
        test_data = pd.DataFrame(columns=['user_idx', 'movie_idx', 'rating', 'timestamp'])
        
        logging.info(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        # Get data statistics
        # Use ONLY train and val data for model dimensions (test data is unseen!)
        n_users = max(train_data['user_idx'].max(), val_data['user_idx'].max()) + 1
        n_items = max(train_data['movie_idx'].max(), val_data['movie_idx'].max()) + 1
        
        logging.info(f"Users: {n_users}, Items: {n_items}")
        
        # Create data loaders
        logging.info("Creating data loaders...")
        train_loader, val_loader, test_loader = create_data_loaders(
            train_data, val_data, test_data,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            max_sequences_per_user=getattr(args, 'max_sequences_per_user', 10)
        )
        
        # Create model
        logging.info("Creating SS4Rec model...")
        config = {
            'd_model': 64,
            'n_layers': 2,
            'd_state': 16,
            'd_conv': 4,
            'expand': 2,
            'dt_min': 0.001,
            'dt_max': 0.1,
            'dropout': 0.1,
            'max_seq_len': args.max_seq_len,
            'rating_prediction': True
        }
        
        model = create_ss4rec_model(n_users, n_items, config)
        model = model.to(device)
        
        # Set global variables for signal handler
        global current_model, current_optimizer, current_best_rmse
        current_model = model
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(f"Trainable parameters: {trainable_params:,}")
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        current_optimizer = optimizer  # Set global for signal handler
        
        # Mixed precision scaler for A6000 optimization
        scaler = torch.cuda.amp.GradScaler() if getattr(args, 'mixed_precision', False) else None
        if scaler:
            logging.info("🚀 Mixed precision training enabled for A6000 optimization")
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training loop
        best_val_rmse = float('inf')
        current_best_rmse = float('inf')  # Initialize global
        best_epoch = 0
        patience_counter = 0
        start_epoch = 1
        
        # Resume from checkpoint if specified
        if args.resume:
            if os.path.exists(args.resume):
                logging.info(f"🔄 Resuming training from checkpoint: {args.resume}")
                checkpoint = torch.load(args.resume, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_val_rmse = checkpoint.get('val_rmse', float('inf'))
                current_best_rmse = best_val_rmse  # Update global
                best_epoch = checkpoint.get('epoch', 0)
                logging.info(f"✅ Resumed from epoch {checkpoint['epoch']}, best RMSE: {best_val_rmse:.6f}")
            else:
                logging.warning(f"⚠️ Checkpoint file not found: {args.resume}. Starting from scratch.")
        
        logging.info(f"Starting training from epoch {start_epoch}...")
        
        # CRITICAL FIX: Add comprehensive error handling for training loop
        try:
            for epoch in range(start_epoch, args.epochs + 1):
                # Update global epoch for signal handler
                global current_epoch, current_best_rmse
                current_epoch = epoch
                
                epoch_start_time = time.time()
                logging.info(f"Starting Epoch {epoch}/{args.epochs} - SS4Rec Training")
                
                try:
                    # Log epoch start with consistent step counter
                    epoch_start_step = (epoch - 1) * len(train_loader) + 1
                    wandb.log({
                        'epoch_start': epoch,
                        'total_epochs': args.epochs
                    }, step=epoch_start_step)
                    
                    # Train
                    train_loss = train_epoch(
                        model, train_loader, optimizer, criterion, device, epoch, scaler
                    )
                    
                    if torch.isnan(torch.tensor(train_loss)):
                        raise ValueError(f"🚨 NaN detected in training loss at epoch {epoch}")
                    
                    # Validate
                    val_loss, val_rmse, val_mae = evaluate_model(
                        model, val_loader, criterion, device
                    )
                    
                    if torch.isnan(torch.tensor([val_loss, val_rmse, val_mae])).any():
                        raise ValueError(f"🚨 NaN detected in validation metrics at epoch {epoch}")
                    
                    # Update learning rate
                    scheduler.step(val_rmse)
                    
                    # Log metrics
                    epoch_time = time.time() - epoch_start_time
                    current_lr = optimizer.param_groups[0]['lr']
                    
                    logging.info(
                        f"Epoch {epoch}/{args.epochs} - "
                        f"Train Loss: {train_loss:.6f}, "
                        f"Val Loss: {val_loss:.6f}, "
                        f"Val RMSE: {val_rmse:.6f}, "
                        f"Val MAE: {val_mae:.6f}, "
                        f"LR: {current_lr:.8f}, "
                        f"Time: {epoch_time:.1f}s"
                    )
                    
                    # W&B epoch metrics logging - use same step counter as batches
                    epoch_step = epoch * len(train_loader)  # End of epoch step
                    wandb.log({
                        'train_loss': train_loss,
                        'val_loss': val_loss, 
                        'val_rmse': val_rmse,
                        'val_mae': val_mae,
                        'learning_rate': current_lr,
                        'epoch_time': epoch_time,
                        'epoch_progress_pct': epoch / args.epochs * 100
                    }, step=epoch_step)
                    
                    # Save best model
                    if val_rmse < best_val_rmse:
                        best_val_rmse = val_rmse
                        current_best_rmse = val_rmse  # Update global for signal handler
                        best_epoch = epoch
                        patience_counter = 0
                        
                        # Save model
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_rmse': val_rmse,
                            'config': getattr(args, 'full_config', {})
                        }, output_dir / 'best_model.pth')
                        
                        logging.info(f"New best model saved! Val RMSE: {val_rmse:.6f}")
                        
                    else:
                        patience_counter += 1
                    
                    # Save periodic checkpoint if enabled
                    if args.save_every > 0 and epoch % args.save_every == 0:
                        checkpoint_file = output_dir / f'checkpoint_epoch_{epoch}.pth'
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_rmse': val_rmse,
                            'best_val_rmse': best_val_rmse,
                            'best_epoch': best_epoch,
                            'config': getattr(args, 'full_config', {})
                        }, checkpoint_file)
                        logging.info(f"📁 Checkpoint saved: {checkpoint_file}")
                        
                        # Keep only last 3 checkpoints to save space
                        checkpoint_pattern = output_dir / 'checkpoint_epoch_*.pth'
                        checkpoints = sorted(output_dir.glob('checkpoint_epoch_*.pth'))
                        if len(checkpoints) > 3:
                            for old_checkpoint in checkpoints[:-3]:
                                old_checkpoint.unlink()
                                logging.info(f"🗑️ Removed old checkpoint: {old_checkpoint}")
                        
                    # Early stopping
                    if patience_counter >= args.early_stopping:
                        logging.info(f"Early stopping triggered after {patience_counter} epochs without improvement")
                        break
                        
                except Exception as epoch_error:
                    logging.error(f"❌ Error in epoch {epoch}: {epoch_error}")
                    logging.error(f"📊 Last train_loss: {train_loss if 'train_loss' in locals() else 'N/A'}")
                    logging.error(f"📊 Last val metrics: loss={val_loss if 'val_loss' in locals() else 'N/A'}, rmse={val_rmse if 'val_rmse' in locals() else 'N/A'}")
                    
                    # Save debug information
                    debug_file = output_dir / f"debug_epoch_{epoch}_error.pt"
                    torch.save({
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'epoch': epoch,
                        'error': str(epoch_error),
                        'train_loss': train_loss if 'train_loss' in locals() else None,
                        'val_metrics': {'loss': val_loss, 'rmse': val_rmse, 'mae': val_mae} if 'val_loss' in locals() else None
                    }, debug_file)
                    logging.info(f"💾 Debug information saved to {debug_file}")
                    
                    # Log error to W&B
                    wandb.log({'training_error': str(epoch_error), 'failed_epoch': epoch})
                    
                    # Re-raise to stop training
                    raise epoch_error
                    
        except Exception as training_error:
            logging.error(f"🚨 CRITICAL: Training failed with error: {training_error}")
            logging.error(f"📍 Full stack trace:", exc_info=True)
            
            # Save final debug state
            final_debug_file = output_dir / "final_training_error_debug.pt"
            torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'error': str(training_error),
                'best_val_rmse': best_val_rmse,
                'best_epoch': best_epoch
            }, final_debug_file)
            logging.info(f"💾 Final debug state saved to {final_debug_file}")
            
            # Log critical error to W&B
            wandb.log({'critical_training_error': str(training_error), 'training_failed': True})
            
            # Exit with error code
            raise training_error
        
        # Final evaluation
        logging.info("Loading best model for final evaluation...")
        checkpoint = torch.load(output_dir / 'best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Test evaluation
        test_loss, test_rmse, test_mae = evaluate_model(
            model, test_loader, criterion, device
        )
        
        logging.info(f"Final Test Results:")
        logging.info(f"Test Loss: {test_loss:.6f}")
        logging.info(f"Test RMSE: {test_rmse:.6f}")
        logging.info(f"Test MAE: {test_mae:.6f}")
        logging.info(f"Best Val RMSE: {best_val_rmse:.6f} (Epoch {best_epoch})")
        
        # Log final results  
        wandb.log({
            'test_loss': test_loss,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'best_val_rmse': best_val_rmse,
            'best_epoch': best_epoch
        })
        
        # Save final results
        results = {
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_loss': test_loss,
            'best_val_rmse': best_val_rmse,
            'best_epoch': best_epoch,
            'total_epochs': args.epochs,
            'config': config,
            'model_params': {
                'total': total_params,
                'trainable': trainable_params
            }
        }
        
        with open(output_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logging.info(f"Training completed! Results saved to {output_dir}")
        
        # Send Discord success notification
        if webhook_url:
            duration_hours = (time.time() - training_start_time) / 3600
            success_message = f"""
**🎉 MovieLens RecSys - SS4Rec - Training Complete!**

⏱️ **Duration:** {duration_hours:.1f} hours
🎯 **Best RMSE:** {best_val_rmse:.6f} {'(✅ SOTA achieved!)' if best_val_rmse < 0.70 else '(Good progress)'}
📊 **Test RMSE:** {test_rmse:.6f}
📊 **Test MAE:** {test_mae:.6f}
🖥️ **System:** {system_info}

📊 **Check your W&B dashboard for detailed metrics**
💾 **Results saved to:** `{output_dir}`

✅ **Instance can now be safely terminated.**
            """.strip()
            
            color = 5763719 if best_val_rmse < 0.70 else 16776960  # Green if SOTA, Yellow if good
            send_discord_notification(success_message, webhook_url, color)
            logging.info("📱 Discord success notification sent")
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        
        # Send Discord failure notification
        if webhook_url:
            duration_hours = (time.time() - training_start_time) / 3600
            failure_message = f"""
**❌ MovieLens RecSys - SS4Rec - Training Failed**

⏱️ **Duration:** {duration_hours:.1f} hours
❌ **Error:** {str(e)[:200]}{'...' if len(str(e)) > 200 else ''}
🖥️ **System:** {system_info}

📝 **Check log file for details**
🔧 **May need to restart training**

Common fixes:
• Check GPU memory: `nvidia-smi`
• Verify data files: `ls data/processed/`
• Check config file
            """.strip()
            
            send_discord_notification(failure_message, webhook_url, color=15158332)  # Red for failure
            logging.info("📱 Discord failure notification sent")
        
        raise
    
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()