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

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

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
            mode: Dataset mode ('train', 'val', 'test')
        """
        self.data = data.copy()
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.timestamp_col = timestamp_col
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
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
            
            # Create training sequences with sliding window
            for i in range(self.min_seq_len, len(items) + 1):
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
        
        # Fix: Pad timestamps with the last timestamp instead of zeros to maintain temporal order
        if seq_len < self.max_seq_len and seq_len > 0:
            last_timestamp = seq_data['timestamp_sequence'][-1]
            timestamp_seq[seq_len:] = last_timestamp
        
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
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for SS4Rec training"""
    
    train_dataset = MovieLensSequentialDataset(
        train_data, max_seq_len=max_seq_len, mode='train'
    )
    val_dataset = MovieLensSequentialDataset(
        val_data, max_seq_len=max_seq_len, mode='val'
    )
    test_dataset = MovieLensSequentialDataset(
        test_data, max_seq_len=max_seq_len, mode='test'
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
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability (more aggressive)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        # Update parameters
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
    
    args = parser.parse_args()
    
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
    output_dir = Path(args.output_dir)
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
            max_seq_len=args.max_seq_len
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
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(f"Trainable parameters: {trainable_params:,}")
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training loop
        best_val_rmse = float('inf')
        best_epoch = 0
        patience_counter = 0
        
        logging.info("Starting training...")
        
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            logging.info(f"Starting Epoch {epoch}/{args.epochs} - SS4Rec Training")
            
            # Log epoch start with consistent step counter
            epoch_start_step = (epoch - 1) * len(train_loader) + 1
            wandb.log({
                'epoch_start': epoch,
                'total_epochs': args.epochs
            }, step=epoch_start_step)
            
            # Train
            train_loss = train_epoch(
                model, train_loader, optimizer, criterion, device, epoch
            )
            
            # Validate
            val_loss, val_rmse, val_mae = evaluate_model(
                model, val_loader, criterion, device
            )
            
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
                best_epoch = epoch
                patience_counter = 0
                
                # Save model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_rmse': val_rmse,
                    'config': config
                }, output_dir / 'best_model.pth')
                
                logging.info(f"New best model saved! Val RMSE: {val_rmse:.6f}")
                
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= args.early_stopping:
                logging.info(f"Early stopping triggered after {patience_counter} epochs without improvement")
                break
        
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
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise
    
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()