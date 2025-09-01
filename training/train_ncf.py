"""
Neural Collaborative Filtering Training Script
A6000 GPU Optimized for RunPod Training
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import time
from datetime import datetime
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.baseline.neural_cf import NeuralCollaborativeFiltering
# from src.data.etl_pipeline import load_processed_data
from training.utils.data_loaders import MovieLensDataset, create_data_loaders, prepare_training_data
from training.utils.metrics import compute_metrics, log_metrics


class NCFTrainer:
    """Neural CF trainer optimized for A6000 GPU"""
    
    def __init__(self, config_path: str):
        """Initialize trainer with configuration"""
        self.config = self._load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_logging()
        
        # W&B integration
        self.wandb_run = None  # Will be set externally if W&B is enabled
        
        # A6000 optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            if hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision('high')
    
    def _load_config(self, config_path: str) -> dict:
        """Load YAML configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path(self.config['paths']['logs_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['log_level']),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def prepare_data(self):
        """Load and prepare ONLY training and validation data"""
        self.logger.info("Loading training and validation data only...")
        self.logger.info("⚠️  Test data remains UNSEEN until after training completion!")
        
        # Load ONLY train and validation data
        data_dir = Path(self.config['paths']['data_dir'])
        train_df, val_df = prepare_training_data(str(data_dir))
        
        # Get unique counts from ONLY training and validation data
        self.n_users = max(train_df['user_idx'].max(), val_df['user_idx'].max()) + 1
        self.n_items = max(train_df['movie_idx'].max(), val_df['movie_idx'].max()) + 1
        
        self.logger.info(f"Dataset: {len(train_df)} train, {len(val_df)} val")
        self.logger.info(f"Users: {self.n_users}, Items: {self.n_items}")
        
        # Create data loaders for ONLY train and validation
        self.train_loader, self.val_loader, _ = create_data_loaders(
            train_df, val_df, pd.DataFrame(),  # Empty test_df
            batch_size=int(self.config['training']['batch_size']),
            num_workers=4
        )
    
    def create_model(self):
        """Create and initialize NCF model"""
        model_config = self.config['model']
        
        self.model = NeuralCollaborativeFiltering(
            n_users=self.n_users,
            n_items=self.n_items,
            mf_dim=model_config['mf_dim'],
            mlp_dims=model_config['mlp_dims'],
            dropout_rate=model_config['dropout_rate']
        ).to(self.device)
        
        # PyTorch 2.0 optimization
        if self.config['gpu']['compile']:
            self.model = torch.compile(self.model)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Model created with {total_params} parameters")
        
        # Log model architecture to W&B
        if self.wandb_run:
            self.wandb_run.log({
                "model/total_parameters": total_params,
                "model/mf_dim": model_config['mf_dim'],
                "model/mlp_dims": str(model_config['mlp_dims']),
                "model/dropout_rate": model_config['dropout_rate']
            })
    
    def setup_training(self):
        """Setup optimizer, scheduler, and loss function"""
        # Optimizer
        optimizer_config = self.config['optimizer']
        if optimizer_config['type'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=float(self.config['training']['learning_rate']),
                weight_decay=float(optimizer_config['weight_decay'])
            )
        
        # Scheduler
        scheduler_config = self.config['scheduler']
        if scheduler_config['type'] == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=float(scheduler_config['factor']),
                patience=int(scheduler_config['patience']),
                min_lr=float(scheduler_config['min_lr'])
            )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Mixed precision scaler for A6000
        if self.config['gpu']['mixed_precision']:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
    
    def train_epoch(self, epoch: int):
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (users, items, ratings) in enumerate(self.train_loader):
            users, items, ratings = users.to(self.device), items.to(self.device), ratings.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision forward pass
            if self.scaler:
                with torch.cuda.amp.autocast():
                    predictions = self.model(users, items)
                    loss = self.criterion(predictions, ratings.float())
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(users, items)
                loss = self.criterion(predictions, ratings.float())
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 100 == 0:
                self.logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        return total_loss / num_batches
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for users, items, ratings in self.val_loader:
                users, items, ratings = users.to(self.device), items.to(self.device), ratings.to(self.device)
                
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        predictions = self.model(users, items)
                        loss = self.criterion(predictions, ratings.float())
                else:
                    predictions = self.model(users, items)
                    loss = self.criterion(predictions, ratings.float())
                
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(ratings.cpu().numpy())
        
        # Compute metrics
        metrics = compute_metrics(all_predictions, all_targets)
        metrics['val_loss'] = total_loss / len(self.val_loader)
        
        return metrics
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting Neural CF training...")
        
        best_val_rmse = float('inf')
        patience_counter = 0
        epoch = -1  # Initialize epoch variable
        
        # Create model directory
        model_dir = Path(self.config['paths']['model_dir'])
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Log training start to W&B
        if self.wandb_run:
            self.wandb_run.log({
                "training/started": True,
                "training/total_epochs": int(self.config['training']['epochs']),
                "training/batch_size": int(self.config['training']['batch_size']),
                "training/learning_rate": float(self.config['training']['learning_rate']),
                "training/early_stopping_patience": int(self.config['training']['early_stopping']['patience'])
            })
        
        for epoch in range(int(self.config['training']['epochs'])):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            if hasattr(self, 'scheduler'):
                self.scheduler.step(val_metrics['val_loss'])
            
            epoch_time = time.time() - start_time
            
            # Log metrics
            self.logger.info(f"Epoch {epoch+1}/{int(self.config['training']['epochs'])}")
            self.logger.info(f"Train Loss: {train_loss:.4f}")
            self.logger.info(f"Val Loss: {val_metrics['val_loss']:.4f}, Val RMSE: {val_metrics['rmse']:.4f}")
            self.logger.info(f"Epoch Time: {epoch_time:.2f}s")
            
            # W&B logging
            if self.wandb_run:
                wandb_metrics = {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_metrics['val_loss'],
                    "val_rmse": val_metrics['rmse'],
                    "epoch_time": epoch_time,
                    "learning_rate": self.optimizer.param_groups[0]['lr'],
                    "best_val_rmse": best_val_rmse
                }
                # Add additional metrics if available
                for key, value in val_metrics.items():
                    if key not in wandb_metrics:
                        wandb_metrics[f"val_{key}"] = value
                
                self.wandb_run.log(wandb_metrics)
            
            # Early stopping
            if val_metrics['rmse'] < best_val_rmse:
                best_val_rmse = val_metrics['rmse']
                patience_counter = 0
                
                # Save best model
                if self.config['logging']['save_model']:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_rmse': best_val_rmse,
                        'config': self.config
                    }, model_dir / 'best_model.pt')
                    
                    self.logger.info(f"Saved best model with RMSE: {best_val_rmse:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping check
            if patience_counter >= int(self.config['training']['early_stopping']['patience']):
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # Checkpoint saving
            if (epoch + 1) % int(self.config['logging']['checkpoint_freq']) == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_rmse': val_metrics['rmse'],
                    'config': self.config
                }, model_dir / f'checkpoint_epoch_{epoch+1}.pt')
        
        self.logger.info(f"Training completed. Best validation RMSE: {best_val_rmse:.4f}")
        
        # Log final results to W&B
        if self.wandb_run:
            self.wandb_run.log({
                "final/best_val_rmse": best_val_rmse,
                "final/total_epochs_trained": epoch + 1,
                "final/early_stopped": patience_counter >= int(self.config['training']['early_stopping']['patience']),
                "final/training_completed": True
            })
        
        return best_val_rmse


def main():
    parser = argparse.ArgumentParser(description="Train Neural CF model")
    parser.add_argument('--config', type=str, default='configs/ncf_baseline.yaml',
                        help='Path to configuration file')
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = NCFTrainer(args.config)
    
    # Prepare data and model
    trainer.prepare_data()
    trainer.create_model()
    trainer.setup_training()
    
    # Train model
    best_rmse = trainer.train()
    
    print(f"Training completed. Best validation RMSE: {best_rmse:.4f}")


if __name__ == "__main__":
    main()