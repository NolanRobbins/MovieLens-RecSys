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
from src.data.etl_pipeline import load_processed_data
from training.utils.data_loaders import MovieLensDataset, create_data_loaders
from training.utils.metrics import compute_metrics, log_metrics


class NCFTrainer:
    """Neural CF trainer optimized for A6000 GPU"""
    
    def __init__(self, config_path: str):
        """Initialize trainer with configuration"""
        self.config = self._load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_logging()
        
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
        """Load and prepare MovieLens data"""
        self.logger.info("Loading MovieLens dataset...")
        
        # Load processed data
        data_dir = Path(self.config['paths']['data_dir'])
        train_df = pd.read_csv(data_dir / 'train_data.csv')
        val_df = pd.read_csv(data_dir / 'val_data.csv')
        test_df = pd.read_csv(data_dir / 'test_data.csv')
        
        # Get unique counts
        self.n_users = max(train_df['user_id'].max(), val_df['user_id'].max(), test_df['user_id'].max()) + 1
        self.n_items = max(train_df['movie_id'].max(), val_df['movie_id'].max(), test_df['movie_id'].max()) + 1
        
        self.logger.info(f"Dataset: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
        self.logger.info(f"Users: {self.n_users}, Items: {self.n_items}")
        
        # Create data loaders
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(
            train_df, val_df, test_df,
            batch_size=self.config['training']['batch_size'],
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
        
        self.logger.info(f"Model created with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def setup_training(self):
        """Setup optimizer, scheduler, and loss function"""
        # Optimizer
        optimizer_config = self.config['optimizer']
        if optimizer_config['type'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=optimizer_config['weight_decay']
            )
        
        # Scheduler
        scheduler_config = self.config['scheduler']
        if scheduler_config['type'] == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_config['factor'],
                patience=scheduler_config['patience'],
                min_lr=scheduler_config['min_lr']
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
        
        # Create model directory
        model_dir = Path(self.config['paths']['model_dir'])
        model_dir.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(self.config['training']['epochs']):
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
            self.logger.info(f"Epoch {epoch+1}/{self.config['training']['epochs']}")
            self.logger.info(f"Train Loss: {train_loss:.4f}")
            self.logger.info(f"Val Loss: {val_metrics['val_loss']:.4f}, Val RMSE: {val_metrics['rmse']:.4f}")
            self.logger.info(f"Epoch Time: {epoch_time:.2f}s")
            
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
            if patience_counter >= self.config['training']['early_stopping']['patience']:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # Checkpoint saving
            if (epoch + 1) % self.config['logging']['checkpoint_freq'] == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_rmse': val_metrics['rmse'],
                    'config': self.config
                }, model_dir / f'checkpoint_epoch_{epoch+1}.pt')
        
        self.logger.info(f"Training completed. Best validation RMSE: {best_val_rmse:.4f}")
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