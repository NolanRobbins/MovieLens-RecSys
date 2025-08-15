#!/usr/bin/env python3
"""
Enhanced Training Script V2 - Targeting RMSE ≤ 0.55
Integrated with model versioning system and advanced regularization techniques
"""

import os
import sys
import json
import time
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import wandb

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from src.versioning.model_version_manager import ModelVersionManager, ExperimentStatus
    from src.models.next_experiment_config import AdvancedTrainingConfig, ImprovedVAELoss, CosineWarmupScheduler, DataAugmentation
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

warnings.filterwarnings('ignore')

class EnhancedHybridVAE(nn.Module):
    """Enhanced Hybrid VAE with advanced regularization features"""
    
    def __init__(self, n_users: int, n_movies: int, n_factors: int = 200,
                 hidden_dims: list = [1024, 512, 256, 128], latent_dim: int = 128,
                 dropout_rate: float = 0.4, use_batch_norm: bool = True, 
                 use_layer_norm: bool = True):
        super().__init__()
        
        self.n_users = n_users
        self.n_movies = n_movies
        self.n_factors = n_factors
        self.latent_dim = latent_dim
        
        # Embeddings with improved initialization
        self.user_embedding = nn.Embedding(n_users, n_factors)
        self.movie_embedding = nn.Embedding(n_movies, n_factors)
        
        # Initialize embeddings properly
        nn.init.normal_(self.user_embedding.weight, 0, 0.1)
        nn.init.normal_(self.movie_embedding.weight, 0, 0.1)
        
        # Enhanced encoder with deeper architecture
        encoder_layers = []
        input_dim = n_factors * 2
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            encoder_layers.append(nn.Linear(input_dim, hidden_dim))
            
            # Normalization layers
            if use_batch_norm:
                encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            if use_layer_norm:
                encoder_layers.append(nn.LayerNorm(hidden_dim))
            
            # Activation and dropout
            encoder_layers.append(nn.GELU())  # GELU performs better than ReLU for transformers
            encoder_layers.append(nn.Dropout(dropout_rate))
            
            input_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # VAE latent layers
        self.mu_layer = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar_layer = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Enhanced decoder
        decoder_layers = []
        input_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
                nn.GELU(),
                nn.Dropout(dropout_rate * 0.5)  # Reduced dropout in decoder
            ])
            input_dim = hidden_dim
        
        # Final prediction layer
        decoder_layers.append(nn.Linear(input_dim, 1))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Prediction scaling (ratings are 0.5-5.0)
        self.rating_scale = nn.Parameter(torch.tensor([4.5]), requires_grad=False)
        self.rating_offset = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
    
    def encode(self, user_ids: torch.Tensor, movie_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode user-movie pairs to latent space"""
        user_emb = self.user_embedding(user_ids)
        movie_emb = self.movie_embedding(movie_ids)
        
        # Concatenate embeddings
        combined = torch.cat([user_emb, movie_emb], dim=1)
        
        # Encode
        encoded = self.encoder(combined)
        
        # Get mu and logvar
        mu = self.mu_layer(encoded)
        logvar = self.logvar_layer(encoded)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to rating prediction"""
        decoded = self.decoder(z)
        
        # Scale to rating range [0.5, 5.0]
        scaled = torch.sigmoid(decoded) * self.rating_scale + self.rating_offset
        
        return scaled
    
    def forward(self, user_ids: torch.Tensor, movie_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass"""
        mu, logvar = self.encode(user_ids, movie_ids)
        z = self.reparameterize(mu, logvar)
        predictions = self.decode(z)
        
        return predictions, mu, logvar

class EnhancedTrainer:
    """Enhanced trainer with versioning integration"""
    
    def __init__(self, experiment_id: str, config: AdvancedTrainingConfig):
        self.experiment_id = experiment_id
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize version manager
        self.version_manager = ModelVersionManager()
        
        # Initialize W&B if configured
        self.use_wandb = True
        
        # Training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        
        print(f"🚀 Enhanced Trainer initialized for experiment: {experiment_id}")
        print(f"📱 Device: {self.device}")
    
    def setup_wandb(self):
        """Setup Weights & Biases tracking"""
        if not self.use_wandb:
            return
        
        try:
            wandb.init(
                project="movielens-hybrid-vae-enhanced",
                name=f"enhanced_v2_{self.experiment_id.split('_')[-2]}_{self.experiment_id.split('_')[-1]}",
                id=self.experiment_id,
                config={
                    'experiment_id': self.experiment_id,
                    'model_config': self.config.model_config,
                    'training_config': self.config.__dict__,
                    'target_rmse': 0.55
                },
                tags=["enhanced_v2", "targeting_0.55", "advanced_regularization"]
            )
            print("✅ W&B initialized")
        except Exception as e:
            print(f"⚠️  W&B initialization failed: {e}")
            self.use_wandb = False
    
    def load_data(self) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
        """Load and prepare data"""
        print("📊 Loading data...")
        
        # Load processed data
        data_dir = Path("data/processed")
        train_data = pd.read_csv(data_dir / "train_data.csv")
        val_data = pd.read_csv(data_dir / "val_data.csv")
        
        # Load mappings
        with open(data_dir / "data_mappings.pkl", 'rb') as f:
            mappings = pickle.load(f)
        
        # Convert to tensors
        train_users = torch.tensor(train_data['user_id'].values, dtype=torch.long)
        train_movies = torch.tensor(train_data['movie_id'].values, dtype=torch.long)
        train_ratings = torch.tensor(train_data['rating'].values, dtype=torch.float32)
        
        val_users = torch.tensor(val_data['user_id'].values, dtype=torch.long)
        val_movies = torch.tensor(val_data['movie_id'].values, dtype=torch.long)
        val_ratings = torch.tensor(val_data['rating'].values, dtype=torch.float32)
        
        # Create datasets
        train_dataset = TensorDataset(train_users, train_movies, train_ratings)
        val_dataset = TensorDataset(val_users, val_movies, val_ratings)
        
        # Enhanced data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size,
            shuffle=True, 
            num_workers=4,
            pin_memory=True if self.device.type == 'cuda' else False,
            drop_last=True  # For batch norm stability
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        data_info = {
            'n_users': len(mappings['user_to_index']),
            'n_movies': len(mappings['movie_to_index']),
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'mappings': mappings
        }
        
        print(f"✅ Data loaded: {data_info['train_samples']:,} train, {data_info['val_samples']:,} val samples")
        
        return train_loader, val_loader, data_info
    
    def create_model(self, n_users: int, n_movies: int) -> EnhancedHybridVAE:
        """Create enhanced model"""
        model = EnhancedHybridVAE(
            n_users=n_users,
            n_movies=n_movies,
            **self.config.model_config
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"🧠 Model created: {total_params:,} parameters ({trainable_params:,} trainable)")
        
        return model
    
    def setup_training_components(self, model: EnhancedHybridVAE):
        """Setup optimizer, scheduler, and loss function"""
        
        # Enhanced optimizer (AdamW with amsgrad)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.base_lr,
            weight_decay=self.config.weight_decay,
            betas=self.config.optimizer_config['betas'],
            eps=self.config.optimizer_config['eps'],
            amsgrad=self.config.optimizer_config['amsgrad']
        )
        
        # Advanced learning rate scheduler
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_epochs=self.config.scheduler_config['warmup_epochs'],
            total_epochs=self.config.n_epochs,
            min_lr_ratio=self.config.scheduler_config['min_lr_ratio'],
            restart_period=self.config.scheduler_config['restart_period']
        )
        
        # Enhanced loss function
        self.loss_fn = ImprovedVAELoss(
            beta_min=self.config.vae_config['beta_min'],
            beta_max=self.config.vae_config['beta_max'],
            free_bits=self.config.vae_config['free_bits'],
            label_smoothing=self.config.regularization['label_smoothing']
        )
        
        print("✅ Training components setup complete")
    
    def train_epoch(self, model: EnhancedHybridVAE, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train one epoch with advanced techniques"""
        model.train()
        epoch_metrics = {
            'train_loss': 0.0,
            'train_recon_loss': 0.0,
            'train_kl_loss': 0.0,
            'train_rmse': 0.0
        }
        
        num_batches = len(train_loader)
        
        for batch_idx, (users, movies, ratings) in enumerate(train_loader):
            users, movies, ratings = users.to(self.device), movies.to(self.device), ratings.to(self.device)
            
            # Data augmentation
            if self.config.regularization['mixup_alpha'] > 0:
                users, movies, ratings = DataAugmentation.mixup_batch(
                    users, movies, ratings, 
                    alpha=self.config.regularization['mixup_alpha']
                )
            
            if self.config.regularization['noise_factor'] > 0:
                ratings = DataAugmentation.add_noise(
                    ratings, 
                    noise_factor=self.config.regularization['noise_factor']
                )
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions, mu, logvar = model(users, movies)
            
            # Calculate loss
            loss_dict = self.loss_fn(
                predictions, ratings, mu, logvar, 
                epoch, self.config.n_epochs, batch_idx
            )
            
            # Backward pass
            loss_dict['total_loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            batch_size = users.size(0)
            epoch_metrics['train_loss'] += loss_dict['total_loss'].item()
            epoch_metrics['train_recon_loss'] += loss_dict['recon_loss'].item()
            epoch_metrics['train_kl_loss'] += loss_dict['kl_loss'].item()
            
            # Calculate RMSE
            with torch.no_grad():
                rmse = torch.sqrt(F.mse_loss(predictions.squeeze(), ratings))
                epoch_metrics['train_rmse'] += rmse.item()
            
            # Log batch metrics to W&B
            if self.use_wandb and batch_idx % 100 == 0:
                wandb.log({
                    'batch/train_loss': loss_dict['total_loss'].item(),
                    'batch/train_recon': loss_dict['recon_loss'].item(),
                    'batch/train_kl': loss_dict['kl_loss'].item(),
                    'batch/beta': loss_dict['beta'].item(),
                    'batch/lr': self.optimizer.param_groups[0]['lr'],
                    'batch/step': epoch * num_batches + batch_idx
                })
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    def validate_epoch(self, model: EnhancedHybridVAE, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Validate one epoch"""
        model.eval()
        epoch_metrics = {
            'val_loss': 0.0,
            'val_recon_loss': 0.0,
            'val_kl_loss': 0.0,
            'val_rmse': 0.0,
            'val_mae': 0.0
        }
        
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for users, movies, ratings in val_loader:
                users, movies, ratings = users.to(self.device), movies.to(self.device), ratings.to(self.device)
                
                # Forward pass
                predictions, mu, logvar = model(users, movies)
                
                # Calculate loss
                loss_dict = self.loss_fn(
                    predictions, ratings, mu, logvar,
                    epoch, self.config.n_epochs
                )
                
                # Update metrics
                epoch_metrics['val_loss'] += loss_dict['total_loss'].item()
                epoch_metrics['val_recon_loss'] += loss_dict['recon_loss'].item()
                epoch_metrics['val_kl_loss'] += loss_dict['kl_loss'].item()
                
                # Calculate RMSE and MAE
                rmse = torch.sqrt(F.mse_loss(predictions.squeeze(), ratings))
                mae = F.l1_loss(predictions.squeeze(), ratings)
                
                epoch_metrics['val_rmse'] += rmse.item()
                epoch_metrics['val_mae'] += mae.item()
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    def train(self):
        """Main training loop with versioning integration"""
        print(f"🚀 Starting enhanced training for experiment: {self.experiment_id}")
        
        # Mark experiment as running
        self.version_manager.start_experiment(self.experiment_id)
        
        # Setup W&B
        self.setup_wandb()
        
        try:
            # Load data
            train_loader, val_loader, data_info = self.load_data()
            
            # Create model
            self.model = self.create_model(data_info['n_users'], data_info['n_movies'])
            
            # Setup training components
            self.setup_training_components(self.model)
            
            # Training state
            best_val_rmse = float('inf')
            best_model_state = None
            patience_counter = 0
            start_time = time.time()
            
            print(f"🎯 Target RMSE: ≤ 0.55")
            print(f"📊 Training for {self.config.n_epochs} epochs...")
            print("=" * 80)
            
            # Training loop
            for epoch in range(self.config.n_epochs):
                epoch_start = time.time()
                
                # Update learning rate
                self.scheduler.step(epoch)
                
                # Train epoch
                train_metrics = self.train_epoch(self.model, train_loader, epoch)
                
                # Validate epoch
                val_metrics = self.validate_epoch(self.model, val_loader, epoch)
                
                epoch_time = time.time() - epoch_start
                
                # Check for improvement
                is_best = val_metrics['val_rmse'] < best_val_rmse
                if is_best:
                    best_val_rmse = val_metrics['val_rmse']
                    best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                    status_icon = "🏆 NEW BEST!"
                else:
                    patience_counter += 1
                    status_icon = ""
                
                # Log progress
                print(f"Epoch {epoch+1:3d}/{self.config.n_epochs} | "
                      f"Train Loss: {train_metrics['train_loss']:.4f} | "
                      f"Val RMSE: {val_metrics['val_rmse']:.4f} | "
                      f"Time: {epoch_time/60:.1f}min {status_icon}")
                
                # Log to W&B
                if self.use_wandb:
                    log_dict = {
                        'epoch': epoch + 1,
                        'lr': self.optimizer.param_groups[0]['lr'],
                        **train_metrics,
                        **val_metrics,
                        'epoch_time_min': epoch_time / 60,
                        'best_val_rmse': best_val_rmse
                    }
                    wandb.log(log_dict)
                
                # Early stopping
                if patience_counter >= self.config.early_stopping['patience']:
                    print(f"\n⏹️ Early stopping triggered after {epoch + 1} epochs")
                    print(f"   Best RMSE: {best_val_rmse:.4f}")
                    break
                
                # Target achieved check
                if best_val_rmse <= 0.55:
                    print(f"\n🎉 TARGET ACHIEVED! RMSE ≤ 0.55")
                    print(f"   Achieved RMSE: {best_val_rmse:.4f}")
                    break
            
            # Load best model
            if best_model_state:
                self.model.load_state_dict(best_model_state)
            
            total_time = time.time() - start_time
            
            # Save model
            model_path = self._save_model(best_val_rmse, total_time, epoch + 1)
            
            # Complete experiment in version manager
            self.version_manager.complete_experiment(
                experiment_id=self.experiment_id,
                model_path=model_path,
                achieved_rmse=best_val_rmse,
                other_metrics={
                    'val_mae': val_metrics['val_mae'],
                    'train_loss': train_metrics['train_loss'],
                    'val_loss': val_metrics['val_loss']
                },
                epochs_trained=epoch + 1,
                training_time_hours=total_time / 3600,
                gpu_type=str(self.device),
                wandb_run_id=wandb.run.id if self.use_wandb else None
            )
            
            # Set as current best if it's the best so far
            if best_val_rmse <= 0.55:
                self.version_manager.set_current_best(self.experiment_id)
            
            print(f"\n🎉 Training completed!")
            print(f"   Best RMSE: {best_val_rmse:.4f}")
            print(f"   Total time: {total_time/3600:.1f} hours")
            print(f"   Target achieved: {'✅' if best_val_rmse <= 0.55 else '❌'}")
            
            if self.use_wandb:
                wandb.finish()
            
            return best_val_rmse
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            self.version_manager.fail_experiment(self.experiment_id, str(e))
            if self.use_wandb:
                wandb.finish()
            raise
    
    def _save_model(self, best_rmse: float, total_time: float, epochs_trained: int) -> str:
        """Save trained model"""
        # Create model save path
        models_dir = Path("models/trained")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        model_filename = f"{self.experiment_id}_rmse_{best_rmse:.4f}.pt"
        model_path = models_dir / model_filename
        
        # Save model with metadata
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'experiment_id': self.experiment_id,
            'config': self.config.__dict__,
            'training_results': {
                'best_rmse': best_rmse,
                'epochs_trained': epochs_trained,
                'training_time_hours': total_time / 3600,
                'target_achieved': best_rmse <= 0.55
            },
            'model_info': {
                'model_type': 'EnhancedHybridVAE',
                'n_parameters': sum(p.numel() for p in self.model.parameters()),
                'device_trained': str(self.device)
            }
        }
        
        torch.save(save_dict, model_path)
        print(f"💾 Model saved: {model_path}")
        
        return str(model_path)

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Enhanced Training V2')
    parser.add_argument('--experiment-id', required=True, 
                       help='Experiment ID from version manager')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable W&B logging')
    
    args = parser.parse_args()
    
    # Create config
    config = AdvancedTrainingConfig()
    
    # Create trainer
    trainer = EnhancedTrainer(args.experiment_id, config)
    if args.no_wandb:
        trainer.use_wandb = False
    
    # Start training
    final_rmse = trainer.train()
    
    print(f"\n🏁 Final RMSE: {final_rmse:.4f}")
    print(f"🎯 Target: ≤ 0.55 {'✅' if final_rmse <= 0.55 else '❌'}")

if __name__ == "__main__":
    # Add required import
    import pickle
    main()