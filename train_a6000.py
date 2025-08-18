#!/usr/bin/env python3
"""
A6000-Optimized Training Script for MovieLens Recommendation System
Consolidated from multiple training scripts with A6000 GPU optimizations
"""

import os
import sys
import json
import time
import pickle
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd

# Optional imports with graceful fallback
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️ W&B not available - training will continue without logging")

# Add project root to path
sys.path.append(str(Path(__file__).parent))

warnings.filterwarnings('ignore')

class A6000OptimizedVAE(nn.Module):
    """
    Hybrid VAE optimized for A6000 GPU training
    Consolidates best practices from enhanced_training_v2.py and ranking_optimized_training.py
    """
    
    def __init__(self, n_users: int, n_movies: int, config: Dict[str, Any]):
        super().__init__()
        
        self.n_users = n_users
        self.n_movies = n_movies
        self.config = config
        
        n_factors = config.get('n_factors', 200)
        hidden_dims = config.get('hidden_dims', [1024, 512, 256, 128])
        latent_dim = config.get('latent_dim', 128)
        dropout_rate = config.get('dropout_rate', 0.4)
        use_batch_norm = config.get('use_batch_norm', True)
        use_layer_norm = config.get('use_layer_norm', True)
        
        # Embeddings with proper initialization
        self.user_embedding = nn.Embedding(n_users, n_factors)
        self.movie_embedding = nn.Embedding(n_movies, n_factors)
        
        # Xavier initialization for better convergence
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.movie_embedding.weight)
        
        # Encoder architecture
        encoder_layers = []
        input_dim = n_factors * 2
        
        for i, hidden_dim in enumerate(hidden_dims):
            encoder_layers.append(nn.Linear(input_dim, hidden_dim))
            
            if use_batch_norm:
                encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            if use_layer_norm:
                encoder_layers.append(nn.LayerNorm(hidden_dim))
            
            encoder_layers.append(nn.GELU())
            encoder_layers.append(nn.Dropout(dropout_rate))
            
            input_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # VAE latent layers
        self.mu_layer = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar_layer = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder architecture
        decoder_layers = []
        input_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
                nn.GELU(),
                nn.Dropout(dropout_rate * 0.5)
            ])
            input_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(input_dim, 1))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Rating scaling parameters
        self.rating_scale = nn.Parameter(torch.tensor([4.5]), requires_grad=False)
        self.rating_offset = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
    
    def encode(self, user_ids: torch.Tensor, movie_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode user-movie pairs to latent space"""
        user_emb = self.user_embedding(user_ids)
        movie_emb = self.movie_embedding(movie_ids)
        
        combined = torch.cat([user_emb, movie_emb], dim=1)
        encoded = self.encoder(combined)
        
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
        scaled = torch.sigmoid(decoded) * self.rating_scale + self.rating_offset
        return scaled
    
    def forward(self, user_ids: torch.Tensor, movie_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass"""
        mu, logvar = self.encode(user_ids, movie_ids)
        z = self.reparameterize(mu, logvar)
        predictions = self.decode(z)
        return predictions, mu, logvar

class A6000Trainer:
    """
    A6000-optimized trainer with consolidated best practices
    """
    
    def __init__(self, config: Dict[str, Any], experiment_id: Optional[str] = None):
        self.config = config
        self.experiment_id = experiment_id or f"a6000_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # GPU setup optimized for A6000
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            # A6000 optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print(f"🚀 GPU: {torch.cuda.get_device_name()} with {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.use_wandb = WANDB_AVAILABLE and config.get('use_wandb', True)
        
        print(f"🎯 Experiment ID: {self.experiment_id}")
    
    def setup_wandb(self):
        """Setup Weights & Biases tracking"""
        if not self.use_wandb:
            return
        
        try:
            wandb.init(
                project="movielens-a6000-training",
                name=f"a6000_{self.experiment_id}",
                id=self.experiment_id,
                config=self.config,
                tags=["a6000", "consolidated", "optimized"]
            )
            print("✅ W&B initialized")
        except Exception as e:
            print(f"⚠️ W&B initialization failed: {e}")
            self.use_wandb = False
    
    def load_data(self) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
        """Load and prepare data with A6000 optimizations"""
        print("📊 Loading data...")
        
        data_dir = Path("data/processed")
        train_data = pd.read_csv(data_dir / "train_data.csv")
        val_data = pd.read_csv(data_dir / "val_data.csv")
        
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
        
        # A6000-optimized data loaders
        batch_size = self.config.get('batch_size', 2048)
        num_workers = min(8, os.cpu_count() or 1)  # A6000 can handle more workers
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True
        )
        
        data_info = {
            'n_users': len(mappings['user_to_index']),
            'n_movies': len(mappings['movie_to_index']),
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'mappings': mappings
        }
        
        print(f"✅ Data loaded: {data_info['train_samples']:,} train, {data_info['val_samples']:,} val samples")
        print(f"📦 Batch size: {batch_size}, Workers: {num_workers}")
        
        return train_loader, val_loader, data_info
    
    def create_model(self, n_users: int, n_movies: int) -> A6000OptimizedVAE:
        """Create A6000-optimized model"""
        model = A6000OptimizedVAE(n_users, n_movies, self.config).to(self.device)
        
        # Enable mixed precision for A6000
        if self.config.get('use_amp', True):
            print("⚡ Mixed precision enabled")
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"🧠 Model: {total_params:,} parameters ({trainable_params:,} trainable)")
        
        return model
    
    def setup_training_components(self, model: A6000OptimizedVAE):
        """Setup optimizer and scheduler"""
        
        # AdamW optimizer optimized for A6000
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.get('lr', 0.0003),
            weight_decay=self.config.get('weight_decay', 0.0001),
            betas=(0.9, 0.999),
            eps=1e-8,
            amsgrad=False  # Faster on A6000
        )
        
        # Cosine annealing scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.get('n_epochs', 100),
            eta_min=1e-6
        )
        
        print("✅ Training components setup")
    
    def enhanced_vae_loss(self, predictions: torch.Tensor, targets: torch.Tensor, 
                         mu: torch.Tensor, logvar: torch.Tensor, epoch: int) -> Dict[str, torch.Tensor]:
        """Enhanced VAE loss with adaptive KL weighting"""
        
        # Reconstruction loss
        recon_loss = F.mse_loss(predictions.squeeze(), targets)
        
        # KL divergence with free bits
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        free_bits = self.config.get('free_bits', 2.0)
        kl_loss = torch.mean(torch.clamp(kl_div, min=free_bits))
        
        # Adaptive KL weight
        kl_weight = self.config.get('kl_weight', 0.01)
        if self.config.get('adaptive_kl', True):
            # Gradually increase KL weight
            warmup_epochs = self.config.get('kl_warmup_epochs', 20)
            if epoch < warmup_epochs:
                kl_weight *= (epoch + 1) / warmup_epochs
        
        total_loss = recon_loss + kl_weight * kl_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'kl_weight': torch.tensor(kl_weight)
        }
    
    def train_epoch(self, model: A6000OptimizedVAE, train_loader: DataLoader, 
                   epoch: int, scaler: Optional[torch.cuda.amp.GradScaler]) -> Dict[str, float]:
        """Train one epoch with A6000 optimizations"""
        model.train()
        epoch_metrics = {
            'train_loss': 0.0,
            'train_recon_loss': 0.0,
            'train_kl_loss': 0.0,
            'train_rmse': 0.0
        }
        
        num_batches = len(train_loader)
        use_amp = scaler is not None
        
        for batch_idx, (users, movies, ratings) in enumerate(train_loader):
            users, movies, ratings = users.to(self.device, non_blocking=True), \
                                   movies.to(self.device, non_blocking=True), \
                                   ratings.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)  # A6000 optimization
            
            # Forward pass with mixed precision
            if use_amp:
                with torch.cuda.amp.autocast():
                    predictions, mu, logvar = model(users, movies)
                    loss_dict = self.enhanced_vae_loss(predictions, ratings, mu, logvar, epoch)
                
                scaler.scale(loss_dict['total_loss']).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                predictions, mu, logvar = model(users, movies)
                loss_dict = self.enhanced_vae_loss(predictions, ratings, mu, logvar, epoch)
                
                loss_dict['total_loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            # Update metrics
            epoch_metrics['train_loss'] += loss_dict['total_loss'].item()
            epoch_metrics['train_recon_loss'] += loss_dict['recon_loss'].item()
            epoch_metrics['train_kl_loss'] += loss_dict['kl_loss'].item()
            
            with torch.no_grad():
                rmse = torch.sqrt(F.mse_loss(predictions.squeeze(), ratings))
                epoch_metrics['train_rmse'] += rmse.item()
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    def validate_epoch(self, model: A6000OptimizedVAE, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
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
                users, movies, ratings = users.to(self.device, non_blocking=True), \
                                       movies.to(self.device, non_blocking=True), \
                                       ratings.to(self.device, non_blocking=True)
                
                predictions, mu, logvar = model(users, movies)
                loss_dict = self.enhanced_vae_loss(predictions, ratings, mu, logvar, epoch)
                
                epoch_metrics['val_loss'] += loss_dict['total_loss'].item()
                epoch_metrics['val_recon_loss'] += loss_dict['recon_loss'].item()
                epoch_metrics['val_kl_loss'] += loss_dict['kl_loss'].item()
                
                rmse = torch.sqrt(F.mse_loss(predictions.squeeze(), ratings))
                mae = F.l1_loss(predictions.squeeze(), ratings)
                
                epoch_metrics['val_rmse'] += rmse.item()
                epoch_metrics['val_mae'] += mae.item()
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    def train(self) -> float:
        """Main training loop optimized for A6000"""
        print(f"🚀 Starting A6000-optimized training: {self.experiment_id}")
        
        self.setup_wandb()
        
        # Load data
        train_loader, val_loader, data_info = self.load_data()
        
        # Create model
        self.model = self.create_model(data_info['n_users'], data_info['n_movies'])
        
        # Setup training components
        self.setup_training_components(self.model)
        
        # Mixed precision scaler for A6000
        scaler = torch.cuda.amp.GradScaler() if self.config.get('use_amp', True) else None
        
        # Training state
        best_val_rmse = float('inf')
        best_model_state = None
        patience_counter = 0
        start_time = time.time()
        
        n_epochs = self.config.get('n_epochs', 100)
        patience = self.config.get('patience', 15)
        target_rmse = self.config.get('target_rmse', 0.55)
        
        print(f"🎯 Target RMSE: ≤ {target_rmse}")
        print(f"📊 Training for up to {n_epochs} epochs...")
        print("=" * 80)
        
        # Training loop
        for epoch in range(n_epochs):
            epoch_start = time.time()
            
            # Update learning rate
            self.scheduler.step()
            
            # Train and validate
            train_metrics = self.train_epoch(self.model, train_loader, epoch, scaler)
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
            print(f"Epoch {epoch+1:3d}/{n_epochs} | "
                  f"Train Loss: {train_metrics['train_loss']:.4f} | "
                  f"Val RMSE: {val_metrics['val_rmse']:.4f} | "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.6f} | "
                  f"Time: {epoch_time:.1f}s {status_icon}")
            
            # Log to W&B
            if self.use_wandb:
                log_dict = {
                    'epoch': epoch + 1,
                    'lr': self.optimizer.param_groups[0]['lr'],
                    'epoch_time': epoch_time,
                    'best_val_rmse': best_val_rmse,
                    **train_metrics,
                    **val_metrics
                }
                wandb.log(log_dict)
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\n⏹️ Early stopping triggered after {epoch + 1} epochs")
                break
            
            # Target achieved
            if best_val_rmse <= target_rmse:
                print(f"\n🎉 TARGET ACHIEVED! RMSE ≤ {target_rmse}")
                break
        
        # Load best model and save
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        total_time = time.time() - start_time
        model_path = self._save_model(best_val_rmse, total_time, epoch + 1)
        
        print(f"\n🎉 Training completed!")
        print(f"   Best RMSE: {best_val_rmse:.4f}")
        print(f"   Total time: {total_time/3600:.1f} hours")
        print(f"   Target achieved: {'✅' if best_val_rmse <= target_rmse else '❌'}")
        print(f"   Model saved: {model_path}")
        
        if self.use_wandb:
            wandb.finish()
        
        return best_val_rmse
    
    def _save_model(self, best_rmse: float, total_time: float, epochs_trained: int) -> str:
        """Save trained model"""
        models_dir = Path("models/a6000_trained")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        model_filename = f"{self.experiment_id}_rmse_{best_rmse:.4f}.pt"
        model_path = models_dir / model_filename
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'experiment_id': self.experiment_id,
            'config': self.config,
            'training_results': {
                'best_rmse': best_rmse,
                'epochs_trained': epochs_trained,
                'training_time_hours': total_time / 3600,
                'target_achieved': best_rmse <= self.config.get('target_rmse', 0.55)
            },
            'model_info': {
                'model_type': 'A6000OptimizedVAE',
                'n_parameters': sum(p.numel() for p in self.model.parameters()),
                'device_trained': str(self.device),
                'gpu_name': torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'
            }
        }
        
        torch.save(save_dict, model_path)
        print(f"💾 Model saved: {model_path}")
        
        return str(model_path)

def get_a6000_optimized_config() -> Dict[str, Any]:
    """Get A6000-optimized configuration"""
    return {
        # Model architecture
        'n_factors': 200,
        'hidden_dims': [1024, 512, 256, 128],
        'latent_dim': 128,
        'dropout_rate': 0.4,
        'use_batch_norm': True,
        'use_layer_norm': True,
        
        # Training parameters
        'batch_size': 2048,  # Optimized for A6000 48GB
        'n_epochs': 100,
        'lr': 0.0003,
        'weight_decay': 0.0001,
        'target_rmse': 0.55,
        
        # VAE parameters
        'kl_weight': 0.008,  # Proven stable value
        'free_bits': 2.0,
        'adaptive_kl': True,
        'kl_warmup_epochs': 20,
        
        # A6000 optimizations
        'use_amp': True,  # Mixed precision
        'use_wandb': True,
        
        # Early stopping
        'patience': 15
    }

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='A6000-Optimized MovieLens Training')
    parser.add_argument('--config', type=str, help='Config file path')
    parser.add_argument('--experiment-id', type=str, help='Experiment identifier')
    parser.add_argument('--target-rmse', type=float, default=0.55, help='Target RMSE')
    parser.add_argument('--no-wandb', action='store_true', help='Disable W&B logging')
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config = json.load(f)
    else:
        config = get_a6000_optimized_config()
    
    # Apply command line overrides
    if args.target_rmse:
        config['target_rmse'] = args.target_rmse
    if args.no_wandb:
        config['use_wandb'] = False
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.epochs:
        config['n_epochs'] = args.epochs
    
    # Create trainer and start training
    trainer = A6000Trainer(config, args.experiment_id)
    final_rmse = trainer.train()
    
    print(f"\n🏁 Final RMSE: {final_rmse:.4f}")
    print(f"🎯 Target: ≤ {config['target_rmse']} {'✅' if final_rmse <= config['target_rmse'] else '❌'}")

if __name__ == "__main__":
    main()