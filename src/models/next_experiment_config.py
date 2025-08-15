"""
Next Experiment Configuration - Targeting RMSE ≤ 0.55
Advanced regularization and optimization strategies to address overfitting and improve generalization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Optional
import numpy as np

class AdvancedTrainingConfig:
    """Enhanced training configuration targeting RMSE ≤ 0.55"""
    
    def __init__(self):
        # Core hyperparameters - aggressive anti-overfitting strategy
        self.batch_size = 2048  # Larger batches for better gradient estimates
        self.n_epochs = 200     # More epochs with proper regularization
        self.base_lr = 3e-4     # Lower base learning rate
        self.weight_decay = 1e-4  # Stronger L2 regularization
        
        # Architecture improvements
        self.model_config = {
            'n_factors': 200,           # Increased capacity
            'hidden_dims': [1024, 512, 256, 128],  # Deeper network
            'latent_dim': 128,          # Larger latent space
            'dropout_rate': 0.4,        # Aggressive dropout
            'use_batch_norm': True,     # Batch normalization
            'use_layer_norm': True,     # Layer normalization
        }
        
        # VAE-specific improvements
        self.vae_config = {
            'beta_schedule': 'cosine_annealing',  # Advanced beta scheduling
            'beta_min': 0.001,
            'beta_max': 0.5,
            'kl_weight_epochs': 50,     # Gradual KL weight increase
            'free_bits': 2.0,           # Prevent posterior collapse
        }
        
        # Advanced optimization
        self.optimizer_config = {
            'type': 'adamw',            # AdamW with decoupled weight decay
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'amsgrad': True,            # Improved convergence
        }
        
        # Learning rate scheduling
        self.scheduler_config = {
            'type': 'cosine_with_warmup',
            'warmup_epochs': 10,
            'min_lr_ratio': 0.01,       # Minimum LR as fraction of base LR
            'restart_period': 50,       # Cosine restart every 50 epochs
        }
        
        # Data augmentation and regularization
        self.regularization = {
            'label_smoothing': 0.1,     # Label smoothing for ratings
            'mixup_alpha': 0.2,         # Mixup data augmentation
            'cutmix_alpha': 0.3,        # CutMix for implicit feedback
            'noise_factor': 0.05,       # Input noise injection
        }
        
        # Early stopping with advanced criteria
        self.early_stopping = {
            'patience': 25,
            'min_delta': 1e-4,
            'restore_best_weights': True,
            'monitor': 'val_rmse',
            'mode': 'min'
        }
        
        # Model ensemble
        self.ensemble_config = {
            'n_models': 5,              # Train 5 models for ensemble
            'diversity_weight': 0.1,    # Encourage model diversity
            'ensemble_method': 'weighted_average'
        }

class ImprovedVAELoss(nn.Module):
    """Enhanced VAE loss with advanced regularization techniques"""
    
    def __init__(self, beta_min: float = 0.001, beta_max: float = 0.5, 
                 free_bits: float = 2.0, label_smoothing: float = 0.1):
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.free_bits = free_bits
        self.label_smoothing = label_smoothing
        self.mse_loss = nn.MSELoss()
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                mu: torch.Tensor, logvar: torch.Tensor, epoch: int, 
                total_epochs: int, batch_idx: int = 0) -> Dict[str, torch.Tensor]:
        """
        Advanced VAE loss with:
        - Beta scheduling
        - Free bits regularization
        - Label smoothing
        """
        batch_size = predictions.size(0)
        
        # 1. Reconstruction loss with label smoothing
        if self.label_smoothing > 0:
            # Apply label smoothing to ratings (shift towards mean)
            target_mean = targets.mean()
            smoothed_targets = (1 - self.label_smoothing) * targets + self.label_smoothing * target_mean
            recon_loss = self.mse_loss(predictions.squeeze(), smoothed_targets)
        else:
            recon_loss = self.mse_loss(predictions.squeeze(), targets)
        
        # 2. KL divergence with free bits
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        
        # Apply free bits: only penalize KL above threshold
        kl_loss = torch.maximum(kl_loss, torch.tensor(self.free_bits).to(kl_loss.device))
        kl_loss = kl_loss.mean()
        
        # 3. Beta scheduling (cosine annealing)
        progress = epoch / total_epochs
        beta = self.beta_min + (self.beta_max - self.beta_min) * (1 + np.cos(np.pi * progress)) / 2
        
        # 4. Total loss
        total_loss = recon_loss + beta * kl_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'beta': torch.tensor(beta),
            'effective_kl': beta * kl_loss
        }

class CosineWarmupScheduler:
    """Cosine annealing with warmup and restarts"""
    
    def __init__(self, optimizer: optim.Optimizer, warmup_epochs: int,
                 total_epochs: int, min_lr_ratio: float = 0.01, restart_period: int = 50):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr_ratio = min_lr_ratio
        self.restart_period = restart_period
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
    def step(self, epoch: int):
        """Update learning rates"""
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if epoch < self.warmup_epochs:
                # Warmup phase
                lr = base_lr * (epoch + 1) / self.warmup_epochs
            else:
                # Cosine annealing with restarts
                epoch_in_cycle = (epoch - self.warmup_epochs) % self.restart_period
                progress = epoch_in_cycle / self.restart_period
                lr = (base_lr * self.min_lr_ratio + 
                      (base_lr - base_lr * self.min_lr_ratio) * 
                      (1 + np.cos(np.pi * progress)) / 2)
            
            param_group['lr'] = lr

class DataAugmentation:
    """Advanced data augmentation for recommendation systems"""
    
    @staticmethod
    def mixup_batch(users: torch.Tensor, movies: torch.Tensor, ratings: torch.Tensor,
                   alpha: float = 0.2) -> tuple:
        """Mixup augmentation for collaborative filtering"""
        if alpha <= 0:
            return users, movies, ratings
            
        batch_size = users.size(0)
        lam = np.random.beta(alpha, alpha)
        
        # Random permutation
        indices = torch.randperm(batch_size)
        
        # Mix ratings
        mixed_ratings = lam * ratings + (1 - lam) * ratings[indices]
        
        # For discrete user/movie IDs, randomly choose between pairs
        mask = torch.rand(batch_size) < lam
        mixed_users = torch.where(mask, users, users[indices])
        mixed_movies = torch.where(mask, movies, movies[indices])
        
        return mixed_users, mixed_movies, mixed_ratings
    
    @staticmethod
    def add_noise(ratings: torch.Tensor, noise_factor: float = 0.05) -> torch.Tensor:
        """Add Gaussian noise to ratings"""
        if noise_factor <= 0:
            return ratings
            
        noise = torch.randn_like(ratings) * noise_factor
        # Clamp to valid rating range [0.5, 5.0]
        noisy_ratings = torch.clamp(ratings + noise, 0.5, 5.0)
        return noisy_ratings

def create_next_experiment_config() -> Dict[str, Any]:
    """Create the complete configuration for next experiment"""
    config = AdvancedTrainingConfig()
    
    return {
        'experiment_name': 'advanced_regularization_v2',
        'target_rmse': 0.55,
        'training': config.__dict__,
        'description': """
        Advanced experiment targeting RMSE ≤ 0.55 with:
        
        1. REGULARIZATION IMPROVEMENTS:
           - Aggressive dropout (0.4)
           - Strong weight decay (1e-4)
           - Label smoothing (0.1)
           - Batch + Layer normalization
        
        2. ARCHITECTURE ENHANCEMENTS:
           - Deeper network (4 hidden layers)
           - Larger capacity (200 factors)
           - Bigger latent space (128 dims)
        
        3. TRAINING OPTIMIZATIONS:
           - AdamW optimizer with amsgrad
           - Cosine warmup scheduling
           - Advanced beta scheduling
           - Free bits for KL stability
        
        4. DATA AUGMENTATION:
           - Mixup for implicit regularization
           - Gaussian noise injection
           - Larger batch sizes (2048)
        
        5. ENSEMBLE STRATEGY:
           - Train 5 diverse models
           - Weighted ensemble averaging
        
        Expected Results:
        - RMSE: 0.45-0.55 (significant improvement)
        - Better generalization (val/train gap < 0.1)
        - Stable training without early stopping
        - Improved ranking metrics
        """
    }

# Usage example
if __name__ == "__main__":
    config = create_next_experiment_config()
    print("Next Experiment Configuration:")
    print(f"Target RMSE: {config['target_rmse']}")
    print(f"Description: {config['description']}")