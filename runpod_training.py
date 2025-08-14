"""
Runpod A100 Optimized Training Script for Hybrid VAE
Fixes NaN issues and maximizes A100 GPU utilization
"""

import os
import sys
import time
import pickle
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class A100OptimizedHybridVAE(nn.Module):
    """
    A100-Optimized Hybrid VAE with numerical stability fixes
    Designed for maximum GPU utilization and stable training
    """
    
    def __init__(self, n_users, n_movies, n_factors=128, hidden_dims=[512, 256, 128], 
                 latent_dim=64, dropout_rate=0.15):
        super(A100OptimizedHybridVAE, self).__init__()
        
        self.n_users = n_users
        self.n_movies = n_movies
        self.n_factors = n_factors
        self.latent_dim = latent_dim
        
        # Embedding layers - optimized for A100
        self.user_embedding = nn.Embedding(n_users, n_factors)
        self.movie_embedding = nn.Embedding(n_movies, n_factors)
        self.embedding_dropout = nn.Dropout(dropout_rate * 0.5)
        
        # Encoder with proper initialization
        encoder_layers = []
        input_dim = n_factors * 2
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),  # LayerNorm better for A100
                nn.GELU(),  # GELU optimized on A100
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # VAE latent layers
        self.mu_layer = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar_layer = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        input_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(input_dim, 1))
        self.decoder = nn.Sequential(*decoder_layers)
        
        self._init_weights()
        
    def _init_weights(self):
        """Proper weight initialization for stability"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier uniform for GELU activation
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def encode(self, users, movies):
        user_emb = self.user_embedding(users)
        movie_emb = self.movie_embedding(movies)
        
        features = torch.cat([user_emb, movie_emb], dim=1)
        features = self.embedding_dropout(features)
        
        encoded = self.encoder(features)
        mu = self.mu_layer(encoded)
        logvar = self.logvar_layer(encoded)
        
        # Clamp logvar for numerical stability
        logvar = torch.clamp(logvar, -10.0, 10.0)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, users, movies, rating_range=(0.5, 5.0)):
        mu, logvar = self.encode(users, movies)
        z = self.reparameterize(mu, logvar)
        rating_pred = self.decode(z)
        
        if rating_range:
            min_r, max_r = rating_range
            rating_pred = torch.sigmoid(rating_pred) * (max_r - min_r) + min_r
            
        return rating_pred, mu, logvar


def stable_vae_loss(predictions, targets, mu, logvar, kl_weight=0.1):
    """Numerically stable VAE loss"""
    # MSE reconstruction loss
    recon_loss = F.mse_loss(predictions.squeeze(), targets, reduction='mean')
    
    # KL divergence with clamping
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = torch.clamp(kl_loss, 0, 100)  # Prevent explosion
    
    total_loss = recon_loss + kl_weight * kl_loss
    
    return total_loss, recon_loss, kl_loss


def setup_a100_environment():
    """Setup optimal environment for A100 training"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available!")
    
    device = torch.device('cuda')
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # A100 optimizations
    torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
    torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for A100
    torch.backends.cudnn.allow_tf32 = True
    
    return device


def create_optimized_data_loaders(data_path, batch_size=16384):
    """Create A100-optimized data loaders"""
    logger.info("Loading pre-split training data...")
    
    # Load data
    train_df = pd.read_csv(os.path.join(data_path, 'train_data.csv'))
    val_df = pd.read_csv(os.path.join(data_path, 'val_data.csv'))
    
    with open(os.path.join(data_path, 'data_mappings.pkl'), 'rb') as f:
        mappings = pickle.load(f)
    
    logger.info(f"Training samples: {len(train_df):,}")
    logger.info(f"Validation samples: {len(val_df):,}")
    logger.info(f"Users: {len(mappings['user_to_index']):,}")
    logger.info(f"Movies: {len(mappings['movie_to_index']):,}")
    
    # Map to indices
    train_df['user_idx'] = train_df['user_id'].map(mappings['user_to_index'])
    train_df['movie_idx'] = train_df['movie_id'].map(mappings['movie_to_index'])
    val_df['user_idx'] = val_df['user_id'].map(mappings['user_to_index'])
    val_df['movie_idx'] = val_df['movie_id'].map(mappings['movie_to_index'])
    
    # Convert to tensors
    X_train = torch.LongTensor(train_df[['user_idx', 'movie_idx']].values)
    y_train = torch.FloatTensor(train_df['rating'].values)
    X_val = torch.LongTensor(val_df[['user_idx', 'movie_idx']].values)
    y_val = torch.FloatTensor(val_df['rating'].values)
    
    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    # A100-optimized data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8,  # A100 can handle more workers
        persistent_workers=True,
        prefetch_factor=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    n_users = len(mappings['user_to_index'])
    n_movies = len(mappings['movie_to_index'])
    
    return train_loader, val_loader, n_users, n_movies


def train_epoch(model, train_loader, optimizer, scaler, device, epoch, kl_weight):
    """Train one epoch with A100 optimizations"""
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    batches = 0
    
    for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
        batch_x, batch_y = batch_x.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with autocast():
            predictions, mu, logvar = model(batch_x[:, 0], batch_x[:, 1])
            loss, recon_loss, kl_loss = stable_vae_loss(predictions, batch_y, mu, logvar, kl_weight)
        
        # Mixed precision backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Accumulate metrics
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        batches += 1
        
        # Log progress
        if batch_idx % 100 == 0:
            logger.info(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    return total_loss / batches, total_recon / batches, total_kl / batches


def validate_model(model, val_loader, device, kl_weight):
    """Validate model"""
    model.eval()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    batches = 0
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
            
            with autocast():
                predictions, mu, logvar = model(batch_x[:, 0], batch_x[:, 1])
                loss, recon_loss, kl_loss = stable_vae_loss(predictions, batch_y, mu, logvar, kl_weight)
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            batches += 1
    
    return total_loss / batches, total_recon / batches, total_kl / batches


def main():
    """Main training function optimized for Runpod A100"""
    
    # Configuration
    config = {
        'data_path': '/workspace/data',  # Standard Runpod path
        'batch_size': 16384,  # Large batch for A100
        'n_epochs': 100,
        'lr': 5e-4,
        'weight_decay': 1e-5,
        'kl_weight_start': 0.0,
        'kl_weight_end': 0.1,
        'kl_warmup_epochs': 20,
        'patience': 15,
        'save_path': '/workspace/models/hybrid_vae_fixed.pt'
    }
    
    logger.info("Starting A100-Optimized Hybrid VAE Training")
    logger.info("=" * 60)
    
    # Setup environment
    device = setup_a100_environment()
    
    # Create data loaders
    train_loader, val_loader, n_users, n_movies = create_optimized_data_loaders(
        config['data_path'], config['batch_size']
    )
    
    # Create model
    model = A100OptimizedHybridVAE(
        n_users=n_users,
        n_movies=n_movies,
        n_factors=128,
        hidden_dims=[512, 256, 128],
        latent_dim=64,
        dropout_rate=0.15
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and scaler
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scaler = GradScaler()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    start_time = time.time()
    
    for epoch in range(config['n_epochs']):
        # KL weight scheduling
        if epoch < config['kl_warmup_epochs']:
            kl_weight = config['kl_weight_start'] + (config['kl_weight_end'] - config['kl_weight_start']) * (epoch / config['kl_warmup_epochs'])
        else:
            kl_weight = config['kl_weight_end']
        
        # Training
        train_loss, train_recon, train_kl = train_epoch(
            model, train_loader, optimizer, scaler, device, epoch, kl_weight
        )
        
        # Validation
        val_loss, val_recon, val_kl = validate_model(model, val_loader, device, kl_weight)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Calculate RMSE
        train_rmse = np.sqrt(train_recon)
        val_rmse = np.sqrt(val_recon)
        
        # Logging
        elapsed = (time.time() - start_time) / 60
        logger.info(
            f'Epoch {epoch+1:03d}/{config["n_epochs"]:03d} | '
            f'Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f} | '
            f'KL Weight: {kl_weight:.3f} | Time: {elapsed:.1f}min'
        )
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Create save directory
            os.makedirs(os.path.dirname(config['save_path']), exist_ok=True)
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'val_rmse': val_rmse,
                'config': config,
                'n_users': n_users,
                'n_movies': n_movies
            }, config['save_path'])
            
            logger.info(f'✅ Model saved! Val RMSE: {val_rmse:.4f}')
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['patience']:
            logger.info(f'Early stopping at epoch {epoch+1}')
            break
    
    # Final results
    total_time = (time.time() - start_time) / 60
    final_rmse = np.sqrt(best_val_loss)
    
    logger.info("=" * 60)
    logger.info(f"✅ Training completed in {total_time:.1f} minutes")
    logger.info(f"🏆 Best validation RMSE: {final_rmse:.4f}")
    logger.info(f"💾 Model saved to: {config['save_path']}")
    
    # Save training summary
    summary = {
        'final_rmse': final_rmse,
        'total_time_minutes': total_time,
        'epochs_trained': epoch + 1,
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'gpu_used': torch.cuda.get_device_name(0),
        'batch_size': config['batch_size']
    }
    
    with open('/workspace/models/training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("Training summary saved!")


if __name__ == "__main__":
    main()