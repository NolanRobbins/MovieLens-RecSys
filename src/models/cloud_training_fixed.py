"""
FIXED MovieLens Hybrid VAE Training Script
Corrects critical issues causing RMSE 1.21 performance problem

Key Fixes:
1. MSE loss uses 'mean' instead of 'sum' reduction (CRITICAL FIX)
2. Reduced KL weight from 1.0 to 0.1 
3. Increased learning rate from 1e-4 to 5e-4
4. Reduced dropout from 0.3 to 0.15
5. Added more epochs and better scheduler

Expected Results: RMSE should drop to 0.85-0.95
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import pickle
import os
import time
import argparse
from pathlib import Path

try:
    import wandb
except ImportError:
    wandb = None

class HybridVAE(nn.Module):
    """FIXED Hybrid VAE + Embedding model"""
    def __init__(self, n_users, n_movies, n_factors=150, 
                 hidden_dims=[512, 256, 128], latent_dim=64, 
                 dropout_rate=0.15):  # FIXED: Reduced default dropout
        super(HybridVAE, self).__init__()
        
        # Embedding layers
        self.user_embedding = nn.Embedding(n_users, n_factors)
        self.movie_embedding = nn.Embedding(n_movies, n_factors)
        self.embedding_dropout = nn.Dropout(dropout_rate * 0.5)
        
        # Encoder network
        encoder_layers = []
        input_dim = n_factors * 2
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim
            
        self.encoder = nn.Sequential(*encoder_layers)
        
        # VAE latent space
        self.mu_layer = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar_layer = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        input_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim
            
        decoder_layers.append(nn.Linear(input_dim, 1))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Improved weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.1)
    
    def encode(self, users, movies):
        user_emb = self.user_embedding(users)
        movie_emb = self.movie_embedding(movies)
        
        x = torch.cat([user_emb, movie_emb], dim=1)
        x = self.embedding_dropout(x)
        
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        
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
    
    def forward(self, users, movies, minmax=None):
        mu, logvar = self.encode(users, movies)
        z = self.reparameterize(mu, logvar)
        rating_pred = self.decode(z)
        
        if minmax is not None:
            min_rating, max_rating = minmax
            rating_pred = torch.sigmoid(rating_pred) * (max_rating - min_rating) + min_rating
            
        return rating_pred, mu, logvar

def fixed_vae_loss_function(predictions, targets, mu, logvar, kl_weight=0.1):
    """FIXED VAE loss function with proper scaling"""
    # CRITICAL FIX #1: Use 'mean' instead of 'sum' reduction
    recon_loss = F.mse_loss(predictions.squeeze(), targets, reduction='mean')
    
    # FIXED: Proper KL divergence scaling
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    total_loss = recon_loss + kl_weight * kl_loss
    return total_loss, recon_loss, kl_loss

def create_data_loaders(data_path, batch_size=1024):
    """Load pre-split training data"""
    print("Loading pre-split training data...")
    
    train_path = os.path.join(data_path, 'train_data.csv')
    val_path = os.path.join(data_path, 'val_data.csv')
    mappings_path = os.path.join(data_path, 'data_mappings.pkl')
    
    if not all(os.path.exists(p) for p in [train_path, val_path, mappings_path]):
        raise FileNotFoundError(
            f"Required files not found in {data_path}:\n"
            f"- train_data.csv\n- val_data.csv\n- data_mappings.pkl"
        )
    
    # Load data
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    with open(mappings_path, 'rb') as f:
        mappings = pickle.load(f)
    
    print(f"Training samples: {len(train_df):,}")
    print(f"Validation samples: {len(val_df):,}")
    print(f"Users: {len(mappings['user_to_idx']):,}")
    print(f"Movies: {len(mappings['movie_to_idx']):,}")
    
    # Create datasets
    train_x = torch.tensor(train_df[['user_idx', 'movie_idx']].values, dtype=torch.long)
    train_y = torch.tensor(train_df['rating'].values, dtype=torch.float32)
    
    val_x = torch.tensor(val_df[['user_idx', 'movie_idx']].values, dtype=torch.long)
    val_y = torch.tensor(val_df['rating'].values, dtype=torch.float32)
    
    train_dataset = TensorDataset(train_x, train_y)
    val_dataset = TensorDataset(val_x, val_y)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    n_users = len(mappings['user_to_idx'])
    n_movies = len(mappings['movie_to_idx'])
    
    return train_loader, val_loader, n_users, n_movies

def train_model_fixed(config):
    """FIXED training function with corrected hyperparameters"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set seeds for reproducibility
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
    
    # Initialize wandb if available and requested
    if config['use_wandb'] and wandb is not None:
        wandb.init(
            project="movielens-hybrid-vae-experiments", 
            name="fixed-training-v1",
            config=config,
            tags=["fixed", "improved", "production"]
        )
    
    # Load data
    train_loader, val_loader, n_users, n_movies = create_data_loaders(
        config['data_path'], config['batch_size']
    )
    
    # Create FIXED model with improved hyperparameters
    model = HybridVAE(
        n_users=n_users,
        n_movies=n_movies,
        n_factors=config['n_factors'],
        hidden_dims=config['hidden_dims'],
        latent_dim=config['latent_dim'],
        dropout_rate=config['dropout_rate']  # Now using reduced dropout
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # FIXED: Better optimizer settings
    optimizer = optim.AdamW(model.parameters(), 
                           lr=config['lr'], 
                           weight_decay=config['weight_decay'])
    
    # IMPROVED: Better learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=8, factor=0.5, min_lr=1e-6, verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    minmax = (0.5, 5.0)  # Rating range for MovieLens
    
    print("Starting FIXED training...")
    print(f"Expected RMSE improvement: 1.21 → 0.85-0.95")
    start_time = time.time()
    
    for epoch in range(config['n_epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_recon_loss = 0.0
        train_kl_loss = 0.0
        train_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            predictions, mu, logvar = model(batch_x[:, 0], batch_x[:, 1], minmax)
            
            # Use FIXED loss function
            total_loss, recon_loss, kl_loss = fixed_vae_loss_function(
                predictions, batch_y, mu, logvar, config['kl_weight']
            )
            
            optimizer.zero_grad()
            total_loss.backward()
            # IMPROVED: Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += total_loss.item()
            train_recon_loss += recon_loss.item()
            train_kl_loss += kl_loss.item()
            train_batches += 1
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_recon_loss = 0.0
        val_kl_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                predictions, mu, logvar = model(batch_x[:, 0], batch_x[:, 1], minmax)
                total_loss, recon_loss, kl_loss = fixed_vae_loss_function(
                    predictions, batch_y, mu, logvar, config['kl_weight']
                )
                
                val_loss += total_loss.item()
                val_recon_loss += recon_loss.item()
                val_kl_loss += kl_loss.item()
                val_batches += 1
        
        # Calculate averages
        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / val_batches
        avg_train_recon = train_recon_loss / train_batches
        avg_val_recon = val_recon_loss / val_batches
        
        # Calculate RMSE for monitoring
        train_rmse = np.sqrt(avg_train_recon)
        val_rmse = np.sqrt(avg_val_recon)
        
        # Log to wandb
        if config['use_wandb'] and wandb is not None:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_rmse': train_rmse,
                'val_rmse': val_rmse,
                'train_kl': train_kl_loss / train_batches,
                'val_kl': val_kl_loss / val_batches,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': avg_val_loss,
                'val_rmse': val_rmse,
                'config': config
            }, config['save_path'])
            print(f'✓ Epoch {epoch+1:03d}: Val RMSE improved to {val_rmse:.4f} (loss: {avg_val_loss:.4f})')
        else:
            patience_counter += 1
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            elapsed = (time.time() - start_time) / 60
            print(f'Epoch {epoch+1:03d}/{config["n_epochs"]:03d} | '
                  f'Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f} | '
                  f'LR: {optimizer.param_groups[0]["lr"]:.2e} | '
                  f'Time: {elapsed:.1f}min')
        
        # Early stopping
        if patience_counter >= config['patience']:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    total_time = (time.time() - start_time) / 60
    final_rmse = np.sqrt(best_val_loss)
    print(f'Training completed in {total_time:.1f} minutes')
    print(f'Best validation RMSE: {final_rmse:.4f}')
    print(f'Expected improvement: {((1.21 - final_rmse) / 1.21 * 100):.1f}% better than before')
    
    return model

def main():
    parser = argparse.ArgumentParser(description='FIXED Hybrid VAE Recommender Training')
    parser.add_argument('--data_path', type=str, default='data/processed', 
                       help='Path to pre-split training data')
    parser.add_argument('--save_path', type=str, default='data/models/hybrid_vae_fixed.pt',
                       help='Path to save best model')
    parser.add_argument('--batch_size', type=int, default=1024)  # FIXED: Larger batch size
    parser.add_argument('--n_epochs', type=int, default=150)     # FIXED: More epochs
    parser.add_argument('--lr', type=float, default=5e-4)        # FIXED: Higher learning rate
    parser.add_argument('--use_wandb', action='store_true', help='Use wandb logging')
    
    args = parser.parse_args()
    
    # FIXED training configuration
    config = {
        'data_path': args.data_path,
        'save_path': args.save_path,
        'batch_size': args.batch_size,
        'n_epochs': args.n_epochs,
        'lr': args.lr,
        'weight_decay': 1e-5,
        'n_factors': 150,
        'hidden_dims': [512, 256, 128],  # IMPROVED: Added third layer
        'latent_dim': 64,
        'dropout_rate': 0.15,            # FIXED: Reduced from 0.3
        'kl_weight': 0.1,               # FIXED: Reduced from 1.0
        'patience': 20,                  # IMPROVED: More patience
        'seed': 42,
        'use_wandb': args.use_wandb
    }
    
    print("🔧 FIXED VAE Training Configuration:")
    print("=" * 50)
    for key, value in config.items():
        print(f"{key}: {value}")
    print("=" * 50)
    print("Key fixes applied:")
    print("✅ MSE loss uses 'mean' instead of 'sum' reduction")
    print("✅ KL weight reduced from 1.0 → 0.1")
    print("✅ Learning rate increased from 1e-4 → 5e-4") 
    print("✅ Dropout reduced from 0.3 → 0.15")
    print("✅ Improved weight initialization and scheduler")
    print("=" * 50)
    
    # Train FIXED model
    model = train_model_fixed(config)
    
    print("🎉 FIXED training completed successfully!")
    print("Expected RMSE: 0.85-0.95 (significant improvement from 1.21)")

if __name__ == "__main__":
    main()