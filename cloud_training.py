#!/usr/bin/env python3
"""
Cloud Training Script for Hybrid VAE Recommendation System
Optimized for RunPod/Lambda Labs deployment
"""

import os
import sys
import time
import pickle
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import wandb  # For experiment tracking 

# Set random seeds for reproducibility
def set_random_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Import your model classes (copy from notebook)
class HybridVAE(nn.Module):
    """Hybrid VAE + Embedding model - copy from your notebook"""
    def __init__(self, n_users, n_movies, n_factors=150, 
                 hidden_dims=[512, 256], latent_dim=64, 
                 dropout_rate=0.3):
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
        
        # Decoder network
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
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.uniform_(self.user_embedding.weight, -0.05, 0.05)
        nn.init.uniform_(self.movie_embedding.weight, -0.05, 0.05)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)
    
    def encode(self, users, movies):
        user_emb = self.user_embedding(users)
        movie_emb = self.movie_embedding(movies)
        features = torch.cat([user_emb, movie_emb], dim=1)
        features = self.embedding_dropout(features)
        encoded = self.encoder(features)
        mu = self.mu_layer(encoded)
        logvar = self.logvar_layer(encoded)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def decode(self, z):
        decoded = self.decoder(z)
        return torch.sigmoid(decoded)
    
    def forward(self, users, movies, minmax=None):
        mu, logvar = self.encode(users, movies)
        z = self.reparameterize(mu, logvar)
        rating_pred = self.decode(z)
        
        if minmax is not None:
            min_rating, max_rating = minmax
            rating_pred = rating_pred * (max_rating - min_rating) + min_rating
            
        return rating_pred, mu, logvar

def vae_loss_function(predictions, targets, mu, logvar, kl_weight=1.0):
    recon_loss = F.mse_loss(predictions.squeeze(), targets, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + kl_weight * kl_loss
    return total_loss, recon_loss, kl_loss

def create_data_loaders(data_path, batch_size=512):
    """Load pre-split training data to prevent data leakage"""
    print("Loading pre-split training data...")
    
    # Load the pre-split CSV files (NO newer data included)
    train_path = os.path.join(data_path, 'train_data.csv')
    val_path = os.path.join(data_path, 'val_data.csv')
    mappings_path = os.path.join(data_path, 'data_mappings.pkl')
    
    if not all(os.path.exists(p) for p in [train_path, val_path, mappings_path]):
        raise FileNotFoundError(
            f"Required files not found in {data_path}:\n"
            f"- train_data.csv\n- val_data.csv\n- data_mappings.pkl\n"
            f"Please run the data export cell in your notebook first!"
        )
    
    # Load training and validation data
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    # Load mappings
    with open(mappings_path, 'rb') as f:
        mappings = pickle.load(f)
    
    n_users = mappings['n_users']
    n_movies = mappings['n_movies']
    
    print(f"✅ Training data: {len(train_df):,} samples")
    print(f"✅ Validation data: {len(val_df):,} samples")
    print(f"✅ Users: {n_users:,}, Movies: {n_movies:,}")
    print("🚨 NOTE: Newer (ETL test) data excluded to prevent leakage!")
    
    # Convert to tensors
    X_train = torch.LongTensor(train_df[['user_id', 'movie_id']].values)
    y_train = torch.FloatTensor(train_df['rating'].values)
    
    X_val = torch.LongTensor(val_df[['user_id', 'movie_id']].values)
    y_val = torch.FloatTensor(val_df['rating'].values)
    
    # Create datasets and loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    return train_loader, val_loader, n_users, n_movies

def train_model(config):
    """Main training function"""
    print(f"Starting training with config: {config}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Set random seed
    set_random_seed(config['seed'])
    
    # Initialize wandb (optional)
    if config.get('use_wandb', False):
        wandb.init(project="movielens-hybrid-vae", config=config)
    
    # Load data (pre-split to prevent leakage)
    train_loader, val_loader, n_users, n_movies = create_data_loaders(
        config['data_path'], config['batch_size']
    )
    
    # Create model
    model = HybridVAE(
        n_users=n_users,
        n_movies=n_movies,
        n_factors=config['n_factors'],
        hidden_dims=config['hidden_dims'],
        latent_dim=config['latent_dim'],
        dropout_rate=config['dropout_rate']
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    minmax = (0.5, 5.0)  # Rating range for MovieLens
    
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(config['n_epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            predictions, mu, logvar = model(batch_x[:, 0], batch_x[:, 1], minmax)
            total_loss, recon_loss, kl_loss = vae_loss_function(
                predictions, batch_y, mu, logvar, config['kl_weight']
            )
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += total_loss.item()
            train_batches += 1
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                predictions, mu, logvar = model(batch_x[:, 0], batch_x[:, 1], minmax)
                total_loss, _, _ = vae_loss_function(
                    predictions, batch_y, mu, logvar, config['kl_weight']
                )
                val_loss += total_loss.item()
                val_batches += 1
        
        # Average losses
        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / val_batches
        
        # Update scheduler
        scheduler.step(avg_val_loss)
        
        # Log metrics
        if config.get('use_wandb', False):
            wandb.log({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': avg_val_loss,
                'config': config
            }, config['save_path'])
            print(f'✓ Epoch {epoch+1:03d}: Val loss improved to {avg_val_loss:.4f}')
        else:
            patience_counter += 1
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            elapsed = (time.time() - start_time) / 60
            print(f'Epoch {epoch+1:03d}/{config["n_epochs"]:03d} | '
                  f'Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | '
                  f'LR: {optimizer.param_groups[0]["lr"]:.2e} | '
                  f'Time: {elapsed:.1f}min')
        
        # Early stopping
        if patience_counter >= config['patience']:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    total_time = (time.time() - start_time) / 60
    print(f'Training completed in {total_time:.1f} minutes')
    print(f'Best validation loss: {best_val_loss:.4f}')
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Train Hybrid VAE Recommender')
    parser.add_argument('--data_path', type=str, default='/workspace/data', 
                       help='Path to pre-split training data (train_data.csv, val_data.csv, data_mappings.pkl)')
    parser.add_argument('--save_path', type=str, default='/workspace/hybrid_vae_best.pt',
                       help='Path to save best model')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--use_wandb', action='store_true', help='Use wandb logging')
    
    args = parser.parse_args()
    
    # Training configuration
    config = {
        'data_path': args.data_path,
        'save_path': args.save_path,
        'batch_size': args.batch_size,
        'n_epochs': args.n_epochs,
        'lr': args.lr,
        'weight_decay': 1e-5,
        'n_factors': 150,
        'hidden_dims': [512, 256],
        'latent_dim': 64,
        'dropout_rate': 0.3,
        'kl_weight': 1.0,
        'patience': 15,
        'seed': 42,
        'use_wandb': args.use_wandb
    }
    
    # Train model
    model = train_model(config)
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()