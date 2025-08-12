#!/usr/bin/env python3
"""
Ranking-Optimized Model Training
Integrates ranking_loss_functions.py for improved recommendation quality
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import pickle
import json
import time
from pathlib import Path
import argparse
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt

# Import our components
from ranking_loss_functions import HybridVAERanking, RankingLosses, train_with_ranking_loss
from evaluate_model import HybridVAE
from ranking_metrics_evaluation import RecommenderMetrics

class RankingOptimizedExperiment:
    """
    Enhanced training experiment with ranking optimization
    Compares vanilla VAE vs ranking-optimized VAE
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics_evaluator = RecommenderMetrics()
        
        # Load data
        self.train_data, self.val_data, self.mappings = self.load_data()
        self.train_loader, self.val_loader = self.create_data_loaders()
        
    def load_data(self):
        """Load processed training data"""
        print("📚 Loading processed data...")
        
        # Load datasets
        train_data = pd.read_csv('processed_data/train_data.csv')
        val_data = pd.read_csv('processed_data/val_data.csv')
        
        # Load mappings
        with open('processed_data/data_mappings.pkl', 'rb') as f:
            mappings = pickle.load(f)
        
        print(f"✅ Loaded {len(train_data)} training samples, {len(val_data)} validation samples")
        
        return train_data, val_data, mappings
    
    def create_data_loaders(self):
        """Create PyTorch data loaders"""
        print("🔄 Creating data loaders...")
        
        # Prepare training data
        train_users = self.train_data['userId'].map(self.mappings['user_mappings']).values
        train_movies = self.train_data['movieId'].map(self.mappings['movie_mappings']).values
        train_ratings = self.train_data['rating'].values
        
        # Filter out unmapped entries
        valid_mask = (~pd.isna(train_users)) & (~pd.isna(train_movies))
        train_users = train_users[valid_mask].astype(int)
        train_movies = train_movies[valid_mask].astype(int)
        train_ratings = train_ratings[valid_mask].astype(np.float32)
        
        # Create datasets
        train_dataset = TensorDataset(
            torch.tensor(np.column_stack([train_users, train_movies]), dtype=torch.long),
            torch.tensor(train_ratings, dtype=torch.float32)
        )
        
        # Same for validation
        val_users = self.val_data['userId'].map(self.mappings['user_mappings']).values
        val_movies = self.val_data['movieId'].map(self.mappings['movie_mappings']).values
        val_ratings = self.val_data['rating'].values
        
        valid_mask = (~pd.isna(val_users)) & (~pd.isna(val_movies))
        val_users = val_users[valid_mask].astype(int)
        val_movies = val_movies[valid_mask].astype(int)
        val_ratings = val_ratings[valid_mask].astype(np.float32)
        
        val_dataset = TensorDataset(
            torch.tensor(np.column_stack([val_users, val_movies]), dtype=torch.long),
            torch.tensor(val_ratings, dtype=torch.float32)
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0
        )
        
        print(f"✅ Created data loaders: {len(train_loader)} train batches, {len(val_loader)} val batches")
        
        return train_loader, val_loader
    
    def create_models(self) -> Tuple[nn.Module, nn.Module]:
        """Create vanilla and ranking-optimized models"""
        
        n_users = len(self.mappings['user_mappings'])
        n_movies = len(self.mappings['movie_mappings'])
        
        # Vanilla VAE model
        vanilla_model = HybridVAE(
            n_users=n_users,
            n_movies=n_movies,
            n_factors=self.config['n_factors'],
            hidden_dims=self.config['hidden_dims'],
            latent_dim=self.config['latent_dim'],
            dropout_rate=self.config['dropout_rate']
        ).to(self.device)
        
        # Ranking-optimized model (same architecture, different loss)
        ranking_model = HybridVAE(
            n_users=n_users,
            n_movies=n_movies,
            n_factors=self.config['n_factors'],
            hidden_dims=self.config['hidden_dims'],
            latent_dim=self.config['latent_dim'],
            dropout_rate=self.config['dropout_rate']
        ).to(self.device)
        
        print(f"✅ Created models with {n_users:,} users and {n_movies:,} movies")
        
        return vanilla_model, ranking_model
    
    def train_vanilla_model(self, model: nn.Module, epochs: int = 10) -> Dict:
        """Train model with standard VAE loss"""
        print(f"\n🔥 Training vanilla VAE model for {epochs} epochs...")
        
        optimizer = optim.Adam(model.parameters(), lr=self.config['lr'])
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            model.train()
            epoch_train_loss = 0
            
            for batch_data, batch_ratings in self.train_loader:
                batch_data = batch_data.to(self.device)
                batch_ratings = batch_ratings.to(self.device)
                
                users = batch_data[:, 0]
                movies = batch_data[:, 1]
                
                optimizer.zero_grad()
                
                predictions, mu, logvar = model(users, movies, (0.5, 5.0))
                
                # Standard VAE loss
                recon_loss = nn.MSELoss()(predictions.squeeze(), batch_ratings)
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                total_loss = recon_loss + self.config['kl_weight'] * kl_loss
                
                total_loss.backward()
                optimizer.step()
                
                epoch_train_loss += total_loss.item()
            
            avg_train_loss = epoch_train_loss / len(self.train_loader)
            
            # Validation
            model.eval()
            epoch_val_loss = 0
            
            with torch.no_grad():
                for batch_data, batch_ratings in self.val_loader:
                    batch_data = batch_data.to(self.device)
                    batch_ratings = batch_ratings.to(self.device)
                    
                    users = batch_data[:, 0]
                    movies = batch_data[:, 1]
                    
                    predictions, mu, logvar = model(users, movies, (0.5, 5.0))
                    
                    recon_loss = nn.MSELoss()(predictions.squeeze(), batch_ratings)
                    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    total_loss = recon_loss + self.config['kl_weight'] * kl_loss
                    
                    epoch_val_loss += total_loss.item()
            
            avg_val_loss = epoch_val_loss / len(self.val_loader)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_val_loss': val_losses[-1]
        }
    
    def train_ranking_model(self, model: nn.Module, epochs: int = 10) -> Dict:
        """Train model with ranking-optimized loss"""
        print(f"\n🎯 Training ranking-optimized VAE model for {epochs} epochs...")
        
        optimizer = optim.Adam(model.parameters(), lr=self.config['lr'])
        
        train_losses = []
        val_losses = []
        
        # Use ranking-optimized training
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train with ranking loss
            avg_train_loss = train_with_ranking_loss(
                model, self.train_loader, optimizer, self.device,
                ranking_weight=self.config['ranking_weight'],
                kl_weight=self.config['kl_weight']
            )
            
            # Validation (standard loss for comparison)
            model.eval()
            epoch_val_loss = 0
            
            with torch.no_grad():
                for batch_data, batch_ratings in self.val_loader:
                    batch_data = batch_data.to(self.device)
                    batch_ratings = batch_ratings.to(self.device)
                    
                    users = batch_data[:, 0]
                    movies = batch_data[:, 1]
                    
                    predictions, mu, logvar = model(users, movies, (0.5, 5.0))
                    
                    recon_loss = nn.MSELoss()(predictions.squeeze(), batch_ratings)
                    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    total_loss = recon_loss + self.config['kl_weight'] * kl_loss
                    
                    epoch_val_loss += total_loss.item()
            
            avg_val_loss = epoch_val_loss / len(self.val_loader)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f} ({epoch_time:.1f}s)")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_val_loss': val_losses[-1]
        }
    
    def compare_models(self, vanilla_model: nn.Module, ranking_model: nn.Module) -> Dict:
        """Compare vanilla vs ranking-optimized models"""
        print(f"\n📊 Comparing model performance...")
        
        results = {
            'vanilla_model': {},
            'ranking_model': {},
            'comparison': {}
        }
        
        # Test on a subset of validation data
        test_users = []
        test_movies = []
        test_ratings = []
        
        # Sample validation data
        val_sample = self.val_data.sample(n=min(1000, len(self.val_data)), random_state=42)
        
        for _, row in val_sample.iterrows():
            user_id = row['userId']
            movie_id = row['movieId']
            
            if (user_id in self.mappings['user_mappings'] and 
                movie_id in self.mappings['movie_mappings']):
                
                test_users.append(self.mappings['user_mappings'][user_id])
                test_movies.append(self.mappings['movie_mappings'][movie_id])
                test_ratings.append(row['rating'])
        
        if not test_users:
            print("❌ No valid test data found")
            return results
        
        test_users = torch.tensor(test_users, dtype=torch.long, device=self.device)
        test_movies = torch.tensor(test_movies, dtype=torch.long, device=self.device)
        test_ratings = np.array(test_ratings)
        
        # Evaluate both models
        for model, model_name in [(vanilla_model, 'vanilla_model'), (ranking_model, 'ranking_model')]:
            model.eval()
            
            with torch.no_grad():
                predictions, _, _ = model(test_users, test_movies, (0.5, 5.0))
                pred_ratings = predictions.cpu().numpy().flatten()
            
            # Calculate metrics
            mse = np.mean((pred_ratings - test_ratings) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(pred_ratings - test_ratings))
            
            # Ranking metrics (simplified)
            map_10 = self.metrics_evaluator.mean_average_precision_at_k(
                test_ratings, pred_ratings, k=min(10, len(pred_ratings))
            )
            
            mrr_10 = self.metrics_evaluator.mean_reciprocal_rank_at_k(
                test_ratings, pred_ratings, k=min(10, len(pred_ratings))
            )
            
            results[model_name] = {
                'rmse': float(rmse),
                'mae': float(mae),
                'mse': float(mse),
                'map@10': float(map_10),
                'mrr@10': float(mrr_10),
                'n_samples': len(test_ratings)
            }
            
            print(f"📈 {model_name.replace('_', ' ').title()} Results:")
            print(f"   RMSE: {rmse:.4f}")
            print(f"   MAE:  {mae:.4f}")
            print(f"   mAP@10: {map_10:.4f}")
            print(f"   MRR@10: {mrr_10:.4f}")
        
        # Calculate improvements
        vanilla_rmse = results['vanilla_model']['rmse']
        ranking_rmse = results['ranking_model']['rmse']
        rmse_improvement = (vanilla_rmse - ranking_rmse) / vanilla_rmse * 100
        
        vanilla_map = results['vanilla_model']['map@10']
        ranking_map = results['ranking_model']['map@10']
        map_improvement = (ranking_map - vanilla_map) / vanilla_map * 100 if vanilla_map > 0 else 0
        
        results['comparison'] = {
            'rmse_improvement_pct': float(rmse_improvement),
            'map_improvement_pct': float(map_improvement),
            'ranking_model_better': ranking_rmse < vanilla_rmse and ranking_map >= vanilla_map
        }
        
        print(f"\n🏆 Model Comparison:")
        print(f"   RMSE Improvement: {rmse_improvement:+.2f}%")
        print(f"   mAP@10 Improvement: {map_improvement:+.2f}%")
        print(f"   Ranking Model Better: {results['comparison']['ranking_model_better']}")
        
        return results
    
    def run_experiment(self) -> Dict:
        """Run complete ranking optimization experiment"""
        print("🚀 Starting ranking optimization experiment...")
        
        start_time = time.time()
        
        # Create models
        vanilla_model, ranking_model = self.create_models()
        
        # Train models
        vanilla_results = self.train_vanilla_model(vanilla_model, epochs=self.config['n_epochs'])
        ranking_results = self.train_ranking_model(ranking_model, epochs=self.config['n_epochs'])
        
        # Compare models
        comparison_results = self.compare_models(vanilla_model, ranking_model)
        
        # Save models
        vanilla_path = f"models/vanilla_vae_experiment.pt"
        ranking_path = f"models/ranking_vae_experiment.pt"
        
        torch.save({
            'model_state_dict': vanilla_model.state_dict(),
            'config': self.config,
            'results': vanilla_results
        }, vanilla_path)
        
        torch.save({
            'model_state_dict': ranking_model.state_dict(),
            'config': self.config,
            'results': ranking_results
        }, ranking_path)
        
        total_time = time.time() - start_time
        
        final_results = {
            'experiment_info': {
                'total_time': total_time,
                'config': self.config,
                'models_saved': [vanilla_path, ranking_path]
            },
            'vanilla_model_training': vanilla_results,
            'ranking_model_training': ranking_results,
            'model_comparison': comparison_results
        }
        
        print(f"\n✅ Experiment completed in {total_time:.2f} seconds")
        print(f"📁 Models saved to models/ directory")
        
        return final_results

def main():
    parser = argparse.ArgumentParser(description='Ranking-optimized training experiment')
    parser.add_argument('--config', default='ranking_experiment_config.json',
                       help='Configuration file')
    parser.add_argument('--output', default='ranking_experiment_results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Default configuration
    default_config = {
        'experiment_name': 'ranking_optimization_experiment',
        'batch_size': 512,
        'n_epochs': 5,  # Quick experiment
        'lr': 1e-4,
        'n_factors': 150,
        'hidden_dims': [512, 256, 128],
        'latent_dim': 64,
        'dropout_rate': 0.3,
        'kl_weight': 0.01,
        'ranking_weight': 0.1  # Weight for ranking loss
    }
    
    # Load config if exists
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        config = {**default_config, **config}
    else:
        config = default_config
        # Save default config
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"📄 Created default config: {config_path}")
    
    # Run experiment
    experiment = RankingOptimizedExperiment(config)
    results = experiment.run_experiment()
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📄 Results saved to {args.output}")
    print("🎉 Ranking optimization experiment complete!")

if __name__ == "__main__":
    main()