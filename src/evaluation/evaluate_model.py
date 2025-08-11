#!/usr/bin/env python3
"""
Model Evaluation Script for Experiment 2
Tests the trained model performance on validation and test data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import time
from typing import Dict, List, Tuple
import argparse
from tqdm import tqdm

# Import model architecture (same as in inference_api.py)
class HybridVAE(nn.Module):
    """Hybrid VAE + Embedding model"""
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
        
    def forward(self, users, movies, minmax=None):
        user_emb = self.user_embedding(users)
        movie_emb = self.movie_embedding(movies)
        features = torch.cat([user_emb, movie_emb], dim=1)
        features = self.embedding_dropout(features)
        encoded = self.encoder(features)
        mu = self.mu_layer(encoded)
        logvar = self.logvar_layer(encoded)
        
        # Reparameterize (always use mean during evaluation)  
        z = mu  # No sampling during evaluation
            
        decoded = self.decoder(z)
        rating_pred = torch.sigmoid(decoded)
        
        if minmax is not None:
            min_rating, max_rating = minmax
            rating_pred = rating_pred * (max_rating - min_rating) + min_rating
            
        return rating_pred, mu, logvar

class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, model_path: str, data_dir: str = "data/processed"):
        self.model_path = Path(model_path)
        self.data_dir = Path(data_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = None
        self.mappings = None
        self.movies_df = None
        self.val_data = None
        self.test_data = None
        
    def load_model_and_data(self):
        """Load trained model and evaluation data"""
        print(f"🔄 Loading model from {self.model_path}")
        
        # Load model checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        config = checkpoint['config']
        
        # Create and load model
        self.model = HybridVAE(
            n_users=checkpoint['n_users'],
            n_movies=checkpoint['n_movies'],
            n_factors=config['n_factors'],
            hidden_dims=config['hidden_dims'],
            latent_dim=config['latent_dim'],
            dropout_rate=config['dropout_rate']
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded successfully!")
        print(f"   - Experiment: {checkpoint['experiment_name']}")
        print(f"   - Epoch: {checkpoint['epoch']}")
        print(f"   - Training Validation Loss: {checkpoint['val_loss']:.4f}")
        print(f"   - Architecture: {config['n_factors']}D embeddings, {config['hidden_dims']} hidden, {config['latent_dim']}D latent")
        
        # Load mappings
        mappings_path = self.data_dir / "data_mappings.pkl"
        with open(mappings_path, 'rb') as f:
            self.mappings = pickle.load(f)
        print(f"✅ Mappings loaded: {len(self.mappings['user_to_index'])} users, {len(self.mappings['movie_to_index'])} movies")
        
        # Load movies data
        try:
            self.movies_df = pd.read_csv(self.data_dir / "movies_processed.csv")
            print(f"✅ Movies data loaded: {len(self.movies_df)} movies")
        except:
            print("⚠️  Movies data not found")
            self.movies_df = None
        
        # Load validation and test data
        try:
            self.val_data = pd.read_csv(self.data_dir / "val_data.csv")
            print(f"✅ Validation data loaded: {len(self.val_data)} samples")
        except:
            print("⚠️  Validation data not found")
            
        try:
            self.test_data = pd.read_csv(self.data_dir / "test_data.csv")
            print(f"✅ Test data loaded: {len(self.test_data)} samples")
        except:
            print("⚠️  Test data not found")
    
    def evaluate_on_dataset(self, data: pd.DataFrame, dataset_name: str, max_samples: int = 10000) -> Dict:
        """Evaluate model on a dataset"""
        print(f"\n📊 Evaluating on {dataset_name} data...")
        
        if data is None or len(data) == 0:
            print(f"❌ No {dataset_name} data available")
            return {}
        
        # Sample data for faster evaluation if needed
        if len(data) > max_samples:
            data_sample = data.sample(n=max_samples, random_state=42)
            print(f"   Sampling {max_samples} examples for evaluation")
        else:
            data_sample = data
        
        predictions = []
        targets = []
        
        batch_size = 1024
        total_batches = len(data_sample) // batch_size + (1 if len(data_sample) % batch_size != 0 else 0)
        
        with torch.no_grad():
            for i in tqdm(range(0, len(data_sample), batch_size), desc=f"Evaluating {dataset_name}"):
                batch = data_sample.iloc[i:i+batch_size]
                
                # Map IDs to indices
                user_indices = []
                movie_indices = []
                valid_ratings = []
                
                for _, row in batch.iterrows():
                    user_id = row['userId'] if 'userId' in row else row['user_id']
                    movie_id = row['movieId'] if 'movieId' in row else row['movie_id']
                    rating = row['rating']
                    
                    if (user_id in self.mappings['user_to_index'] and 
                        movie_id in self.mappings['movie_to_index']):
                        user_indices.append(self.mappings['user_to_index'][user_id])
                        movie_indices.append(self.mappings['movie_to_index'][movie_id])
                        valid_ratings.append(rating)
                
                if len(user_indices) == 0:
                    continue
                
                # Predict
                user_tensor = torch.tensor(user_indices, dtype=torch.long, device=self.device)
                movie_tensor = torch.tensor(movie_indices, dtype=torch.long, device=self.device)
                
                pred, _, _ = self.model(user_tensor, movie_tensor, (0.5, 5.0))
                
                predictions.extend(pred.cpu().numpy().flatten())
                targets.extend(valid_ratings)
        
        if len(predictions) == 0:
            print(f"❌ No valid predictions for {dataset_name}")
            return {}
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Calculate metrics
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))
        
        # R² score
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'n_samples': len(predictions),
            'prediction_mean': float(np.mean(predictions)),
            'prediction_std': float(np.std(predictions)),
            'target_mean': float(np.mean(targets)),
            'target_std': float(np.std(targets))
        }
        
        # Print results
        print(f"📈 {dataset_name.title()} Results:")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE:  {mae:.4f}")
        print(f"   MSE:  {mse:.4f}")
        print(f"   R²:   {r2:.4f}")
        print(f"   Samples: {len(predictions)}")
        print(f"   Prediction range: [{np.min(predictions):.3f}, {np.max(predictions):.3f}]")
        
        return metrics
    
    def generate_sample_recommendations(self, n_users: int = 5) -> Dict:
        """Generate sample recommendations for analysis"""
        print(f"\n🎯 Generating sample recommendations for {n_users} users...")
        
        recommendations = {}
        
        # Get random user IDs
        user_ids = list(self.mappings['user_to_index'].keys())[:n_users]
        
        for user_id in user_ids:
            user_idx = self.mappings['user_to_index'][user_id]
            
            # Get random movie candidates
            movie_candidates = list(self.mappings['movie_to_index'].keys())[:100]  # Top 100 for speed
            movie_indices = [self.mappings['movie_to_index'][mid] for mid in movie_candidates]
            
            with torch.no_grad():
                user_tensor = torch.tensor([user_idx] * len(movie_indices), dtype=torch.long, device=self.device)
                movie_tensor = torch.tensor(movie_indices, dtype=torch.long, device=self.device)
                
                predictions, _, _ = self.model(user_tensor, movie_tensor, (0.5, 5.0))
                scores = predictions.cpu().numpy().flatten()
            
            # Get top 10 recommendations
            top_indices = np.argsort(scores)[-10:][::-1]
            top_movies = [movie_candidates[i] for i in top_indices]
            top_scores = [scores[i] for i in top_indices]
            
            recommendations[str(user_id)] = {
                'movie_ids': [int(x) for x in top_movies],
                'predicted_ratings': [float(x) for x in top_scores]
            }
            
            print(f"   User {user_id}: Top movie = {top_movies[0]} (score: {top_scores[0]:.3f})")
        
        return recommendations
    
    def run_full_evaluation(self) -> Dict:
        """Run comprehensive evaluation"""
        print("🚀 Starting comprehensive model evaluation...")
        start_time = time.time()
        
        results = {
            'model_info': {
                'model_path': str(self.model_path),
                'device': str(self.device),
                'evaluation_time': None
            }
        }
        
        # Evaluate on validation data
        if self.val_data is not None:
            results['validation'] = self.evaluate_on_dataset(self.val_data, "validation")
        
        # Evaluate on test data  
        if self.test_data is not None:
            results['test'] = self.evaluate_on_dataset(self.test_data, "test")
        
        # Generate sample recommendations
        results['sample_recommendations'] = self.generate_sample_recommendations()
        
        evaluation_time = time.time() - start_time
        results['model_info']['evaluation_time'] = evaluation_time
        
        print(f"\n✅ Evaluation completed in {evaluation_time:.2f} seconds")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate MovieLens VAE model')
    parser.add_argument('--model_path', default='models/hybrid_vae_best.pt', 
                       help='Path to trained model')
    parser.add_argument('--data_dir', default='data/processed',
                       help='Directory containing processed data')
    parser.add_argument('--output_file', default='evaluation_results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluator = ModelEvaluator(args.model_path, args.data_dir)
    evaluator.load_model_and_data()
    results = evaluator.run_full_evaluation()
    
    # Save results
    import json
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📄 Results saved to {args.output_file}")
    print("\n🎉 Evaluation complete!")

if __name__ == "__main__":
    main()