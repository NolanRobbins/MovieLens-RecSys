"""
Multi-Model Experiment Runner
Trains and compares multiple recommendation models in parallel
Supports Hybrid VAE + baseline models with standardized evaluation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import json
import pickle
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Import models
from .baseline_models import ModelFactory, count_parameters
from .cloud_training import HybridVAE  # Your existing VAE
from .ranking_loss_functions import HybridVAERanking

# Import evaluation
try:
    from ..evaluation.advanced_evaluation import AdvancedEvaluator
except ImportError:
    AdvancedEvaluator = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExperimentConfig:
    """Configuration for multi-model experiments"""
    
    def __init__(self):
        # Training configuration
        self.batch_size = 512
        self.epochs = 50
        self.learning_rate = 0.001
        self.weight_decay = 1e-5
        self.early_stopping_patience = 10
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Models to compare
        self.models_to_test = [
            {
                'name': 'matrix_factorization',
                'type': 'matrix_factorization',
                'params': {'n_factors': 128, 'dropout_rate': 0.2}
            },
            {
                'name': 'neural_cf',
                'type': 'neural_cf', 
                'params': {'mf_dim': 64, 'mlp_dims': [128, 64, 32], 'dropout_rate': 0.2}
            },
            {
                'name': 'two_tower',
                'type': 'two_tower',
                'params': {'embedding_dim': 128, 'tower_dims': [256, 128, 64], 'dropout_rate': 0.3}
            },
            {
                'name': 'hybrid_vae',
                'type': 'hybrid_vae',
                'params': {'n_factors': 150, 'hidden_dims': [512, 256], 'latent_dim': 64, 'dropout_rate': 0.3}
            }
        ]
        
        # Evaluation metrics
        self.evaluation_metrics = [
            'rmse', 'mae', 'precision_at_10', 'recall_at_10', 
            'ndcg_at_10', 'map_at_10', 'diversity', 'coverage'
        ]
        
        # Data paths
        self.data_dir = Path("data/processed")
        self.output_dir = Path("data/experiments")
        self.output_dir.mkdir(parents=True, exist_ok=True)

class ModelTrainer:
    """Individual model trainer with standardized interface"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device(config.device)
    
    def load_data(self) -> Tuple[DataLoader, DataLoader, Dict]:
        """Load and prepare training data"""
        logger.info("Loading training data...")
        
        # Load processed data
        train_data = pd.read_csv(self.config.data_dir / "train_data.csv")
        val_data = pd.read_csv(self.config.data_dir / "val_data.csv")
        
        # Load mappings
        with open(self.config.data_dir / "data_mappings.pkl", 'rb') as f:
            mappings = pickle.load(f)
        
        # Convert to tensors
        train_users = torch.tensor(train_data['user_id'].values, dtype=torch.long)
        train_movies = torch.tensor(train_data['movie_id'].values, dtype=torch.long)
        train_ratings = torch.tensor(train_data['rating'].values, dtype=torch.float32)
        
        val_users = torch.tensor(val_data['user_id'].values, dtype=torch.long)
        val_movies = torch.tensor(val_data['movie_id'].values, dtype=torch.long)
        val_ratings = torch.tensor(val_data['rating'].values, dtype=torch.float32)
        
        # Create data loaders
        train_dataset = TensorDataset(train_users, train_movies, train_ratings)
        val_dataset = TensorDataset(val_users, val_movies, val_ratings)
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, 
            shuffle=True, num_workers=2
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size, 
            shuffle=False, num_workers=2
        )
        
        return train_loader, val_loader, mappings
    
    def create_model(self, model_config: Dict, n_users: int, n_movies: int) -> nn.Module:
        """Create model based on configuration"""
        model_type = model_config['type']
        
        if model_type == 'hybrid_vae':
            # Create your existing VAE model
            model = HybridVAE(
                n_users=n_users,
                n_movies=n_movies,
                **model_config['params']
            )
        else:
            # Create baseline model
            model = ModelFactory.create_model(
                model_type=model_type,
                n_users=n_users,
                n_movies=n_movies,
                **model_config['params']
            )
        
        return model.to(self.device)
    
    def train_model(self, model_config: Dict, train_loader: DataLoader, 
                   val_loader: DataLoader, n_users: int, n_movies: int) -> Dict:
        """Train a single model and return results"""
        model_name = model_config['name']
        logger.info(f"🚀 Starting training for {model_name}")
        
        start_time = time.time()
        
        # Create model
        model = self.create_model(model_config, n_users, n_movies)
        param_count = count_parameters(model)
        
        # Setup optimizer
        optimizer = optim.Adam(
            model.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Loss function
        if model_config['type'] == 'hybrid_vae':
            # VAE uses custom loss (MSE + KL divergence)
            criterion = self._vae_loss
        else:
            # Standard MSE for other models
            criterion = nn.MSELoss()
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(self.config.epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for users, movies, ratings in train_loader:
                users, movies, ratings = users.to(self.device), movies.to(self.device), ratings.to(self.device)
                
                optimizer.zero_grad()
                
                if model_config['type'] == 'hybrid_vae':
                    # VAE forward pass
                    predictions, mu, logvar = model(users, movies)
                    loss = self._vae_loss(predictions, ratings, mu, logvar)
                else:
                    # Standard forward pass
                    predictions = model(users, movies)
                    loss = criterion(predictions, ratings)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for users, movies, ratings in val_loader:
                    users, movies, ratings = users.to(self.device), movies.to(self.device), ratings.to(self.device)
                    
                    if model_config['type'] == 'hybrid_vae':
                        predictions, mu, logvar = model(users, movies)
                        loss = self._vae_loss(predictions, ratings, mu, logvar)
                    else:
                        predictions = model(users, movies)
                        loss = criterion(predictions, ratings)
                    
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping for {model_name} at epoch {epoch + 1}")
                break
            
            # Progress logging
            if (epoch + 1) % 10 == 0:
                logger.info(f"{model_name} - Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Load best model state
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        training_time = time.time() - start_time
        
        # Save model
        model_path = self.config.output_dir / f"{model_name}_best.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': model_config,
            'training_stats': {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss,
                'training_time': training_time,
                'epochs_trained': len(train_losses),
                'param_count': param_count
            }
        }, model_path)
        
        logger.info(f"✅ {model_name} training completed in {training_time:.2f}s, best val loss: {best_val_loss:.4f}")
        
        return {
            'model_name': model_name,
            'model': model,
            'model_path': str(model_path),
            'best_val_loss': best_val_loss,
            'training_time': training_time,
            'param_count': param_count,
            'train_losses': train_losses,
            'val_losses': val_losses
        }
    
    def _vae_loss(self, predictions: torch.Tensor, targets: torch.Tensor, 
                  mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        """VAE loss function with KL divergence"""
        # Reconstruction loss
        mse_loss = nn.MSELoss()(predictions, targets)
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / targets.size(0)  # Normalize by batch size
        
        # Total loss
        total_loss = mse_loss + beta * kl_loss
        
        return total_loss

class MultiModelExperiment:
    """Orchestrates multi-model experiments with parallel training"""
    
    def __init__(self, config: Optional[ExperimentConfig] = None):
        self.config = config or ExperimentConfig()
        self.trainer = ModelTrainer(self.config)
        self.results = {}
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.config.output_dir / f"experiment_{timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Multi-model experiment initialized: {self.experiment_dir}")
    
    def run_experiment(self, parallel: bool = True) -> Dict[str, Any]:
        """Run multi-model experiment"""
        logger.info("🔬 Starting multi-model experiment...")
        
        # Load data once for all models
        train_loader, val_loader, mappings = self.trainer.load_data()
        n_users = len(mappings['user_to_idx'])
        n_movies = len(mappings['movie_to_idx'])
        
        experiment_start = time.time()
        
        if parallel and len(self.config.models_to_test) > 1:
            # Parallel training
            logger.info("Running models in parallel...")
            
            with ThreadPoolExecutor(max_workers=min(4, len(self.config.models_to_test))) as executor:
                # Submit training jobs
                future_to_model = {
                    executor.submit(
                        self.trainer.train_model, 
                        model_config, train_loader, val_loader, n_users, n_movies
                    ): model_config['name'] 
                    for model_config in self.config.models_to_test
                }
                
                # Collect results
                for future in as_completed(future_to_model):
                    model_name = future_to_model[future]
                    try:
                        result = future.result()
                        self.results[model_name] = result
                    except Exception as e:
                        logger.error(f"Model {model_name} failed: {e}")
                        self.results[model_name] = {'error': str(e)}
        
        else:
            # Sequential training
            logger.info("Running models sequentially...")
            
            for model_config in self.config.models_to_test:
                try:
                    result = self.trainer.train_model(
                        model_config, train_loader, val_loader, n_users, n_movies
                    )
                    self.results[result['model_name']] = result
                except Exception as e:
                    logger.error(f"Model {model_config['name']} failed: {e}")
                    self.results[model_config['name']] = {'error': str(e)}
        
        experiment_time = time.time() - experiment_start
        
        # Generate experiment summary
        summary = self._generate_experiment_summary(experiment_time)
        
        # Save results
        self._save_experiment_results(summary)
        
        logger.info(f"🎉 Multi-model experiment completed in {experiment_time:.2f}s")
        
        return {
            'summary': summary,
            'detailed_results': self.results,
            'experiment_dir': str(self.experiment_dir)
        }
    
    def _generate_experiment_summary(self, experiment_time: float) -> Dict[str, Any]:
        """Generate experiment summary with model rankings"""
        summary = {
            'experiment_timestamp': datetime.now().isoformat(),
            'experiment_duration': experiment_time,
            'models_tested': len(self.results),
            'successful_models': len([r for r in self.results.values() if 'error' not in r]),
            'model_rankings': [],
            'best_model': None
        }
        
        # Rank models by validation loss
        valid_results = [(name, result) for name, result in self.results.items() if 'error' not in result]
        valid_results.sort(key=lambda x: x[1]['best_val_loss'])
        
        for rank, (model_name, result) in enumerate(valid_results, 1):
            model_info = {
                'rank': rank,
                'model_name': model_name,
                'val_loss': result['best_val_loss'],
                'training_time': result['training_time'],
                'param_count': result['param_count'],
                'efficiency_score': result['best_val_loss'] * (result['training_time'] / 100)  # Loss * normalized time
            }
            summary['model_rankings'].append(model_info)
        
        if valid_results:
            summary['best_model'] = valid_results[0][0]
        
        return summary
    
    def _save_experiment_results(self, summary: Dict[str, Any]):
        """Save experiment results to files"""
        # Save summary
        summary_path = self.experiment_dir / "experiment_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save detailed results (without model objects)
        detailed_results = {}
        for model_name, result in self.results.items():
            if 'error' not in result:
                detailed_results[model_name] = {
                    k: v for k, v in result.items() 
                    if k != 'model'  # Exclude model object
                }
            else:
                detailed_results[model_name] = result
        
        results_path = self.experiment_dir / "detailed_results.json"
        with open(results_path, 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        logger.info(f"Experiment results saved to {self.experiment_dir}")
    
    def get_model_comparison(self) -> pd.DataFrame:
        """Get model comparison as DataFrame"""
        comparison_data = []
        
        for model_name, result in self.results.items():
            if 'error' not in result:
                comparison_data.append({
                    'Model': model_name,
                    'Val Loss (RMSE)': f"{result['best_val_loss']:.4f}",
                    'Training Time (s)': f"{result['training_time']:.1f}",
                    'Parameters': f"{result['param_count']:,}",
                    'Epochs Trained': len(result['train_losses']),
                    'Status': '✅ Success'
                })
            else:
                comparison_data.append({
                    'Model': model_name,
                    'Val Loss (RMSE)': 'N/A',
                    'Training Time (s)': 'N/A', 
                    'Parameters': 'N/A',
                    'Epochs Trained': 'N/A',
                    'Status': f"❌ Error: {result['error']}"
                })
        
        return pd.DataFrame(comparison_data)

def main():
    """Run multi-model experiment"""
    # Create experiment config
    config = ExperimentConfig()
    
    # For quick testing, reduce epochs
    config.epochs = 20
    config.models_to_test = config.models_to_test[:2]  # Test first 2 models
    
    # Run experiment
    experiment = MultiModelExperiment(config)
    results = experiment.run_experiment(parallel=False)  # Sequential for debugging
    
    # Print results
    print("\n📊 Experiment Results:")
    print(experiment.get_model_comparison().to_string(index=False))
    
    print(f"\n🏆 Best model: {results['summary']['best_model']}")
    print(f"📁 Results saved to: {results['experiment_dir']}")

if __name__ == "__main__":
    main()