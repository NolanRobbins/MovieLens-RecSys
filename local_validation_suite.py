"""
Local Validation Suite for MovieLens RecSys
Tests data quality, model architecture, and training stability before Runpod deployment
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
# Optional plotting imports - not required for validation
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
from torch.utils.data import DataLoader, TensorDataset

# Import our data analysis
from data_analysis_improvements import run_comprehensive_analysis

class ValidationResults:
    """Store validation results"""
    def __init__(self):
        self.passed = []
        self.failed = []
        self.warnings = []
        
    def add_pass(self, test_name, message=""):
        self.passed.append(f"✅ {test_name}: {message}")
        
    def add_fail(self, test_name, message=""):
        self.failed.append(f"❌ {test_name}: {message}")
        
    def add_warning(self, test_name, message=""):
        self.warnings.append(f"⚠️  {test_name}: {message}")
        
    def print_summary(self):
        print("\n" + "="*60)
        print("🔍 VALIDATION RESULTS SUMMARY")
        print("="*60)
        
        if self.passed:
            print(f"\n✅ PASSED TESTS ({len(self.passed)}):")
            for test in self.passed:
                print(f"  {test}")
                
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for test in self.warnings:
                print(f"  {test}")
                
        if self.failed:
            print(f"\n❌ FAILED TESTS ({len(self.failed)}):")
            for test in self.failed:
                print(f"  {test}")
        else:
            print("\n🎉 All critical tests passed!")
            
        print(f"\n📊 Overall Score: {len(self.passed)}/{len(self.passed) + len(self.failed)} critical tests passed")
        
        return len(self.failed) == 0


def validate_data_files(data_path):
    """Validate data file existence and basic properties"""
    results = ValidationResults()
    
    print("🔍 Step 1: Data File Validation")
    print("-" * 40)
    
    required_files = ['train_data.csv', 'val_data.csv', 'data_mappings.pkl']
    data_path = Path(data_path)
    
    for filename in required_files:
        filepath = data_path / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            results.add_pass(f"File {filename}", f"{size_mb:.1f} MB")
        else:
            results.add_fail(f"File {filename}", "Missing required file")
    
    return results


def validate_data_quality(data_path):
    """Deep data quality validation"""
    results = ValidationResults()
    
    print("\n🔍 Step 2: Data Quality Analysis")
    print("-" * 40)
    
    try:
        # Load data
        train_df = pd.read_csv(data_path / 'train_data.csv')
        val_df = pd.read_csv(data_path / 'val_data.csv')
        
        with open(data_path / 'data_mappings.pkl', 'rb') as f:
            mappings = pickle.load(f)
            
        # Basic shape validation
        if len(train_df) > 1000000:  # Expect millions of samples
            results.add_pass("Training data size", f"{len(train_df):,} samples")
        else:
            results.add_warning("Training data size", f"Only {len(train_df):,} samples - seems small")
            
        # Check for required columns
        required_cols = ['user_id', 'movie_id', 'rating']
        missing_cols = [col for col in required_cols if col not in train_df.columns]
        
        if not missing_cols:
            results.add_pass("Required columns", "All present")
        else:
            results.add_fail("Required columns", f"Missing: {missing_cols}")
            
        # Rating range validation
        train_ratings = train_df['rating']
        val_ratings = val_df['rating']
        
        if train_ratings.min() >= 0.5 and train_ratings.max() <= 5.0:
            results.add_pass("Rating range (train)", f"{train_ratings.min()} to {train_ratings.max()}")
        else:
            results.add_fail("Rating range (train)", f"Invalid range: {train_ratings.min()} to {train_ratings.max()}")
            
        # Check for NaN values
        if train_df.isnull().sum().sum() == 0:
            results.add_pass("NaN values (train)", "None found")
        else:
            results.add_fail("NaN values (train)", f"{train_df.isnull().sum().sum()} found")
            
        # Data leakage check
        train_pairs = set(zip(train_df['user_id'], train_df['movie_id']))
        val_pairs = set(zip(val_df['user_id'], val_df['movie_id']))
        overlap = len(train_pairs & val_pairs)
        
        if overlap == 0:
            results.add_pass("Data leakage", "No overlap found")
        else:
            results.add_fail("Data leakage", f"{overlap} overlapping pairs")
            
        # User/Movie ID consistency
        train_users = set(train_df['user_id'].unique())
        val_users = set(val_df['user_id'].unique())
        
        if len(val_users - train_users) == 0:
            results.add_pass("User ID consistency", "All validation users in training")
        else:
            cold_users = len(val_users - train_users)
            results.add_warning("User ID consistency", f"{cold_users} cold start users in validation")
            
        # Rating distribution analysis
        train_mean = train_ratings.mean()
        train_std = train_ratings.std()
        
        if 3.0 <= train_mean <= 4.5:
            results.add_pass("Rating distribution", f"Mean: {train_mean:.2f}, Std: {train_std:.2f}")
        else:
            results.add_warning("Rating distribution", f"Unusual mean: {train_mean:.2f}")
            
        # Check mapping consistency
        expected_users = len(mappings['user_to_index'])
        expected_movies = len(mappings['movie_to_index'])
        
        actual_users = len(set(train_df['user_id']) | set(val_df['user_id']))
        actual_movies = len(set(train_df['movie_id']) | set(val_df['movie_id']))
        
        if expected_users == actual_users:
            results.add_pass("User mapping consistency", f"{expected_users} users")
        else:
            results.add_fail("User mapping consistency", f"Expected {expected_users}, got {actual_users}")
            
    except Exception as e:
        results.add_fail("Data loading", f"Error: {str(e)}")
    
    return results, train_df, val_df, mappings


def create_test_model(n_users, n_movies):
    """Create a minimal test model based on experiment_2 success"""
    
    class TestHybridVAE(nn.Module):
        def __init__(self, n_users, n_movies, n_factors=64, hidden_dims=[128, 64], latent_dim=32):
            super(TestHybridVAE, self).__init__()
            
            # Embeddings
            self.user_embedding = nn.Embedding(n_users, n_factors)
            self.movie_embedding = nn.Embedding(n_movies, n_factors)
            
            # Encoder
            encoder_layers = []
            input_dim = n_factors * 2
            
            for hidden_dim in hidden_dims:
                encoder_layers.extend([
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                ])
                input_dim = hidden_dim
                
            self.encoder = nn.Sequential(*encoder_layers)
            
            # VAE layers
            self.mu_layer = nn.Linear(hidden_dims[-1], latent_dim)
            self.logvar_layer = nn.Linear(hidden_dims[-1], latent_dim)
            
            # Decoder
            decoder_layers = []
            input_dim = latent_dim
            
            for hidden_dim in reversed(hidden_dims):
                decoder_layers.extend([
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                ])
                input_dim = hidden_dim
                
            decoder_layers.append(nn.Linear(input_dim, 1))
            self.decoder = nn.Sequential(*decoder_layers)
            
            self._init_weights()
            
        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight, gain=0.1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Embedding):
                    nn.init.normal_(m.weight, 0, 0.02)
                    
        def encode(self, users, movies):
            user_emb = self.user_embedding(users)
            movie_emb = self.movie_embedding(movies)
            features = torch.cat([user_emb, movie_emb], dim=1)
            encoded = self.encoder(features)
            mu = self.mu_layer(encoded)
            logvar = self.logvar_layer(encoded)
            logvar = torch.clamp(logvar, -10, 10)  # Stability
            return mu, logvar
            
        def reparameterize(self, mu, logvar):
            if self.training:
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return mu + eps * std
            else:
                return mu
                
        def forward(self, users, movies):
            mu, logvar = self.encode(users, movies)
            z = self.reparameterize(mu, logvar)
            rating_pred = self.decoder(z)
            rating_pred = torch.sigmoid(rating_pred) * 4.5 + 0.5  # Scale to [0.5, 5.0]
            return rating_pred, mu, logvar
    
    return TestHybridVAE(n_users, n_movies)


def validate_model_architecture(n_users, n_movies):
    """Test model architecture for stability"""
    results = ValidationResults()
    
    print("\n🔍 Step 3: Model Architecture Validation")
    print("-" * 40)
    
    try:
        # Create test model
        model = create_test_model(n_users, n_movies)
        model.eval()
        
        # Count parameters
        param_count = sum(p.numel() for p in model.parameters())
        results.add_pass("Model creation", f"{param_count:,} parameters")
        
        # Test forward pass with sample data
        batch_size = 100
        test_users = torch.randint(0, min(n_users, 1000), (batch_size,))
        test_movies = torch.randint(0, min(n_movies, 1000), (batch_size,))
        
        with torch.no_grad():
            predictions, mu, logvar = model(test_users, test_movies)
            
        # Check for NaN outputs
        if torch.isnan(predictions).any():
            results.add_fail("Forward pass", "NaN values in predictions")
        else:
            results.add_pass("Forward pass", "No NaN values")
            
        # Check output range
        pred_min, pred_max = predictions.min().item(), predictions.max().item()
        if 0.4 <= pred_min and pred_max <= 5.1:
            results.add_pass("Output range", f"{pred_min:.2f} to {pred_max:.2f}")
        else:
            results.add_warning("Output range", f"Unusual range: {pred_min:.2f} to {pred_max:.2f}")
            
        # Check latent statistics
        mu_mean = mu.mean().item()
        logvar_mean = logvar.mean().item()
        
        if -2 <= mu_mean <= 2 and -5 <= logvar_mean <= 5:
            results.add_pass("Latent statistics", f"μ: {mu_mean:.2f}, logvar: {logvar_mean:.2f}")
        else:
            results.add_warning("Latent statistics", f"Unusual values: μ: {mu_mean:.2f}, logvar: {logvar_mean:.2f}")
            
    except Exception as e:
        results.add_fail("Model architecture", f"Error: {str(e)}")
        
    return results, model


def validate_training_step(model, train_df, mappings):
    """Test a few training steps for stability"""
    results = ValidationResults()
    
    print("\n🔍 Step 4: Training Step Validation")  
    print("-" * 40)
    
    try:
        # Create small sample for testing
        sample_size = 1000
        sample_df = train_df.sample(n=sample_size).copy()
        
        # Map to indices
        sample_df['user_idx'] = sample_df['user_id'].map(mappings['user_to_index'])
        sample_df['movie_idx'] = sample_df['movie_id'].map(mappings['movie_to_index'])
        
        # Check for mapping failures
        if sample_df['user_idx'].isnull().any() or sample_df['movie_idx'].isnull().any():
            results.add_fail("Index mapping", "Some IDs failed to map")
            return results
            
        # Create tensors
        X_sample = torch.LongTensor(sample_df[['user_idx', 'movie_idx']].values)
        y_sample = torch.FloatTensor(sample_df['rating'].values)
        
        # Test loss function
        def vae_loss(predictions, targets, mu, logvar, kl_weight=0.01):
            recon_loss = F.mse_loss(predictions.squeeze(), targets)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss = torch.clamp(kl_loss, 0, 100)
            total_loss = recon_loss + kl_weight * kl_loss
            return total_loss, recon_loss, kl_loss
        
        # Setup optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        model.train()
        
        # Test multiple training steps
        losses = []
        for step in range(5):
            optimizer.zero_grad()
            
            predictions, mu, logvar = model(X_sample[:, 0], X_sample[:, 1])
            total_loss, recon_loss, kl_loss = vae_loss(predictions, y_sample, mu, logvar)
            
            # Check for NaN in loss
            if torch.isnan(total_loss):
                results.add_fail(f"Training step {step+1}", "NaN loss detected")
                break
                
            total_loss.backward()
            
            # Check gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if torch.isnan(grad_norm):
                results.add_fail(f"Training step {step+1}", "NaN gradients detected")
                break
                
            optimizer.step()
            losses.append(total_loss.item())
            
        if len(losses) == 5:
            results.add_pass("Training stability", f"5 steps completed, final loss: {losses[-1]:.4f}")
            
            # Check if loss is improving or at least stable
            if losses[-1] < losses[0] * 1.1:  # Allow 10% tolerance
                results.add_pass("Loss trend", "Stable or improving")
            else:
                results.add_warning("Loss trend", "Loss may be unstable")
        
    except Exception as e:
        results.add_fail("Training step", f"Error: {str(e)}")
        
    return results


def run_comprehensive_data_analysis(data_path):
    """Run our data analysis tool and validate results"""
    results = ValidationResults()
    
    print("\n🔍 Step 5: Comprehensive Data Analysis")
    print("-" * 40)
    
    try:
        # Load data for analysis
        train_df = pd.read_csv(data_path / 'train_data.csv')
        val_df = pd.read_csv(data_path / 'val_data.csv')
        
        # Run the analysis
        analysis_results = run_comprehensive_analysis(train_df, val_df)
        
        # Check for critical issues from analysis
        recommendations = analysis_results.get('recommendations', [])
        
        critical_issues = [rec for rec in recommendations if 'CRITICAL' in rec]
        if critical_issues:
            for issue in critical_issues:
                results.add_fail("Data analysis", issue)
        else:
            results.add_pass("Data analysis", "No critical issues found")
            
        # Check sparsity
        sparsity = analysis_results.get('sparsity', {}).get('sparsity_percentage', 0)
        if sparsity > 99.95:
            results.add_warning("Data sparsity", f"{sparsity:.4f}% - very sparse")
        else:
            results.add_pass("Data sparsity", f"{sparsity:.4f}%")
            
    except Exception as e:
        results.add_fail("Data analysis", f"Error: {str(e)}")
        
    return results


def main():
    """Main validation function"""
    print("🚀 MovieLens RecSys Local Validation Suite")
    print("=" * 60)
    print("Testing data quality, model architecture, and training stability")
    print("before expensive A100 deployment on Runpod")
    print("")
    
    # Configuration
    data_path = Path('data/processed')
    
    if not data_path.exists():
        print(f"❌ Data path not found: {data_path}")
        print("Please ensure your processed data is in the correct location")
        return False
        
    all_results = []
    
    # Step 1: Validate data files
    file_results = validate_data_files(data_path)
    all_results.append(file_results)
    
    # Step 2: Validate data quality
    quality_results, train_df, val_df, mappings = validate_data_quality(data_path)
    all_results.append(quality_results)
    
    if len(quality_results.failed) > 0:
        print("\n⚠️  Critical data issues found. Stopping validation.")
        quality_results.print_summary()
        return False
    
    # Step 3: Validate model architecture
    n_users = len(mappings['user_to_index'])
    n_movies = len(mappings['movie_to_index'])
    
    arch_results, model = validate_model_architecture(n_users, n_movies)
    all_results.append(arch_results)
    
    # Step 4: Validate training steps
    train_results = validate_training_step(model, train_df, mappings)
    all_results.append(train_results)
    
    # Step 5: Run comprehensive data analysis
    analysis_results = run_comprehensive_data_analysis(data_path)
    all_results.append(analysis_results)
    
    # Print final summary
    print("\n" + "="*60)
    print("🏁 FINAL VALIDATION SUMMARY")
    print("="*60)
    
    total_passed = sum(len(r.passed) for r in all_results)
    total_failed = sum(len(r.failed) for r in all_results)
    total_warnings = sum(len(r.warnings) for r in all_results)
    
    print(f"✅ Passed: {total_passed}")
    print(f"⚠️  Warnings: {total_warnings}")  
    print(f"❌ Failed: {total_failed}")
    
    if total_failed == 0:
        print("\n🎉 ALL VALIDATIONS PASSED!")
        print("✅ Ready for A100 training on Runpod")
        return True
    else:
        print(f"\n⚠️  {total_failed} critical issues found")
        print("❌ Fix these issues before deploying to Runpod")
        
        # Print all failures
        for i, results in enumerate(all_results):
            if results.failed:
                results.print_summary()
        
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)