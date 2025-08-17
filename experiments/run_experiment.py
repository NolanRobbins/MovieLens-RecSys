#!/usr/bin/env python3
"""
Unified Experiment Runner
Executes experiments from YAML configurations
"""

import yaml
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from datetime import datetime
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from experiments.experiment_manager import ExperimentManager
from src.models.enhanced_training_v2 import HybridVAE  # Import your models
from src.data.etl_pipeline import load_data  # Import your data loading


class ExperimentRunner:
    """Runs experiments from YAML configurations"""
    
    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        self.manager = ExperimentManager()
        self.config = self._load_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load experiment configuration"""
        config_path = Path("experiments/configs") / f"{self.experiment_id}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_wandb(self):
        """Initialize Weights & Biases tracking"""
        tracking_config = self.config.get('tracking', {})
        
        if tracking_config.get('wandb_project'):
            wandb.init(
                project=tracking_config['wandb_project'],
                name=self.experiment_id,
                config=self.config,
                tags=self.config.get('metadata', {}).get('tags', [])
            )
            self.logger.info("W&B tracking initialized")
    
    def _create_model(self) -> nn.Module:
        """Create model from configuration"""
        model_config = self.config['model']
        model_type = model_config['type']
        arch_config = model_config['architecture']
        
        if model_type == "hybrid_vae":
            # Get data dimensions (you'll need to implement this)
            n_users, n_items = self._get_data_dimensions()
            
            model = HybridVAE(
                n_users=n_users,
                n_items=n_items,
                n_factors=arch_config['n_factors'],
                hidden_dims=arch_config['hidden_dims'],
                latent_dim=arch_config['latent_dim'],
                dropout_rate=arch_config['dropout_rate']
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model.to(self.device)
    
    def _create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Create optimizer from configuration"""
        train_config = self.config['training']
        optimizer_name = train_config.get('optimizer', 'adam').lower()
        
        params = {
            'lr': train_config['learning_rate'],
            'weight_decay': train_config.get('weight_decay', 0)
        }
        
        if optimizer_name == 'adam':
            params['betas'] = train_config.get('betas', [0.9, 0.999])
            return optim.Adam(model.parameters(), **params)
        elif optimizer_name == 'adamw':
            params['betas'] = train_config.get('betas', [0.9, 0.999])
            params['amsgrad'] = train_config.get('amsgrad', False)
            return optim.AdamW(model.parameters(), **params)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _create_data_loaders(self) -> tuple:
        """Create data loaders from configuration"""
        data_config = self.config['data']
        train_config = self.config['training']
        
        # Load your data (implement this based on your data loading)
        train_data, val_data, test_data = load_data(data_config)
        
        batch_size = train_config['batch_size']
        
        train_loader = DataLoader(
            train_data, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=4
        )
        
        val_loader = DataLoader(
            val_data, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4
        )
        
        test_loader = DataLoader(
            test_data, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4
        ) if test_data else None
        
        return train_loader, val_loader, test_loader
    
    def _get_data_dimensions(self) -> tuple:
        """Get data dimensions for model creation"""
        # Implement this based on your data structure
        # This is a placeholder
        return 1000, 1000  # n_users, n_items
    
    def _train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                    optimizer: optim.Optimizer, epoch: int) -> Dict[str, float]:
        """Train one epoch"""
        model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Implement your training step based on your model
            # This is a placeholder structure
            
            optimizer.zero_grad()
            
            # Forward pass
            # output = model(batch)
            # loss_dict = self._compute_loss(output, batch, epoch)
            
            # Backward pass
            # loss_dict['total_loss'].backward()
            # optimizer.step()
            
            # Accumulate metrics
            # total_loss += loss_dict['total_loss'].item()
            
            # Log batch metrics to W&B
            if wandb.run and batch_idx % self.config.get('tracking', {}).get('log_frequency', 100) == 0:
                wandb.log({
                    'batch/loss': total_loss / (batch_idx + 1),
                    'batch/epoch': epoch,
                    'batch/step': epoch * len(train_loader) + batch_idx
                })
        
        return {
            'train_loss': total_loss / len(train_loader),
            'train_recon_loss': total_recon_loss / len(train_loader),
            'train_kl_loss': total_kl_loss / len(train_loader)
        }
    
    def _validate(self, model: nn.Module, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model"""
        model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Implement validation step
                # output = model(batch)
                # loss = self._compute_loss(output, batch, 0)
                # total_loss += loss['total_loss'].item()
                pass
        
        return {
            'val_loss': total_loss / len(val_loader),
            'val_rmse': 0.0  # Compute actual RMSE
        }
    
    def run(self):
        """Run the complete experiment"""
        self.logger.info(f"Starting experiment: {self.experiment_id}")
        
        try:
            # Mark experiment as running
            self.manager.start_experiment(self.experiment_id)
            
            # Setup tracking
            self._setup_wandb()
            
            # Create model and optimizer
            model = self._create_model()
            optimizer = self._create_optimizer(model)
            
            # Create data loaders
            train_loader, val_loader, test_loader = self._create_data_loaders()
            
            # Training configuration
            train_config = self.config['training']
            max_epochs = train_config['max_epochs']
            patience = train_config.get('early_stopping_patience', 10)
            
            # Training loop
            best_val_rmse = float('inf')
            patience_counter = 0
            
            for epoch in range(max_epochs):
                self.logger.info(f"Epoch {epoch + 1}/{max_epochs}")
                
                # Train
                train_metrics = self._train_epoch(model, train_loader, optimizer, epoch)
                
                # Validate
                val_metrics = self._validate(model, val_loader)
                
                # Log metrics
                if wandb.run:
                    wandb.log({
                        'epoch': epoch,
                        **train_metrics,
                        **val_metrics
                    })
                
                # Early stopping
                if val_metrics['val_rmse'] < best_val_rmse:
                    best_val_rmse = val_metrics['val_rmse']
                    patience_counter = 0
                    
                    # Save best model
                    results_dir = Path("experiments/results") / self.experiment_id
                    model_path = results_dir / "best_model.pt"
                    torch.save(model.state_dict(), model_path)
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    self.logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
            
            # Final evaluation
            final_results = {
                'rmse': best_val_rmse,
                'epochs_trained': epoch + 1,
                'training_time_hours': 0.0,  # Calculate actual time
                'best_epoch': epoch + 1 - patience_counter
            }
            
            # Complete experiment
            self.manager.complete_experiment(
                self.experiment_id,
                final_results,
                str(model_path) if 'model_path' in locals() else None
            )
            
            if wandb.run:
                wandb.finish()
                
            self.logger.info(f"Experiment completed successfully: {self.experiment_id}")
            return final_results
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {str(e)}")
            self.manager.fail_experiment(self.experiment_id, str(e))
            
            if wandb.run:
                wandb.finish(exit_code=1)
            
            raise


def main():
    parser = argparse.ArgumentParser(description="Run experiment from configuration")
    parser.add_argument('experiment_id', help='Experiment ID to run')
    parser.add_argument('--dry-run', action='store_true', help='Show config without running')
    
    args = parser.parse_args()
    
    if args.dry_run:
        runner = ExperimentRunner(args.experiment_id)
        print("Experiment Configuration:")
        print(yaml.dump(runner.config, indent=2))
        return
    
    runner = ExperimentRunner(args.experiment_id)
    results = runner.run()
    
    print(f"Experiment completed with RMSE: {results['rmse']:.4f}")


if __name__ == "__main__":
    main()