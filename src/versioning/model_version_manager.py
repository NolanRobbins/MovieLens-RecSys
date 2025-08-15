"""
Model Version Manager
Comprehensive system for tracking, versioning, and managing ML model experiments
"""

import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import torch
import pandas as pd
from enum import Enum

class ExperimentStatus(Enum):
    """Experiment status tracking"""
    PLANNED = "planned"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"

class ModelType(Enum):
    """Model architecture types"""
    HYBRID_VAE = "hybrid_vae"
    MATRIX_FACTORIZATION = "matrix_factorization"
    NEURAL_CF = "neural_cf"
    TWO_TOWER = "two_tower"
    ENSEMBLE = "ensemble"

@dataclass
class ExperimentMetadata:
    """Complete experiment metadata"""
    # Experiment identifiers
    experiment_id: str
    experiment_name: str
    version: str
    timestamp: str
    
    # Model information
    model_type: ModelType
    model_architecture: Dict[str, Any]
    hyperparameters: Dict[str, Any]
    
    # Performance metrics
    target_rmse: float
    achieved_rmse: Optional[float]
    other_metrics: Dict[str, float]
    
    # Training information
    epochs_trained: int
    training_time_hours: float
    gpu_type: str
    batch_size: int
    
    # Data information
    dataset_version: str
    train_samples: int
    val_samples: int
    test_samples: Optional[int]
    
    # Experiment status
    status: ExperimentStatus
    notes: str
    tags: List[str]
    
    # File paths
    model_path: Optional[str] = None
    config_path: Optional[str] = None
    wandb_run_id: Optional[str] = None
    
    # Comparison metrics
    improvement_over_baseline: Optional[float] = None
    rank_among_experiments: Optional[int] = None

class ModelVersionManager:
    """Manages model versions and experiment tracking"""
    
    def __init__(self, base_dir: Union[str, Path] = "models"):
        self.base_dir = Path(base_dir)
        self.versions_dir = self.base_dir / "versions"
        self.current_dir = self.base_dir / "current"
        self.archive_dir = self.base_dir / "archive"
        
        # Create directories
        for dir_path in [self.versions_dir, self.current_dir, self.archive_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Metadata storage
        self.registry_file = self.base_dir / "experiment_registry.json"
        self.registry = self._load_registry()
        
        # Version tracking
        self.version_counter = self._get_next_version_number()
    
    def _load_registry(self) -> Dict[str, Dict]:
        """Load experiment registry"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_registry(self):
        """Save experiment registry"""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2, default=str)
    
    def _get_next_version_number(self) -> int:
        """Get next version number"""
        if not self.registry:
            return 1
        
        versions = [int(exp['version'].replace('v', '')) 
                   for exp in self.registry.values() 
                   if exp['version'].startswith('v')]
        
        return max(versions) + 1 if versions else 1
    
    def create_experiment(self, 
                         experiment_name: str,
                         model_type: ModelType,
                         model_architecture: Dict[str, Any],
                         hyperparameters: Dict[str, Any],
                         target_rmse: float,
                         dataset_version: str = "v1.0",
                         notes: str = "",
                         tags: List[str] = None) -> str:
        """Create a new experiment with versioning"""
        
        # Generate experiment ID
        timestamp = datetime.now()
        experiment_id = f"{model_type.value}_v{self.version_counter}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        version = f"v{self.version_counter}"
        
        # Create experiment metadata
        metadata = ExperimentMetadata(
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            version=version,
            timestamp=timestamp.isoformat(),
            model_type=model_type,
            model_architecture=model_architecture,
            hyperparameters=hyperparameters,
            target_rmse=target_rmse,
            achieved_rmse=None,
            other_metrics={},
            epochs_trained=0,
            training_time_hours=0.0,
            gpu_type="unknown",
            batch_size=hyperparameters.get('batch_size', 1024),
            dataset_version=dataset_version,
            train_samples=0,
            val_samples=0,
            test_samples=None,
            status=ExperimentStatus.PLANNED,
            notes=notes,
            tags=tags or []
        )
        
        # Create experiment directory
        exp_dir = self.versions_dir / experiment_id
        exp_dir.mkdir(exist_ok=True)
        
        # Save metadata
        metadata_path = exp_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2, default=str)
        
        # Add to registry
        self.registry[experiment_id] = asdict(metadata)
        self._save_registry()
        
        # Increment version counter
        self.version_counter += 1
        
        print(f"✅ Created experiment: {experiment_id}")
        print(f"📁 Directory: {exp_dir}")
        
        return experiment_id
    
    def start_experiment(self, experiment_id: str):
        """Mark experiment as running"""
        if experiment_id in self.registry:
            self.registry[experiment_id]['status'] = ExperimentStatus.RUNNING.value
            self.registry[experiment_id]['start_time'] = datetime.now().isoformat()
            self._save_registry()
            print(f"🚀 Started experiment: {experiment_id}")
    
    def complete_experiment(self, 
                           experiment_id: str,
                           model_path: str,
                           achieved_rmse: float,
                           other_metrics: Dict[str, float],
                           epochs_trained: int,
                           training_time_hours: float,
                           gpu_type: str = "A100",
                           config_path: str = None,
                           wandb_run_id: str = None):
        """Complete experiment and save model"""
        
        if experiment_id not in self.registry:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # Update registry
        self.registry[experiment_id].update({
            'status': ExperimentStatus.COMPLETED.value,
            'achieved_rmse': achieved_rmse,
            'other_metrics': other_metrics,
            'epochs_trained': epochs_trained,
            'training_time_hours': training_time_hours,
            'gpu_type': gpu_type,
            'model_path': model_path,
            'config_path': config_path,
            'wandb_run_id': wandb_run_id,
            'completed_at': datetime.now().isoformat()
        })
        
        # Calculate improvement over baseline
        baseline_rmse = self._get_baseline_rmse()
        if baseline_rmse:
            improvement = ((baseline_rmse - achieved_rmse) / baseline_rmse) * 100
            self.registry[experiment_id]['improvement_over_baseline'] = improvement
        
        # Update ranking
        self._update_experiment_rankings()
        
        # Copy model to versioned directory
        exp_dir = self.versions_dir / experiment_id
        if Path(model_path).exists():
            versioned_model_path = exp_dir / f"model_{experiment_id}.pt"
            shutil.copy2(model_path, versioned_model_path)
            self.registry[experiment_id]['versioned_model_path'] = str(versioned_model_path)
        
        # Copy config if provided
        if config_path and Path(config_path).exists():
            versioned_config_path = exp_dir / f"config_{experiment_id}.json"
            shutil.copy2(config_path, versioned_config_path)
            self.registry[experiment_id]['versioned_config_path'] = str(versioned_config_path)
        
        self._save_registry()
        
        print(f"✅ Completed experiment: {experiment_id}")
        print(f"🎯 Achieved RMSE: {achieved_rmse}")
        print(f"📈 Improvement over baseline: {self.registry[experiment_id].get('improvement_over_baseline', 'N/A'):.2f}%")
        print(f"🏆 Rank: #{self.registry[experiment_id].get('rank_among_experiments', 'N/A')}")
    
    def fail_experiment(self, experiment_id: str, error_message: str):
        """Mark experiment as failed"""
        if experiment_id in self.registry:
            self.registry[experiment_id]['status'] = ExperimentStatus.FAILED.value
            self.registry[experiment_id]['error_message'] = error_message
            self.registry[experiment_id]['failed_at'] = datetime.now().isoformat()
            self._save_registry()
            print(f"❌ Failed experiment: {experiment_id}")
    
    def _get_baseline_rmse(self) -> Optional[float]:
        """Get baseline RMSE for comparison"""
        completed_experiments = [
            exp for exp in self.registry.values() 
            if exp['status'] == ExperimentStatus.COMPLETED.value and exp.get('achieved_rmse')
        ]
        
        if completed_experiments:
            return max(exp['achieved_rmse'] for exp in completed_experiments)
        return None
    
    def _update_experiment_rankings(self):
        """Update experiment rankings based on RMSE"""
        completed_experiments = [
            (exp_id, exp) for exp_id, exp in self.registry.items()
            if exp['status'] == ExperimentStatus.COMPLETED.value and exp.get('achieved_rmse')
        ]
        
        # Sort by RMSE (lower is better)
        completed_experiments.sort(key=lambda x: x[1]['achieved_rmse'])
        
        # Update rankings
        for rank, (exp_id, exp) in enumerate(completed_experiments, 1):
            self.registry[exp_id]['rank_among_experiments'] = rank
    
    def set_current_best(self, experiment_id: str):
        """Set experiment as current best model"""
        if experiment_id not in self.registry:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        exp = self.registry[experiment_id]
        if exp['status'] != ExperimentStatus.COMPLETED.value:
            raise ValueError(f"Experiment {experiment_id} is not completed")
        
        # Clear current directory
        for file in self.current_dir.glob("*"):
            if file.is_file():
                file.unlink()
        
        # Copy model to current
        versioned_model_path = exp.get('versioned_model_path')
        if versioned_model_path and Path(versioned_model_path).exists():
            current_model_path = self.current_dir / "current_best_model.pt"
            shutil.copy2(versioned_model_path, current_model_path)
        
        # Copy config to current
        versioned_config_path = exp.get('versioned_config_path')
        if versioned_config_path and Path(versioned_config_path).exists():
            current_config_path = self.current_dir / "current_best_config.json"
            shutil.copy2(versioned_config_path, current_config_path)
        
        # Create current info file
        current_info = {
            'current_best_experiment': experiment_id,
            'rmse': exp['achieved_rmse'],
            'set_at': datetime.now().isoformat()
        }
        
        with open(self.current_dir / "current_info.json", 'w') as f:
            json.dump(current_info, f, indent=2)
        
        print(f"🏆 Set {experiment_id} as current best model (RMSE: {exp['achieved_rmse']})")
    
    def get_experiment_summary(self) -> pd.DataFrame:
        """Get summary of all experiments"""
        data = []
        
        for exp_id, exp in self.registry.items():
            data.append({
                'Experiment ID': exp_id,
                'Name': exp['experiment_name'],
                'Version': exp['version'],
                'Type': exp['model_type'],
                'Status': exp['status'],
                'Target RMSE': exp['target_rmse'],
                'Achieved RMSE': exp.get('achieved_rmse', 'N/A'),
                'Improvement %': exp.get('improvement_over_baseline', 'N/A'),
                'Rank': exp.get('rank_among_experiments', 'N/A'),
                'Training Time (h)': exp.get('training_time_hours', 'N/A'),
                'GPU': exp.get('gpu_type', 'N/A'),
                'Created': exp['timestamp'][:10]  # Just date
            })
        
        df = pd.DataFrame(data)
        if not df.empty:
            # Sort by rank (best first), then by RMSE
            df = df.sort_values(['Rank', 'Achieved RMSE'], na_position='last')
        
        return df
    
    def archive_experiment(self, experiment_id: str):
        """Archive an experiment"""
        if experiment_id not in self.registry:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # Move to archive
        exp_dir = self.versions_dir / experiment_id
        archive_path = self.archive_dir / experiment_id
        
        if exp_dir.exists():
            shutil.move(str(exp_dir), str(archive_path))
        
        # Update status
        self.registry[experiment_id]['status'] = ExperimentStatus.ARCHIVED.value
        self.registry[experiment_id]['archived_at'] = datetime.now().isoformat()
        self._save_registry()
        
        print(f"📦 Archived experiment: {experiment_id}")

def register_current_model():
    """Register the current model (RMSE 0.806) in version system"""
    manager = ModelVersionManager()
    
    # Create experiment for current model
    experiment_id = manager.create_experiment(
        experiment_name="Baseline Hybrid VAE - First Successful Training",
        model_type=ModelType.HYBRID_VAE,
        model_architecture={
            'n_factors': 150,
            'hidden_dims': [512, 256],
            'latent_dim': 64,
            'dropout_rate': 0.3
        },
        hyperparameters={
            'batch_size': 1024,
            'lr': 5e-4,
            'weight_decay': 1e-5,
            'kl_weight': 0.1,
            'epochs': 100
        },
        target_rmse=0.55,
        notes="First successful training run after fixing major issues. Achieved 0.806 RMSE but shows overfitting.",
        tags=["baseline", "v1", "overfitting", "needs_regularization"]
    )
    
    # Mark as completed (we'll update with actual model when downloaded)
    manager.complete_experiment(
        experiment_id=experiment_id,
        model_path="models/downloaded/current_model_rmse_0.806.pt",  # Will be created when downloaded
        achieved_rmse=0.806,
        other_metrics={
            'train_loss': 0.587,
            'val_loss': 0.685,
            'early_stopping_epoch': 22
        },
        epochs_trained=22,
        training_time_hours=0.78,
        gpu_type="A100-80GB",
        wandb_run_id="2uw7m29i"
    )
    
    return experiment_id

def create_next_experiment():
    """Create next experiment with advanced regularization"""
    manager = ModelVersionManager()
    
    experiment_id = manager.create_experiment(
        experiment_name="Advanced Regularization - Targeting RMSE 0.55",
        model_type=ModelType.HYBRID_VAE,
        model_architecture={
            'n_factors': 200,
            'hidden_dims': [1024, 512, 256, 128],
            'latent_dim': 128,
            'dropout_rate': 0.4,
            'use_batch_norm': True,
            'use_layer_norm': True
        },
        hyperparameters={
            'batch_size': 2048,
            'base_lr': 3e-4,
            'weight_decay': 1e-4,
            'beta_min': 0.001,
            'beta_max': 0.5,
            'label_smoothing': 0.1,
            'mixup_alpha': 0.2,
            'free_bits': 2.0,
            'epochs': 200
        },
        target_rmse=0.55,
        notes="Advanced regularization experiment with deeper architecture, aggressive dropout, label smoothing, and mixup augmentation.",
        tags=["v2", "advanced_regularization", "deep_architecture", "data_augmentation", "targeting_0.55_rmse"]
    )
    
    return experiment_id

if __name__ == "__main__":
    # Demo the version manager
    print("🔬 Model Version Manager Demo")
    print("=" * 50)
    
    # Register current model
    current_exp_id = register_current_model()
    print(f"Current model registered: {current_exp_id}")
    
    # Create next experiment
    next_exp_id = create_next_experiment()
    print(f"Next experiment created: {next_exp_id}")
    
    # Show summary
    manager = ModelVersionManager()
    summary = manager.get_experiment_summary()
    print("\n📊 Experiment Summary:")
    print(summary.to_string(index=False))