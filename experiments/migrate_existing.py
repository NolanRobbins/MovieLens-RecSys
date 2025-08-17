#!/usr/bin/env python3
"""
Migration script to import existing experiments into the new system
"""

import json
import yaml
from pathlib import Path
from datetime import datetime
from experiments.experiment_manager import ExperimentManager


def migrate_experiment_registry():
    """Migrate existing experiment_registry.json to new system"""
    
    # Load existing registry
    old_registry_path = Path("models/experiment_registry.json")
    if not old_registry_path.exists():
        print("❌ No existing experiment registry found")
        return
    
    with open(old_registry_path, 'r') as f:
        old_registry = json.load(f)
    
    manager = ExperimentManager()
    migrated_count = 0
    
    print(f"🔄 Migrating {len(old_registry)} experiments...")
    
    for exp_id, exp_data in old_registry.items():
        try:
            # Convert old experiment to new template format
            config = convert_to_new_format(exp_data)
            
            # Save as new experiment config
            config_path = manager.configs_dir / f"{exp_id}.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f, indent=2)
            
            # Create results directory
            results_dir = manager.results_dir / exp_id
            results_dir.mkdir(exist_ok=True)
            
            # Save results if experiment completed
            if exp_data.get('status') == 'completed':
                results = {
                    'rmse': exp_data.get('achieved_rmse'),
                    'train_loss': exp_data.get('other_metrics', {}).get('train_loss'),
                    'val_loss': exp_data.get('other_metrics', {}).get('val_loss'),
                    'epochs_trained': exp_data.get('epochs_trained', 0),
                    'training_time_hours': exp_data.get('training_time_hours', 0),
                    'early_stopping_epoch': exp_data.get('other_metrics', {}).get('early_stopping_epoch')
                }
                
                # Remove None values
                results = {k: v for k, v in results.items() if v is not None}
                
                with open(results_dir / "results.json", 'w') as f:
                    json.dump(results, f, indent=2)
            
            # Register in new system
            registry_entry = {
                'experiment_id': exp_id,
                'name': exp_data.get('experiment_name', 'Migrated Experiment'),
                'template': 'migrated',
                'config_path': str(config_path),
                'results_path': str(results_dir),
                'status': exp_data.get('status', 'unknown'),
                'created_at': exp_data.get('timestamp', datetime.now().isoformat()),
                'tags': exp_data.get('tags', []),
                'migrated': True,
                'original_system': 'ModelVersionManager'
            }
            
            manager._save_registry_entry(registry_entry)
            
            # Update status
            if exp_data.get('status') == 'completed':
                manager.status[exp_id] = {
                    'status': 'completed',
                    'created_at': exp_data.get('timestamp'),
                    'completed_at': exp_data.get('completed_at'),
                    'results': results if 'results' in locals() else {}
                }
            else:
                manager.status[exp_id] = {
                    'status': exp_data.get('status', 'unknown'),
                    'created_at': exp_data.get('timestamp')
                }
            
            migrated_count += 1
            print(f"✅ Migrated: {exp_id}")
            
        except Exception as e:
            print(f"❌ Failed to migrate {exp_id}: {str(e)}")
    
    # Save status
    manager._save_status()
    
    print(f"🎉 Successfully migrated {migrated_count}/{len(old_registry)} experiments")
    
    # Show summary
    df = manager.get_experiments_dataframe()
    print("\n📊 Migrated Experiments Summary:")
    print(df[df.index < 10].to_string(index=False))  # Show first 10


def convert_to_new_format(exp_data: dict) -> dict:
    """Convert old experiment format to new YAML config format"""
    
    # Extract model architecture
    model_arch = exp_data.get('model_architecture', {})
    hyperparams = exp_data.get('hyperparameters', {})
    
    # Determine model type
    model_type = exp_data.get('model_type', 'ModelType.HYBRID_VAE')
    if 'HYBRID_VAE' in model_type:
        model_type = 'hybrid_vae'
    elif 'NEURAL_CF' in model_type:
        model_type = 'neural_cf'
    else:
        model_type = 'unknown'
    
    config = {
        'metadata': {
            'name': exp_data.get('experiment_name', 'Migrated Experiment'),
            'description': exp_data.get('notes', 'Migrated from previous system'),
            'version': exp_data.get('version', 'unknown'),
            'tags': exp_data.get('tags', []) + ['migrated'],
            'experiment_id': exp_data.get('experiment_id'),
            'created_at': exp_data.get('timestamp'),
            'migrated_from': 'ModelVersionManager'
        },
        
        'model': {
            'type': model_type,
            'architecture': {
                'n_factors': model_arch.get('n_factors', 150),
                'hidden_dims': model_arch.get('hidden_dims', [512, 256]),
                'latent_dim': model_arch.get('latent_dim', 64),
                'dropout_rate': model_arch.get('dropout_rate', 0.3),
                'use_batch_norm': model_arch.get('use_batch_norm', False),
                'use_layer_norm': model_arch.get('use_layer_norm', False)
            }
        },
        
        'training': {
            'optimizer': 'adam',  # Default, wasn't tracked in old system
            'learning_rate': hyperparams.get('lr', hyperparams.get('base_lr', 0.0005)),
            'weight_decay': hyperparams.get('weight_decay', 1e-5),
            'batch_size': hyperparams.get('batch_size', 1024),
            'max_epochs': hyperparams.get('epochs', 100),
            'early_stopping_patience': 15,  # Default
            'kl_weight': hyperparams.get('kl_weight', hyperparams.get('beta_min', 0.1)),
            'beta_schedule': 'constant'  # Default
        },
        
        'data': {
            'dataset_version': exp_data.get('dataset_version', 'v1.0'),
            'train_split': 0.8,
            'val_split': 0.1,
            'test_split': 0.1,
            'min_user_interactions': 20,
            'min_item_interactions': 5,
            'rating_threshold': 3.0
        },
        
        'targets': {
            'rmse': exp_data.get('target_rmse', 0.75),
            'precision_at_10': 0.35,
            'ndcg_at_10': 0.42,
            'diversity': 0.50,
            'coverage': 0.60
        },
        
        'resources': {
            'gpu_type': exp_data.get('gpu_type', 'A100'),
            'memory_gb': 32,
            'max_training_hours': exp_data.get('training_time_hours', 4),
            'mixed_precision': False
        },
        
        'tracking': {
            'wandb_project': 'movielens-migrated',
            'wandb_run_id': exp_data.get('wandb_run_id'),
            'log_frequency': 100,
            'save_checkpoints': True,
            'checkpoint_frequency': 10
        },
        
        'original_data': exp_data  # Keep original for reference
    }
    
    return config


def create_migrated_template():
    """Create a template based on migrated experiments"""
    
    manager = ExperimentManager()
    
    # Find the best migrated experiment
    df = manager.get_experiments_dataframe()
    migrated_df = df[df['tags'].str.contains('migrated', na=False)]
    
    if migrated_df.empty:
        print("❌ No migrated experiments found")
        return
    
    # Get best performing migrated experiment
    completed_df = migrated_df[migrated_df['status'] == 'completed']
    if completed_df.empty:
        print("❌ No completed migrated experiments found")
        return
    
    best_exp_id = completed_df.loc[completed_df['rmse'].idxmin(), 'experiment_id']
    print(f"🏆 Best migrated experiment: {best_exp_id}")
    
    # Load its config
    best_config = manager.export_experiment_config(best_exp_id)
    
    # Create template from best config
    template_config = best_config.copy()
    
    # Remove experiment-specific data
    template_config['metadata'] = {
        'name': 'hybrid_vae_migrated',
        'description': 'Template based on best migrated experiment',
        'version': '1.0',
        'tags': ['hybrid_vae', 'migrated', 'proven'],
        'author': 'Migration System'
    }
    
    # Remove original data
    template_config.pop('original_data', None)
    
    # Save as template
    template_path = manager.templates_dir / "hybrid_vae_migrated.yaml"
    with open(template_path, 'w') as f:
        yaml.dump(template_config, f, indent=2)
    
    print(f"✅ Created template: {template_path}")
    print(f"📊 Based on experiment with RMSE: {completed_df.loc[completed_df['experiment_id'] == best_exp_id, 'rmse'].iloc[0]}")


def main():
    """Main migration function"""
    print("🔄 Starting migration from old experiment system...")
    
    # Migrate existing experiments
    migrate_experiment_registry()
    
    # Create template from best migrated experiment
    create_migrated_template()
    
    print("\n✅ Migration complete!")
    print("\nNext steps:")
    print("1. Review migrated experiments: python experiments/experiment_manager.py list")
    print("2. Create new experiment from migrated template:")
    print("   python experiments/experiment_manager.py create hybrid_vae_migrated my_new_experiment")


if __name__ == "__main__":
    main()