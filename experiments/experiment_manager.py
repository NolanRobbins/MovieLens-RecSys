"""
Streamlined Experiment Manager
Unified system for experiment tracking, configuration, and execution
"""

import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import pandas as pd
from enum import Enum
import shutil
import hashlib


class ExperimentStatus(Enum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"


@dataclass
class ExperimentConfig:
    """Standardized experiment configuration"""
    # Basic info
    name: str
    description: str
    tags: List[str]
    
    # Model configuration
    model_type: str
    architecture: Dict[str, Any]
    
    # Training configuration
    training: Dict[str, Any]
    
    # Data configuration
    data: Dict[str, Any]
    
    # Target metrics
    targets: Dict[str, float]
    
    # Resource requirements
    resources: Dict[str, Any]


class ExperimentManager:
    """Unified experiment management system"""
    
    def __init__(self, base_dir: Union[str, Path] = "."):
        self.base_dir = Path(base_dir)
        self.experiments_dir = self.base_dir / "experiments"
        self.configs_dir = self.experiments_dir / "configs"
        self.results_dir = self.experiments_dir / "results"
        self.templates_dir = self.experiments_dir / "templates"
        
        # Create directory structure
        for dir_path in [self.experiments_dir, self.configs_dir, 
                        self.results_dir, self.templates_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Registry files
        self.registry_file = self.experiments_dir / "registry.jsonl"
        self.status_file = self.experiments_dir / "status.json"
        
        # Load existing data
        self.registry = self._load_registry()
        self.status = self._load_status()
    
    def _load_registry(self) -> List[Dict]:
        """Load experiment registry from JSONL"""
        registry = []
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                for line in f:
                    registry.append(json.loads(line.strip()))
        return registry
    
    def _save_registry_entry(self, entry: Dict):
        """Append entry to registry"""
        with open(self.registry_file, 'a') as f:
            f.write(json.dumps(entry, default=str) + '\n')
    
    def _load_status(self) -> Dict:
        """Load current experiment status"""
        if self.status_file.exists():
            with open(self.status_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_status(self):
        """Save experiment status"""
        with open(self.status_file, 'w') as f:
            json.dump(self.status, f, indent=2, default=str)
    
    def create_experiment_from_template(self, 
                                      template_name: str,
                                      experiment_name: str,
                                      modifications: Dict[str, Any] = None) -> str:
        """Create experiment from template with modifications"""
        
        # Load template
        template_path = self.templates_dir / f"{template_name}.yaml"
        if not template_path.exists():
            raise ValueError(f"Template {template_name} not found")
        
        with open(template_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Apply modifications
        if modifications:
            config = self._deep_update(config, modifications)
        
        # Generate experiment ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"{experiment_name}_{timestamp}"
        config['metadata']['experiment_id'] = experiment_id
        config['metadata']['created_at'] = datetime.now().isoformat()
        
        # Save experiment config
        config_path = self.configs_dir / f"{experiment_id}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, indent=2)
        
        # Create results directory
        exp_results_dir = self.results_dir / experiment_id
        exp_results_dir.mkdir(exist_ok=True)
        
        # Register experiment
        registry_entry = {
            'experiment_id': experiment_id,
            'name': experiment_name,
            'template': template_name,
            'config_path': str(config_path),
            'results_path': str(exp_results_dir),
            'status': ExperimentStatus.PLANNED.value,
            'created_at': datetime.now().isoformat(),
            'tags': config.get('metadata', {}).get('tags', [])
        }
        
        self._save_registry_entry(registry_entry)
        self.registry.append(registry_entry)
        
        # Update status
        self.status[experiment_id] = {
            'status': ExperimentStatus.PLANNED.value,
            'created_at': datetime.now().isoformat()
        }
        self._save_status()
        
        print(f"✅ Created experiment: {experiment_id}")
        print(f"📁 Config: {config_path}")
        print(f"📊 Results: {exp_results_dir}")
        
        return experiment_id
    
    def start_experiment(self, experiment_id: str):
        """Mark experiment as running"""
        self.status[experiment_id] = {
            'status': ExperimentStatus.RUNNING.value,
            'started_at': datetime.now().isoformat()
        }
        self._save_status()
        
        # Log to registry
        self._save_registry_entry({
            'experiment_id': experiment_id,
            'event': 'started',
            'timestamp': datetime.now().isoformat()
        })
        
        print(f"🚀 Started experiment: {experiment_id}")
    
    def complete_experiment(self, 
                           experiment_id: str,
                           results: Dict[str, Any],
                           model_path: Optional[str] = None):
        """Complete experiment with results"""
        
        # Save results
        results_path = self.results_dir / experiment_id / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Update status
        self.status[experiment_id].update({
            'status': ExperimentStatus.COMPLETED.value,
            'completed_at': datetime.now().isoformat(),
            'results': results
        })
        self._save_status()
        
        # Log to registry
        self._save_registry_entry({
            'experiment_id': experiment_id,
            'event': 'completed',
            'timestamp': datetime.now().isoformat(),
            'results': results
        })
        
        print(f"✅ Completed experiment: {experiment_id}")
        if 'rmse' in results:
            print(f"🎯 RMSE: {results['rmse']:.4f}")
    
    def fail_experiment(self, experiment_id: str, error: str):
        """Mark experiment as failed"""
        self.status[experiment_id].update({
            'status': ExperimentStatus.FAILED.value,
            'failed_at': datetime.now().isoformat(),
            'error': error
        })
        self._save_status()
        
        # Log to registry
        self._save_registry_entry({
            'experiment_id': experiment_id,
            'event': 'failed',
            'timestamp': datetime.now().isoformat(),
            'error': error
        })
        
        print(f"❌ Failed experiment: {experiment_id}")
    
    def get_experiments_dataframe(self) -> pd.DataFrame:
        """Get experiments as pandas DataFrame"""
        data = []
        
        for entry in self.registry:
            if entry.get('event') == 'completed' or 'status' in entry:
                exp_id = entry['experiment_id']
                status_info = self.status.get(exp_id, {})
                
                data.append({
                    'experiment_id': exp_id,
                    'name': entry.get('name', 'Unknown'),
                    'template': entry.get('template', 'Unknown'),
                    'status': status_info.get('status', 'Unknown'),
                    'rmse': status_info.get('results', {}).get('rmse', None),
                    'created_at': entry.get('created_at', ''),
                    'completed_at': status_info.get('completed_at', ''),
                    'tags': ', '.join(entry.get('tags', []))
                })
        
        df = pd.DataFrame(data)
        if not df.empty and 'rmse' in df.columns:
            df = df.sort_values('rmse', na_position='last')
        
        return df
    
    def compare_experiments(self, experiment_ids: List[str]) -> pd.DataFrame:
        """Compare multiple experiments"""
        data = []
        
        for exp_id in experiment_ids:
            config_path = self.configs_dir / f"{exp_id}.yaml"
            status_info = self.status.get(exp_id, {})
            
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                data.append({
                    'experiment_id': exp_id,
                    'model_type': config.get('model', {}).get('type', 'Unknown'),
                    'batch_size': config.get('training', {}).get('batch_size', None),
                    'learning_rate': config.get('training', {}).get('learning_rate', None),
                    'rmse': status_info.get('results', {}).get('rmse', None),
                    'status': status_info.get('status', 'Unknown')
                })
        
        return pd.DataFrame(data)
    
    def search_experiments(self, 
                          status: Optional[str] = None,
                          tag: Optional[str] = None,
                          min_rmse: Optional[float] = None,
                          max_rmse: Optional[float] = None) -> pd.DataFrame:
        """Search experiments with filters"""
        df = self.get_experiments_dataframe()
        
        if df.empty:
            return df
        
        # Apply filters
        if status:
            df = df[df['status'] == status]
        
        if tag:
            df = df[df['tags'].str.contains(tag, na=False)]
        
        if min_rmse is not None:
            df = df[df['rmse'] >= min_rmse]
        
        if max_rmse is not None:
            df = df[df['rmse'] <= max_rmse]
        
        return df
    
    def get_best_experiment(self, metric: str = 'rmse') -> Optional[str]:
        """Get best experiment by metric"""
        df = self.get_experiments_dataframe()
        
        if df.empty or metric not in df.columns:
            return None
        
        completed_df = df[df['status'] == 'completed']
        if completed_df.empty:
            return None
        
        if metric == 'rmse':
            best_idx = completed_df[metric].idxmin()
        else:
            best_idx = completed_df[metric].idxmax()
        
        return completed_df.loc[best_idx, 'experiment_id']
    
    def export_experiment_config(self, experiment_id: str) -> Dict[str, Any]:
        """Export experiment configuration"""
        config_path = self.configs_dir / f"{experiment_id}.yaml"
        
        if not config_path.exists():
            raise ValueError(f"Config for {experiment_id} not found")
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def clone_experiment(self, 
                        source_experiment_id: str,
                        new_name: str,
                        modifications: Dict[str, Any] = None) -> str:
        """Clone an existing experiment with modifications"""
        
        # Load source config
        source_config = self.export_experiment_config(source_experiment_id)
        
        # Apply modifications
        if modifications:
            source_config = self._deep_update(source_config, modifications)
        
        # Create new experiment
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_experiment_id = f"{new_name}_{timestamp}"
        source_config['metadata']['experiment_id'] = new_experiment_id
        source_config['metadata']['created_at'] = datetime.now().isoformat()
        source_config['metadata']['cloned_from'] = source_experiment_id
        
        # Save new config
        config_path = self.configs_dir / f"{new_experiment_id}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(source_config, f, indent=2)
        
        # Register
        registry_entry = {
            'experiment_id': new_experiment_id,
            'name': new_name,
            'cloned_from': source_experiment_id,
            'config_path': str(config_path),
            'status': ExperimentStatus.PLANNED.value,
            'created_at': datetime.now().isoformat()
        }
        
        self._save_registry_entry(registry_entry)
        self.registry.append(registry_entry)
        
        print(f"✅ Cloned experiment: {new_experiment_id}")
        print(f"📋 Source: {source_experiment_id}")
        
        return new_experiment_id
    
    @staticmethod
    def _deep_update(base_dict: Dict, update_dict: Dict) -> Dict:
        """Deep update dictionary"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                base_dict[key] = ExperimentManager._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
        return base_dict


# CLI Interface
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Experiment Manager")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # List experiments
    list_parser = subparsers.add_parser('list', help='List experiments')
    list_parser.add_argument('--status', help='Filter by status')
    list_parser.add_argument('--tag', help='Filter by tag')
    
    # Create experiment
    create_parser = subparsers.add_parser('create', help='Create experiment')
    create_parser.add_argument('template', help='Template name')
    create_parser.add_argument('name', help='Experiment name')
    create_parser.add_argument('--modifications', help='JSON modifications')
    
    # Compare experiments
    compare_parser = subparsers.add_parser('compare', help='Compare experiments')
    compare_parser.add_argument('experiments', nargs='+', help='Experiment IDs')
    
    args = parser.parse_args()
    manager = ExperimentManager()
    
    if args.command == 'list':
        df = manager.search_experiments(status=args.status, tag=args.tag)
        print(df.to_string(index=False))
    
    elif args.command == 'create':
        modifications = json.loads(args.modifications) if args.modifications else None
        exp_id = manager.create_experiment_from_template(
            args.template, args.name, modifications
        )
        print(f"Created: {exp_id}")
    
    elif args.command == 'compare':
        df = manager.compare_experiments(args.experiments)
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()