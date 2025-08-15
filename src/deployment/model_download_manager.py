"""
Model Download Manager
Comprehensive system for downloading trained models from RunPod/cloud instances
"""

import os
import json
import boto3
import paramiko
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import zipfile
import tarfile
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    """Model metadata for tracking and downloading"""
    model_name: str
    experiment_id: str
    timestamp: str
    rmse_score: float
    model_path: str
    config_path: Optional[str] = None
    wandb_run_id: Optional[str] = None
    file_size_mb: Optional[float] = None
    download_url: Optional[str] = None

class ModelDownloadManager:
    """Manages model downloads from various cloud platforms"""
    
    def __init__(self, local_models_dir: str = "models/downloaded"):
        self.local_models_dir = Path(local_models_dir)
        self.local_models_dir.mkdir(parents=True, exist_ok=True)
        
        # Model registry for tracking downloads
        self.registry_file = self.local_models_dir / "model_registry.json"
        self.model_registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Dict]:
        """Load model registry from disk"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_registry(self):
        """Save model registry to disk"""
        with open(self.registry_file, 'w') as f:
            json.dump(self.model_registry, f, indent=2, default=str)
    
    def register_model(self, model_info: ModelInfo):
        """Register a model in the local registry"""
        self.model_registry[model_info.experiment_id] = {
            'model_name': model_info.model_name,
            'experiment_id': model_info.experiment_id,
            'timestamp': model_info.timestamp,
            'rmse_score': model_info.rmse_score,
            'model_path': model_info.model_path,
            'config_path': model_info.config_path,
            'wandb_run_id': model_info.wandb_run_id,
            'file_size_mb': model_info.file_size_mb,
            'download_url': model_info.download_url,
            'local_path': None,
            'downloaded_at': None
        }
        self._save_registry()
        logger.info(f"Registered model: {model_info.model_name} (RMSE: {model_info.rmse_score})")

class RunPodDownloader:
    """Download models from RunPod instances"""
    
    def __init__(self, manager: ModelDownloadManager):
        self.manager = manager
    
    def download_via_ssh(self, host: str, username: str, password: str, 
                        remote_path: str, experiment_id: str) -> str:
        """Download model via SSH/SCP"""
        logger.info(f"🔄 Downloading model from RunPod: {host}")
        
        try:
            # Setup SSH client
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(host, username=username, password=password)
            
            # Setup SCP client
            scp = ssh.open_sftp()
            
            # Download model file
            local_filename = f"{experiment_id}_model.pt"
            local_path = self.manager.local_models_dir / local_filename
            
            scp.get(remote_path, str(local_path))
            
            # Download additional files if they exist
            additional_files = [
                f"{remote_path.replace('.pt', '_config.json')}",
                f"{remote_path.replace('.pt', '_summary.json')}",
                f"{remote_path.replace('.pt', '_evaluation.json')}"
            ]
            
            for remote_file in additional_files:
                try:
                    local_file = local_path.parent / Path(remote_file).name
                    scp.get(remote_file, str(local_file))
                    logger.info(f"✅ Downloaded: {Path(remote_file).name}")
                except:
                    logger.warning(f"⚠️  Optional file not found: {Path(remote_file).name}")
            
            # Update registry
            if experiment_id in self.manager.model_registry:
                self.manager.model_registry[experiment_id]['local_path'] = str(local_path)
                self.manager.model_registry[experiment_id]['downloaded_at'] = datetime.now().isoformat()
                self.manager._save_registry()
            
            scp.close()
            ssh.close()
            
            logger.info(f"✅ Model downloaded successfully: {local_path}")
            return str(local_path)
            
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            raise

    def download_via_web_interface(self, download_url: str, experiment_id: str) -> str:
        """Download model via RunPod web interface/direct URL"""
        logger.info(f"🔄 Downloading model from URL: {download_url}")
        
        local_filename = f"{experiment_id}_model.pt"
        local_path = self.manager.local_models_dir / local_filename
        
        try:
            response = requests.get(download_url, stream=True)
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Update registry
            if experiment_id in self.manager.model_registry:
                self.manager.model_registry[experiment_id]['local_path'] = str(local_path)
                self.manager.model_registry[experiment_id]['downloaded_at'] = datetime.now().isoformat()
                self.manager._save_registry()
            
            logger.info(f"✅ Model downloaded successfully: {local_path}")
            return str(local_path)
            
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            raise

class WandBDownloader:
    """Download models from Weights & Biases"""
    
    def __init__(self, manager: ModelDownloadManager):
        self.manager = manager
        try:
            import wandb
            self.wandb = wandb
        except ImportError:
            logger.error("wandb not installed. Run: pip install wandb")
            self.wandb = None
    
    def download_model_artifact(self, run_id: str, project: str, 
                               artifact_name: str = "model") -> str:
        """Download model artifact from W&B"""
        if not self.wandb:
            raise ImportError("wandb not installed")
        
        logger.info(f"🔄 Downloading model from W&B: {project}/{run_id}")
        
        try:
            # Initialize W&B API
            api = self.wandb.Api()
            run = api.run(f"{project}/{run_id}")
            
            # Download artifact
            artifact = run.use_artifact(f"{artifact_name}:latest")
            artifact_dir = artifact.download(root=str(self.manager.local_models_dir))
            
            # Find model file
            model_files = list(Path(artifact_dir).glob("*.pt"))
            if model_files:
                model_path = model_files[0]
                logger.info(f"✅ Model downloaded from W&B: {model_path}")
                return str(model_path)
            else:
                raise FileNotFoundError("No .pt model file found in artifact")
                
        except Exception as e:
            logger.error(f"❌ W&B download failed: {e}")
            raise

class CloudStorageDownloader:
    """Download models from cloud storage (S3, GCS, etc.)"""
    
    def __init__(self, manager: ModelDownloadManager):
        self.manager = manager
    
    def download_from_s3(self, bucket: str, key: str, experiment_id: str,
                        aws_access_key: str = None, aws_secret_key: str = None) -> str:
        """Download model from S3"""
        logger.info(f"🔄 Downloading model from S3: s3://{bucket}/{key}")
        
        try:
            # Setup S3 client
            if aws_access_key and aws_secret_key:
                s3 = boto3.client('s3', 
                                aws_access_key_id=aws_access_key,
                                aws_secret_access_key=aws_secret_key)
            else:
                s3 = boto3.client('s3')  # Use default credentials
            
            # Download model
            local_filename = f"{experiment_id}_model.pt"
            local_path = self.manager.local_models_dir / local_filename
            
            s3.download_file(bucket, key, str(local_path))
            
            # Update registry
            if experiment_id in self.manager.model_registry:
                self.manager.model_registry[experiment_id]['local_path'] = str(local_path)
                self.manager.model_registry[experiment_id]['downloaded_at'] = datetime.now().isoformat()
                self.manager._save_registry()
            
            logger.info(f"✅ Model downloaded from S3: {local_path}")
            return str(local_path)
            
        except Exception as e:
            logger.error(f"❌ S3 download failed: {e}")
            raise

class ModelArchiveManager:
    """Create and manage model archives"""
    
    def __init__(self, manager: ModelDownloadManager):
        self.manager = manager
    
    def create_model_archive(self, experiment_id: str, 
                           include_config: bool = True,
                           include_evaluation: bool = True) -> str:
        """Create a complete model archive with all artifacts"""
        logger.info(f"📦 Creating model archive for experiment: {experiment_id}")
        
        if experiment_id not in self.manager.model_registry:
            raise ValueError(f"Experiment {experiment_id} not found in registry")
        
        model_info = self.manager.model_registry[experiment_id]
        archive_name = f"{experiment_id}_complete.tar.gz"
        archive_path = self.manager.local_models_dir / archive_name
        
        try:
            with tarfile.open(archive_path, 'w:gz') as tar:
                # Add model file
                if model_info['local_path'] and Path(model_info['local_path']).exists():
                    tar.add(model_info['local_path'], arcname=f"{experiment_id}_model.pt")
                
                # Add config files
                if include_config:
                    config_files = list(self.manager.local_models_dir.glob(f"{experiment_id}*config*.json"))
                    for config_file in config_files:
                        tar.add(config_file, arcname=config_file.name)
                
                # Add evaluation files
                if include_evaluation:
                    eval_files = list(self.manager.local_models_dir.glob(f"{experiment_id}*evaluation*.json"))
                    for eval_file in eval_files:
                        tar.add(eval_file, arcname=eval_file.name)
                
                # Add metadata
                metadata = {
                    'experiment_info': model_info,
                    'archive_created_at': datetime.now().isoformat(),
                    'archive_version': '1.0'
                }
                
                metadata_path = self.manager.local_models_dir / f"{experiment_id}_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
                
                tar.add(metadata_path, arcname=f"{experiment_id}_metadata.json")
                metadata_path.unlink()  # Clean up temp file
            
            logger.info(f"✅ Model archive created: {archive_path}")
            return str(archive_path)
            
        except Exception as e:
            logger.error(f"❌ Archive creation failed: {e}")
            raise

def create_download_commands() -> Dict[str, str]:
    """Generate download commands for common scenarios"""
    return {
        'runpod_ssh': '''
# Download via SSH (replace with your RunPod details)
python -c "
from src.deployment.model_download_manager import ModelDownloadManager, RunPodDownloader
manager = ModelDownloadManager()
downloader = RunPodDownloader(manager)
downloader.download_via_ssh(
    host='your-runpod-ip',
    username='root',
    password='your-password', 
    remote_path='/workspace/models/hybrid_vae_best.pt',
    experiment_id='a100-experiment-0815-1805-rmse0.55'
)
"
        ''',
        
        'runpod_web': '''
# Download via web interface URL
python -c "
from src.deployment.model_download_manager import ModelDownloadManager, RunPodDownloader
manager = ModelDownloadManager()
downloader = RunPodDownloader(manager)
downloader.download_via_web_interface(
    download_url='https://your-runpod-instance.com/download/model.pt',
    experiment_id='a100-experiment-0815-1805-rmse0.55'
)
"
        ''',
        
        'wandb': '''
# Download from Weights & Biases
python -c "
from src.deployment.model_download_manager import ModelDownloadManager, WandBDownloader
manager = ModelDownloadManager()
downloader = WandBDownloader(manager)
downloader.download_model_artifact(
    run_id='2uw7m29i',
    project='movielens-hybrid-vae-a100',
    artifact_name='model'
)
"
        ''',
        
        's3': '''
# Download from S3
python -c "
from src.deployment.model_download_manager import ModelDownloadManager, CloudStorageDownloader
manager = ModelDownloadManager()
downloader = CloudStorageDownloader(manager)
downloader.download_from_s3(
    bucket='my-ml-models',
    key='movielens/hybrid_vae_best.pt',
    experiment_id='a100-experiment-0815-1805-rmse0.55'
)
"
        '''
    }

# Example usage functions
def download_current_model():
    """Download the model from your most recent training run"""
    manager = ModelDownloadManager()
    
    # Register your current model
    current_model = ModelInfo(
        model_name="hybrid_vae_current",
        experiment_id="a100-experiment-0815-1805-rmse0.806",
        timestamp="2025-08-15T18:05:01",
        rmse_score=0.806,
        model_path="/workspace/models/hybrid_vae_best.pt",
        wandb_run_id="2uw7m29i"
    )
    manager.register_model(current_model)
    
    # Download via RunPod SSH (replace with your details)
    runpod_downloader = RunPodDownloader(manager)
    return runpod_downloader.download_via_ssh(
        host="your-runpod-ip",
        username="root", 
        password="your-password",
        remote_path="/workspace/models/hybrid_vae_best.pt",
        experiment_id=current_model.experiment_id
    )

def download_next_experiment_model():
    """Download the model from your next advanced experiment"""
    manager = ModelDownloadManager()
    
    # Register your next experiment model (when it's ready)
    next_model = ModelInfo(
        model_name="hybrid_vae_advanced_v2",
        experiment_id="a100-advanced-experiment-targeting-rmse-0.55",
        timestamp=datetime.now().isoformat(),
        rmse_score=0.55,  # Target RMSE
        model_path="/workspace/models/hybrid_vae_advanced_best.pt"
    )
    manager.register_model(next_model)
    
    # Multiple download options will be available
    print("Download options for next experiment:")
    commands = create_download_commands()
    for method, command in commands.items():
        print(f"\n{method.upper()}:")
        print(command)

if __name__ == "__main__":
    # Demo usage
    print("Model Download Manager - Available Commands:")
    print("=" * 50)
    
    commands = create_download_commands()
    for method, command in commands.items():
        print(f"\n{method.upper()} Download:")
        print(command)