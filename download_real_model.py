#!/usr/bin/env python3
"""
Download Real Model from Weights & Biases
Replace mock model with actual trained model from your run
"""

import os
import sys
import wandb
import json
import shutil
from pathlib import Path
from typing import Optional

def download_from_wandb(project: str = "movielens-hybrid-vae-a100", 
                       run_id: str = "2uw7m29i",
                       entity: str = "nolanrobbins") -> bool:
    """Download model from W&B"""
    print(f"🔄 Downloading model from W&B...")
    print(f"   Project: {project}")
    print(f"   Run ID: {run_id}")
    print(f"   Entity: {entity}")
    
    try:
        # Initialize W&B API (will prompt for login if needed)
        api = wandb.Api()
        
        # Get the run
        run_path = f"{entity}/{project}/{run_id}"
        print(f"📡 Accessing run: {run_path}")
        
        run = api.run(run_path)
        print(f"✅ Found run: {run.name}")
        print(f"   State: {run.state}")
        print(f"   RMSE: {run.summary.get('final/best_rmse', 'N/A')}")
        
        # Create download directory
        download_dir = Path("models/downloaded")
        download_dir.mkdir(parents=True, exist_ok=True)
        
        # Download files
        print("\n📥 Downloading files...")
        files = run.files()
        
        model_files = []
        config_files = []
        
        for file in files:
            file_name = file.name
            print(f"   Found: {file_name}")
            
            # Check if it's a model file
            if any(ext in file_name.lower() for ext in ['.pt', '.pth', '.ckpt']):
                model_files.append(file)
            elif any(keyword in file_name.lower() for keyword in ['config', 'summary', '.yaml', '.json']):
                config_files.append(file)
        
        # Download model files
        if model_files:
            for file in model_files:
                local_path = download_dir / f"current_model_rmse_0.806.pt"
                print(f"📥 Downloading model: {file.name} -> {local_path}")
                file.download(root=str(download_dir), replace=True)
                
                # Rename to our expected filename
                downloaded_path = download_dir / file.name
                if downloaded_path.exists() and downloaded_path != local_path:
                    shutil.move(str(downloaded_path), str(local_path))
                
                print(f"✅ Model downloaded: {local_path}")
        else:
            print("⚠️  No model files found in run")
        
        # Download config files
        for file in config_files:
            local_path = download_dir / file.name
            print(f"📥 Downloading config: {file.name}")
            file.download(root=str(download_dir), replace=True)
            print(f"✅ Config downloaded: {local_path}")
        
        # Get run summary and save metadata
        summary_data = {
            'run_id': run_id,
            'run_name': run.name,
            'project': project,
            'entity': entity,
            'state': run.state,
            'created_at': run.created_at.isoformat() if run.created_at else None,
            'runtime': run.summary.get('_runtime', 0),
            'final_rmse': run.summary.get('final/best_rmse', None),
            'epochs_trained': run.summary.get('final/epochs_trained', None),
            'training_time_hours': run.summary.get('final/total_time_hours', None),
            'config': dict(run.config) if run.config else {},
            'summary': dict(run.summary)
        }
        
        metadata_path = download_dir / "current_model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        print(f"✅ Metadata saved: {metadata_path}")
        
        return True
        
    except wandb.errors.CommError as e:
        print(f"❌ W&B API error: {e}")
        print("💡 Try: wandb login")
        return False
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

def check_wandb_login() -> bool:
    """Check if user is logged into W&B"""
    try:
        api = wandb.Api()
        user = api.viewer
        print(f"✅ Logged in to W&B as: {user.get('username', 'unknown')}")
        return True
    except Exception:
        return False

def prompt_wandb_login():
    """Prompt user to login to W&B"""
    print("🔐 W&B login required")
    print("Please run: wandb login")
    print("Or visit: https://wandb.ai/authorize")
    
    response = input("\nHave you logged in? (y/N): ").strip().lower()
    return response in ['y', 'yes']

def update_version_registry(model_path: str):
    """Update the version registry with real model path"""
    print("📝 Updating version registry with real model...")
    
    try:
        # Import version manager
        sys.path.append('src')
        from versioning.model_version_manager import ModelVersionManager
        
        manager = ModelVersionManager()
        
        # Find the v1 experiment
        v1_experiments = [exp_id for exp_id, exp in manager.registry.items() 
                         if 'v1' in exp_id and exp['status'] == 'completed']
        
        if v1_experiments:
            exp_id = v1_experiments[0]
            
            # Update model path
            manager.registry[exp_id]['model_path'] = str(model_path)
            
            # Copy to versioned directory
            exp_dir = Path("models/versions") / exp_id
            versioned_path = exp_dir / f"model_{exp_id}.pt"
            
            if Path(model_path).exists():
                shutil.copy2(model_path, versioned_path)
                manager.registry[exp_id]['versioned_model_path'] = str(versioned_path)
            
            manager._save_registry()
            print(f"✅ Updated experiment {exp_id} with real model path")
        else:
            print("⚠️  No v1 experiment found to update")
            
    except Exception as e:
        print(f"⚠️  Could not update registry: {e}")

def main():
    """Main download function"""
    print("📥 Download Real Model from Weights & Biases")
    print("=" * 50)
    
    # Check if already logged in
    if not check_wandb_login():
        print("❌ Not logged in to W&B")
        
        if not prompt_wandb_login():
            print("❌ W&B login required. Exiting.")
            return False
        
        # Check again after user claims to have logged in
        if not check_wandb_login():
            print("❌ Still not logged in. Please run: wandb login")
            return False
    
    # Download model
    success = download_from_wandb()
    
    if success:
        # Check if model file exists
        model_path = Path("models/downloaded/current_model_rmse_0.806.pt")
        
        if model_path.exists():
            file_size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"\n🎉 Download successful!")
            print(f"📄 Model file: {model_path} ({file_size_mb:.1f} MB)")
            
            # Update version registry
            update_version_registry(str(model_path))
            
            print(f"\n✅ Real model ready for training!")
            print(f"📁 Files downloaded to: {model_path.parent}")
            
            # Show what was downloaded
            download_files = list(model_path.parent.glob("*"))
            print(f"\n📋 Downloaded files:")
            for file in download_files:
                if file.is_file():
                    size_mb = file.stat().st_size / (1024 * 1024)
                    print(f"   • {file.name} ({size_mb:.1f} MB)")
            
            return True
        else:
            print("❌ Model file not found after download")
            return False
    else:
        print("❌ Download failed")
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🚀 Ready for next step:")
        print("   python run_next_experiment.py")
    else:
        print("\n🔧 Alternative download methods:")
        print("   1. RunPod SSH: scp root@YOUR_IP:/workspace/models/hybrid_vae_best.pt models/downloaded/")
        print("   2. Manual download from RunPod web interface")
        print("   3. Check W&B run page directly: https://wandb.ai/nolanrobbins/movielens-hybrid-vae-a100/runs/2uw7m29i")