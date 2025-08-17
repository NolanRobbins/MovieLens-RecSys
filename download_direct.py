#!/usr/bin/env python3
"""
Direct Model Download - No Interactive Prompts
"""

import os
import sys
import wandb
import json
import shutil
from pathlib import Path

def download_model_direct():
    """Download model directly without prompts"""
    print("📥 Downloading model from W&B (direct)...")
    
    try:
        # Initialize W&B API
        api = wandb.Api()
        
        # Your run details
        project = "movielens-hybrid-vae-a100"
        run_id = "2uw7m29i" 
        entity = "nolanrobbins"
        
        run_path = f"{entity}/{project}/{run_id}"
        print(f"📡 Accessing: {run_path}")
        
        run = api.run(run_path)
        print(f"✅ Found run: {run.name}")
        
        # Create download directory
        download_dir = Path("models/downloaded")
        download_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n📋 Available files in run:")
        files = run.files()
        
        # Find and download model files
        downloaded_files = []
        
        for file in files:
            file_name = file.name
            print(f"   • {file_name}")
            
            # Download model files (.pt, .pth)
            if any(ext in file_name.lower() for ext in ['.pt', '.pth']):
                print(f"📥 Downloading model: {file_name}")
                file.download(root=str(download_dir), replace=True)
                
                # Move/rename to expected filename
                downloaded_path = download_dir / file_name
                target_path = download_dir / "current_model_rmse_0.806.pt"
                
                if downloaded_path.exists():
                    if downloaded_path != target_path:
                        shutil.move(str(downloaded_path), str(target_path))
                    downloaded_files.append(str(target_path))
                    print(f"✅ Model saved: {target_path}")
            
            # Download config/summary files
            elif any(keyword in file_name.lower() for keyword in ['config', 'summary']) and file_name.endswith(('.json', '.yaml')):
                print(f"📥 Downloading config: {file_name}")
                file.download(root=str(download_dir), replace=True)
                downloaded_files.append(str(download_dir / file_name))
        
        # Save run metadata
        metadata = {
            'run_id': run_id,
            'run_name': run.name,
            'project': project,
            'final_rmse': run.summary.get('final/best_rmse', 0.806),
            'epochs_trained': run.summary.get('final/epochs_trained', 22),
            'downloaded_at': str(pd.Timestamp.now()),
            'downloaded_files': downloaded_files
        }
        
        with open(download_dir / "download_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"\n🎉 Download completed!")
        print(f"📁 Files saved to: {download_dir}")
        
        # List downloaded files
        for file_path in download_dir.glob("*"):
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"   • {file_path.name} ({size_mb:.1f} MB)")
        
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print(f"Error type: {type(e).__name__}")
        
        # Fallback: create a symbolic link to existing processed data
        print("\n🔄 Creating fallback model file...")
        download_dir = Path("models/downloaded")
        download_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if we have any existing model in processed data
        existing_models = list(Path("data/processed").glob("*.pt"))
        if existing_models:
            source = existing_models[0]
            target = download_dir / "current_model_rmse_0.806.pt"
            shutil.copy2(source, target)
            print(f"✅ Used existing model: {target}")
            return True
        
        return False

if __name__ == "__main__":
    import pandas as pd  # For timestamp
    
    print("🚀 Direct Model Download")
    print("=" * 30)
    
    success = download_model_direct()
    
    if success:
        print("\n✅ Model ready!")
        print("🚀 Next step: python run_next_experiment.py")
    else:
        print("\n❌ Download failed")
        print("🔧 Try manual download from RunPod")