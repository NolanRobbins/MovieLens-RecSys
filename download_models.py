#!/usr/bin/env python3
"""
Quick Model Download Script
Simple script to download your trained models from RunPod
"""

import os
import subprocess
import sys
from pathlib import Path

def download_current_model_scp():
    """Download current model via SCP (most reliable method)"""
    print("🔄 Downloading current model (RMSE: 0.806) via SCP...")
    
    # You'll need to replace these with your actual RunPod details
    runpod_ip = input("Enter your RunPod IP address: ").strip()
    
    if not runpod_ip:
        print("❌ RunPod IP required")
        return False
    
    # Create local models directory
    models_dir = Path("models/downloaded")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Download commands
    files_to_download = [
        {
            'remote': '/workspace/models/hybrid_vae_best.pt',
            'local': 'models/downloaded/current_model_rmse_0.806.pt'
        },
        {
            'remote': '/workspace/models/training_summary.json',
            'local': 'models/downloaded/current_training_summary.json'
        },
        {
            'remote': '/workspace/wandb/latest-run/files/config.yaml',
            'local': 'models/downloaded/current_config.yaml'
        }
    ]
    
    success_count = 0
    for file_info in files_to_download:
        print(f"Downloading {Path(file_info['remote']).name}...")
        
        cmd = [
            'scp', '-o', 'StrictHostKeyChecking=no',
            f"root@{runpod_ip}:{file_info['remote']}",
            file_info['local']
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Downloaded: {file_info['local']}")
                success_count += 1
            else:
                print(f"⚠️  Failed to download {Path(file_info['remote']).name}: {result.stderr}")
        except Exception as e:
            print(f"⚠️  Error downloading {Path(file_info['remote']).name}: {e}")
    
    if success_count > 0:
        print(f"\n🎉 Successfully downloaded {success_count}/{len(files_to_download)} files")
        print(f"📁 Files saved to: {models_dir.absolute()}")
        return True
    else:
        print("❌ No files downloaded successfully")
        return False

def download_via_rsync():
    """Download entire models directory via rsync (more efficient)"""
    print("🔄 Downloading all models via rsync...")
    
    runpod_ip = input("Enter your RunPod IP address: ").strip()
    
    if not runpod_ip:
        print("❌ RunPod IP required")
        return False
    
    # Create local directory
    models_dir = Path("models/downloaded")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Rsync command
    cmd = [
        'rsync', '-avz', '--progress',
        f"root@{runpod_ip}:/workspace/models/",
        str(models_dir) + "/"
    ]
    
    try:
        print("Starting rsync transfer...")
        result = subprocess.run(cmd, text=True)
        
        if result.returncode == 0:
            print(f"✅ All models downloaded successfully")
            print(f"📁 Files saved to: {models_dir.absolute()}")
            
            # List downloaded files
            print("\nDownloaded files:")
            for file_path in models_dir.rglob("*"):
                if file_path.is_file():
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    print(f"  📄 {file_path.name} ({size_mb:.1f} MB)")
            
            return True
        else:
            print("❌ Rsync failed")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def download_from_wandb():
    """Download model artifacts from Weights & Biases"""
    print("🔄 Downloading model from Weights & Biases...")
    
    try:
        import wandb
    except ImportError:
        print("❌ wandb not installed. Run: pip install wandb")
        return False
    
    # Your W&B project details
    project = "movielens-hybrid-vae-a100"
    run_id = "2uw7m29i"  # From your recent training
    
    try:
        # Initialize W&B API
        api = wandb.Api()
        run = api.run(f"nolanrobbins/{project}/{run_id}")
        
        # Download model files
        models_dir = Path("models/downloaded/wandb")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Download run files
        print("Downloading run files...")
        files = run.files()
        
        model_files = [f for f in files if f.name.endswith(('.pt', '.pth', '.ckpt'))]
        config_files = [f for f in files if 'config' in f.name or f.name.endswith('.yaml')]
        
        downloaded_count = 0
        
        for file in model_files + config_files:
            local_path = models_dir / file.name
            print(f"Downloading {file.name}...")
            file.download(root=str(models_dir), replace=True)
            downloaded_count += 1
            print(f"✅ Downloaded: {local_path}")
        
        if downloaded_count > 0:
            print(f"\n🎉 Downloaded {downloaded_count} files from W&B")
            print(f"📁 Files saved to: {models_dir.absolute()}")
            return True
        else:
            print("⚠️  No model files found in W&B run")
            return False
            
    except Exception as e:
        print(f"❌ W&B download failed: {e}")
        return False

def show_download_options():
    """Show all available download methods"""
    print("""
📥 Model Download Options:

1. SCP Download (Recommended)
   - Downloads specific model files
   - Most reliable method
   - Requires RunPod IP address

2. Rsync Download 
   - Downloads entire models directory
   - Faster for multiple files
   - Shows progress

3. Weights & Biases
   - Downloads from your W&B project
   - Includes experiment metadata
   - No RunPod access needed

4. Manual Download
   - Use RunPod's web interface
   - Navigate to /workspace/models/
   - Download files directly

Choose your preferred method:
""")

def main():
    """Main download interface"""
    print("=" * 60)
    print("🚀 MovieLens Model Download Manager")
    print("=" * 60)
    
    show_download_options()
    
    while True:
        choice = input("Enter choice (1-4) or 'q' to quit: ").strip().lower()
        
        if choice == 'q':
            print("👋 Goodbye!")
            break
        elif choice == '1':
            download_current_model_scp()
        elif choice == '2':
            download_via_rsync()
        elif choice == '3':
            download_from_wandb()
        elif choice == '4':
            print("""
Manual Download Instructions:
1. Open your RunPod instance web interface
2. Navigate to /workspace/models/ directory
3. Download these files:
   - hybrid_vae_best.pt (main model)
   - training_summary.json (training stats)
   - Any .yaml config files
4. Save to your local models/ directory
            """)
        else:
            print("❌ Invalid choice. Please enter 1-4 or 'q'")
        
        print("\n" + "-" * 40 + "\n")

if __name__ == "__main__":
    main()