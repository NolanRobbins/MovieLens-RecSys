#!/usr/bin/env python3
"""
Download official SS4Rec implementation
Following NEXT_STEPS.md Step 2: Download Official SS4Rec Requirements

This script:
1. Clones the official SS4Rec repository from GitHub
2. Copies important files to models/official_ss4rec/
3. Prepares sequential_dataset.py for RecBole integration
"""

import subprocess
import os
import shutil
from pathlib import Path

def main():
    print("📥 STEP 2: Downloading official SS4Rec implementation...")
    
    # Check if we're in the right directory
    if not os.path.exists('runpod_entrypoint.sh'):
        print("❌ Not in MovieLens-RecSys directory. Please run this from the project root.")
        return False
    
    print("✅ In correct directory")
    
    # Create a temporary directory for downloading
    temp_dir = Path("temp_ss4rec_official")
    if temp_dir.exists():
        print(f"🗑️ Removing existing {temp_dir}")
        shutil.rmtree(temp_dir)
    
    try:
        # Clone the official SS4Rec repository
        print("📥 Cloning official SS4Rec repository...")
        result = subprocess.run([
            'git', 'clone', 
            'https://github.com/XiaoWei-i/SS4Rec.git', 
            str(temp_dir)
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Successfully cloned official SS4Rec repository")
            
            # Check what files we got
            if (temp_dir / "sequential_dataset.py").exists():
                print("✅ Found sequential_dataset.py in official implementation")
                
                # Copy the official sequential_dataset.py to our project
                target_dir = Path("models/official_ss4rec")
                target_dir.mkdir(parents=True, exist_ok=True)
                
                shutil.copy2(temp_dir / "sequential_dataset.py", target_dir / "sequential_dataset_official.py")
                print(f"✅ Copied official sequential_dataset.py to {target_dir}/")
                
                # Also copy other important files if they exist
                important_files = ['SS4Rec.py', 'SS4Rec_sequential.py', 'README.md']
                for file in important_files:
                    if (temp_dir / file).exists():
                        shutil.copy2(temp_dir / file, target_dir / file)
                        print(f"✅ Copied {file} to {target_dir}/")
                
                # List all files in the official repository
                print("\n📁 Files in official SS4Rec repository:")
                for item in sorted(temp_dir.iterdir()):
                    if item.is_file():
                        size = item.stat().st_size
                        print(f"  📄 {item.name} ({size:,} bytes)")
                    else:
                        print(f"  📁 {item.name}/")
                
                print("\n🎯 Official SS4Rec files ready for integration!")
                
            else:
                print("❌ sequential_dataset.py not found in official repository")
                print("Available files:")
                for item in sorted(temp_dir.iterdir()):
                    print(f"  - {item.name}")
                    
        else:
            print(f"❌ Failed to clone repository: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Git clone timed out. Please check your internet connection.")
        return False
    except Exception as e:
        print(f"❌ Error downloading official SS4Rec: {e}")
        return False
    
    # Clean up temporary directory
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        print("🗑️ Cleaned up temporary directory")
    
    # Verify files were copied
    official_dir = Path("models/official_ss4rec")
    if official_dir.exists():
        files = list(official_dir.iterdir())
        print(f"\n📁 Files in {official_dir}:")
        for file in files:
            if file.is_file():
                size = file.stat().st_size
                print(f"  ✅ {file.name} ({size:,} bytes)")
    
    print("\n🎯 STEP 2 COMPLETE: Official SS4Rec implementation downloaded!")
    print("📁 Check models/official_ss4rec/ for the official files")
    print("💡 Next: Integrate official sequential_dataset.py with RecBole")
    
    return True

if __name__ == "__main__":
    main()
