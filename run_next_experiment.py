#!/usr/bin/env python3
"""
Run Next Experiment - Advanced Regularization Targeting RMSE ≤ 0.55
"""

import sys
import subprocess
from pathlib import Path

# Add src to path
sys.path.append('src')

try:
    from versioning.model_version_manager import ModelVersionManager
except ImportError:
    print("❌ Could not import version manager")
    sys.exit(1)

def show_experiment_status():
    """Show current experiment status"""
    print("🔬 MovieLens Hybrid VAE - Experiment Status")
    print("=" * 60)
    
    manager = ModelVersionManager()
    summary = manager.get_experiment_summary()
    
    if not summary.empty:
        print("\n📊 Experiment Summary:")
        print(summary.to_string(index=False))
        
        # Get next experiment details
        next_experiments = summary[summary['Status'] == 'ExperimentStatus.PLANNED']
        if not next_experiments.empty:
            next_exp = next_experiments.iloc[0]
            print(f"\n🚀 Next Experiment Ready:")
            print(f"   ID: {next_exp['Experiment ID']}")
            print(f"   Name: {next_exp['Name']}")
            print(f"   Target RMSE: {next_exp['Target RMSE']}")
            print(f"   Expected Improvements:")
            print(f"     • Deeper architecture (4 layers vs 2)")
            print(f"     • Advanced regularization (dropout 0.4)")
            print(f"     • Data augmentation (mixup, label smoothing)")
            print(f"     • Enhanced optimization (AdamW, cosine scheduling)")
            print(f"     • Larger batches (2048 vs 1024)")
            
            return next_exp['Experiment ID']
    
    return None

def run_training(experiment_id: str):
    """Run the enhanced training script"""
    print(f"\n🚀 Starting Training for Experiment: {experiment_id}")
    print("=" * 60)
    
    # Check if required files exist
    required_files = [
        "data/processed/train_data.csv",
        "data/processed/val_data.csv", 
        "data/processed/data_mappings.pkl"
    ]
    
    missing_files = [f for f in required_files if not Path(f).exists()]
    if missing_files:
        print("❌ Missing required data files:")
        for file in missing_files:
            print(f"   • {file}")
        print("\n💡 Please ensure data is processed first:")
        print("   python src/data/etl_pipeline.py")
        return False
    
    # Activate virtual environment and run training
    cmd = [
        sys.executable,  # Use current Python interpreter
        "src/models/enhanced_training_v2.py",
        "--experiment-id", experiment_id
    ]
    
    print(f"🔧 Command: {' '.join(cmd)}")
    print("\nStarting training...")
    print("📊 You can monitor progress with Weights & Biases")
    print("⏱️  Expected training time: 4-6 hours")
    print("🎯 Target: RMSE ≤ 0.55 (vs current best: 0.806)")
    
    try:
        # Run training
        result = subprocess.run(cmd, check=True, text=True)
        
        if result.returncode == 0:
            print("\n🎉 Training completed successfully!")
            
            # Show updated status
            print("\n📊 Updated Experiment Status:")
            show_experiment_status()
            
            return True
        else:
            print("❌ Training failed")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with error: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        return False

def main():
    """Main function"""
    print("🧪 MovieLens Enhanced Training Runner")
    print("=" * 50)
    
    # Show current status
    next_exp_id = show_experiment_status()
    
    if not next_exp_id:
        print("❌ No planned experiments found")
        return
    
    # Ask user if they want to start training
    print(f"\n❓ Ready to start training experiment: {next_exp_id}?")
    print("   This will:")
    print("     • Train with advanced regularization techniques")
    print("     • Target RMSE ≤ 0.55 (significant improvement from 0.806)")
    print("     • Take approximately 4-6 hours")
    print("     • Log everything to Weights & Biases")
    print("     • Automatically version the trained model")
    
    response = input("\nStart training? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        success = run_training(next_exp_id)
        
        if success:
            print("\n🏆 Next Steps:")
            print("   • Review training results in W&B dashboard")
            print("   • Compare with baseline model (v1)")
            print("   • If RMSE ≤ 0.55 achieved, deploy to production")
            print("   • Consider ensemble methods for further improvement")
        else:
            print("\n🔧 Troubleshooting:")
            print("   • Check data files are present")
            print("   • Ensure virtual environment is activated")
            print("   • Review error logs above")
    else:
        print("Training cancelled. Run again when ready!")

if __name__ == "__main__":
    main()