#!/usr/bin/env python3
"""
Quick GCP Deployment Script for SS4Rec Training
==============================================

This script automates the Google Cloud VM creation and setup process.

Usage:
    python deploy_to_gcp.py --create-instance
    python deploy_to_gcp.py --upload-code
    python deploy_to_gcp.py --start-training
"""

import subprocess
import sys
import time
import argparse
import logging
from pathlib import Path

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def run_gcloud_command(command: list, description: str) -> bool:
    """Run a gcloud command and return success status"""
    logger = setup_logging()

    try:
        logger.info(f"🔧 {description}...")
        logger.info(f"Command: {' '.join(command)}")

        result = subprocess.run(command, capture_output=True, text=True, check=True)

        logger.info(f"✅ {description} completed")
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    logger.info(f"  {line}")

        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed")
        logger.error(f"Error: {e}")
        if e.stderr:
            logger.error(f"Stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("❌ gcloud command not found. Please install Google Cloud SDK")
        return False

def create_gpu_instance(instance_name: str = "ss4rec-training", zone: str = "us-central1-a") -> bool:
    """Create a GPU-enabled VM instance"""
    logger = setup_logging()

    logger.info(f"🚀 Creating GPU instance: {instance_name}")

    command = [
        'gcloud', 'compute', 'instances', 'create', instance_name,
        '--zone', zone,
        '--machine-type', 'n1-standard-4',
        '--accelerator', 'type=nvidia-tesla-t4,count=1',
        '--image-family', 'pytorch-2-7-cu128-ubuntu-2204-nvidia-570',
        '--image-project', 'deeplearning-platform-release',
        '--boot-disk-size', '100GB',
        '--boot-disk-type', 'pd-ssd',
        '--maintenance-policy', 'TERMINATE',
        '--metadata', 'install-nvidia-driver=True',
        '--scopes', 'cloud-platform'
    ]

    return run_gcloud_command(command, f"Creating instance {instance_name}")

def upload_codebase(instance_name: str = "ss4rec-training", zone: str = "us-central1-a") -> bool:
    """Upload the codebase to the VM instance"""
    logger = setup_logging()

    # Create archive of current directory (excluding certain directories)
    logger.info("📦 Creating codebase archive...")

    # Files to exclude
    exclude_patterns = [
        '--exclude=.git',
        '--exclude=.venv',
        '--exclude=__pycache__',
        '--exclude=*.pyc',
        '--exclude=.DS_Store',
        '--exclude=logs',
        '--exclude=results',
        '--exclude=data/raw',  # Exclude raw data (too large)
    ]

    # Create tar archive
    tar_command = ['tar', 'czf', 'ss4rec_code.tar.gz'] + exclude_patterns + ['.']

    try:
        subprocess.run(tar_command, check=True)
        logger.info("✅ Codebase archive created")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to create archive: {e}")
        return False

    # Upload to instance
    upload_command = [
        'gcloud', 'compute', 'scp',
        'ss4rec_code.tar.gz',
        f'{instance_name}:~/',
        '--zone', zone
    ]

    if not run_gcloud_command(upload_command, "Uploading codebase"):
        return False

    # Extract on instance
    extract_command = [
        'gcloud', 'compute', 'ssh', instance_name,
        '--zone', zone,
        '--command', 'tar -xzf ss4rec_code.tar.gz && rm ss4rec_code.tar.gz'
    ]

    return run_gcloud_command(extract_command, "Extracting codebase on instance")

def setup_environment(instance_name: str = "ss4rec-training", zone: str = "us-central1-a") -> bool:
    """Setup the training environment on the VM"""
    logger = setup_logging()

    # Run setup script
    setup_command = [
        'gcloud', 'compute', 'ssh', instance_name,
        '--zone', zone,
        '--command', 'chmod +x gcp_setup.sh && ./gcp_setup.sh'
    ]

    return run_gcloud_command(setup_command, "Setting up environment")

def start_training(instance_name: str = "ss4rec-training", zone: str = "us-central1-a", debug: bool = True) -> bool:
    """Start the training process"""
    logger = setup_logging()

    # Prepare training command
    training_cmd = "source activate_ss4rec.sh && "

    if debug:
        training_cmd += "python pre_training_check.py --fix-issues && "
        training_cmd += "python gcp_training.py --log-level DEBUG 2>&1 | tee training_debug.log"
    else:
        training_cmd += "python gcp_training.py 2>&1 | tee training.log"

    # Run training command
    command = [
        'gcloud', 'compute', 'ssh', instance_name,
        '--zone', zone,
        '--command', training_cmd
    ]

    logger.info("🏃 Starting training...")
    logger.info("💡 This will take several hours. You can disconnect and reconnect later.")
    logger.info(f"💡 Monitor with: gcloud compute ssh {instance_name} --zone {zone}")

    return run_gcloud_command(command, "Starting training process")

def monitor_training(instance_name: str = "ss4rec-training", zone: str = "us-central1-a") -> bool:
    """Connect to instance for monitoring"""
    logger = setup_logging()

    logger.info("📊 Connecting to instance for monitoring...")
    logger.info("💡 Use these commands once connected:")
    logger.info("  tail -f training_debug.log     # Monitor training progress")
    logger.info("  watch -n 2 nvidia-smi         # Monitor GPU usage")
    logger.info("  tail -f logs/stability/stability_*.log  # Monitor stability")

    # Connect to instance
    command = [
        'gcloud', 'compute', 'ssh', instance_name,
        '--zone', zone
    ]

    try:
        subprocess.run(command)
        return True
    except KeyboardInterrupt:
        logger.info("👋 Disconnected from instance")
        return True
    except Exception as e:
        logger.error(f"❌ Connection failed: {e}")
        return False

def cleanup_resources(instance_name: str = "ss4rec-training", zone: str = "us-central1-a") -> bool:
    """Stop or delete the instance"""
    logger = setup_logging()

    response = input(f"Delete instance {instance_name}? (y/N): ")

    if response.lower() == 'y':
        command = [
            'gcloud', 'compute', 'instances', 'delete', instance_name,
            '--zone', zone,
            '--quiet'
        ]
        return run_gcloud_command(command, f"Deleting instance {instance_name}")
    else:
        command = [
            'gcloud', 'compute', 'instances', 'stop', instance_name,
            '--zone', zone
        ]
        return run_gcloud_command(command, f"Stopping instance {instance_name}")

def main():
    parser = argparse.ArgumentParser(description='Deploy SS4Rec training to Google Cloud')

    parser.add_argument('--instance-name', default='ss4rec-training',
                       help='Name of the VM instance')
    parser.add_argument('--zone', default='us-central1-a',
                       help='Google Cloud zone')

    # Action arguments
    parser.add_argument('--create-instance', action='store_true',
                       help='Create a new GPU instance')
    parser.add_argument('--upload-code', action='store_true',
                       help='Upload codebase to existing instance')
    parser.add_argument('--setup-env', action='store_true',
                       help='Setup environment on instance')
    parser.add_argument('--start-training', action='store_true',
                       help='Start training process')
    parser.add_argument('--monitor', action='store_true',
                       help='Connect to instance for monitoring')
    parser.add_argument('--cleanup', action='store_true',
                       help='Stop or delete instance')
    parser.add_argument('--full-deploy', action='store_true',
                       help='Run complete deployment (create + upload + setup + train)')

    # Training options
    parser.add_argument('--production', action='store_true',
                       help='Run production training (not debug mode)')

    args = parser.parse_args()

    logger = setup_logging()

    logger.info("🌟 SS4Rec Google Cloud Deployment Script")
    logger.info("=" * 50)

    success = True

    if args.full_deploy:
        logger.info("🚀 Running full deployment pipeline...")
        success &= create_gpu_instance(args.instance_name, args.zone)
        if success:
            logger.info("⏳ Waiting 60 seconds for instance to boot...")
            time.sleep(60)
            success &= upload_codebase(args.instance_name, args.zone)
        if success:
            success &= setup_environment(args.instance_name, args.zone)
        if success:
            success &= start_training(args.instance_name, args.zone, not args.production)
    else:
        if args.create_instance:
            success &= create_gpu_instance(args.instance_name, args.zone)

        if args.upload_code:
            success &= upload_codebase(args.instance_name, args.zone)

        if args.setup_env:
            success &= setup_environment(args.instance_name, args.zone)

        if args.start_training:
            success &= start_training(args.instance_name, args.zone, not args.production)

        if args.monitor:
            success &= monitor_training(args.instance_name, args.zone)

        if args.cleanup:
            success &= cleanup_resources(args.instance_name, args.zone)

    if success:
        logger.info("🎉 All operations completed successfully!")

        if args.create_instance or args.full_deploy:
            logger.info("")
            logger.info("🔗 Next steps:")
            logger.info(f"  Monitor: gcloud compute ssh {args.instance_name} --zone {args.zone}")
            logger.info("  Stop: python deploy_to_gcp.py --cleanup")

        return 0
    else:
        logger.error("❌ Some operations failed. Check logs above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)