#!/usr/bin/env python3
"""
CPU-Only GCP Deployment for SS4Rec (while waiting for GPU quota)
===============================================================

This creates a CPU instance to test the deployment pipeline
while we wait for GPU quota approval.
"""

import subprocess
import sys
import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def create_cpu_instance():
    logger = setup_logging()

    logger.info("🚀 Creating CPU-only instance for testing...")

    command = [
        'gcloud', 'compute', 'instances', 'create', 'ss4rec-cpu-test',
        '--zone', 'us-central1-a',
        '--machine-type', 'n1-standard-4',
        '--image-family', 'ubuntu-2004-lts',
        '--image-project', 'ubuntu-os-cloud',
        '--boot-disk-size', '50GB',
        '--scopes', 'cloud-platform'
    ]

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        logger.info("✅ CPU instance created successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to create instance: {e}")
        logger.error(f"Stderr: {e.stderr}")
        return False

if __name__ == "__main__":
    success = create_cpu_instance()
    if success:
        print("\n🎯 Next steps for GPU quota:")
        print("1. Go to: https://console.cloud.google.com/iam-admin/quotas")
        print("2. Filter by: 'GPUs (all regions)'")
        print("3. Select the quota and click 'EDIT QUOTAS'")
        print("4. Request increase to 1 GPU")
        print("5. Provide justification: 'Machine learning research and model training'")
        print("\n⏱️ Quota approval usually takes 24-48 hours")
        print("💡 Meanwhile, you can test the deployment on CPU (very slow but works)")
    sys.exit(0 if success else 1)