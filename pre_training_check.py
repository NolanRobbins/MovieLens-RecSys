#!/usr/bin/env python3
"""
Complete Pre-Training Validation Check for SS4Rec
================================================

This script runs all validation checks before training to ensure
maximum probability of successful training completion.

Usage:
    python pre_training_check.py [--fix-issues] [--dataset ml-25m]
"""

import sys
import subprocess
import logging
import argparse
from pathlib import Path

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def run_validation_script(script_name: str, args: list = None, fix_issues: bool = False) -> bool:
    """Run a validation script and return success status"""
    logger = setup_logging()

    cmd = ['python', script_name]
    if args:
        cmd.extend(args)
    if fix_issues and script_name == 'validate_dependencies.py':
        cmd.append('--fix-missing')

    try:
        logger.info(f"🔍 Running {script_name}...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            logger.info(f"✅ {script_name} passed")
            if result.stdout:
                # Show key success messages
                for line in result.stdout.split('\n'):
                    if '✅' in line or 'PASS' in line or 'Ready for training' in line:
                        logger.info(f"  {line.strip()}")
            return True
        else:
            logger.error(f"❌ {script_name} failed")
            if result.stderr:
                for line in result.stderr.split('\n'):
                    if line.strip():
                        logger.error(f"  {line.strip()}")
            if result.stdout:
                for line in result.stdout.split('\n'):
                    if '❌' in line or 'FAIL' in line or 'ERROR' in line:
                        logger.error(f"  {line.strip()}")
            return False

    except FileNotFoundError:
        logger.error(f"❌ {script_name} not found")
        return False
    except Exception as e:
        logger.error(f"❌ Error running {script_name}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Complete pre-training validation check')
    parser.add_argument('--fix-issues', action='store_true',
                       help='Attempt to fix issues automatically')
    parser.add_argument('--dataset', default='ml-1m',
                       help='Dataset name to validate (ml-1m for debug)')
    parser.add_argument('--config', default='configs/official/ss4rec_official.yaml',
                       help='Config file to validate')
    args = parser.parse_args()

    logger = setup_logging()

    logger.info("🚀 SS4Rec Pre-Training Validation Suite")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Fix Issues: {args.fix_issues}")
    logger.info("=" * 60)

    validation_results = []

    # 1. Dependency Validation
    logger.info("📦 PHASE 1: Dependency Validation")
    logger.info("-" * 40)
    deps_ok = run_validation_script('validate_dependencies.py', fix_issues=args.fix_issues)
    validation_results.append(("Dependencies", deps_ok))

    # 2. Data Pipeline Validation
    logger.info("\n📊 PHASE 2: Data Pipeline Validation")
    logger.info("-" * 40)
    data_ok = run_validation_script('validate_data_pipeline.py', [
        '--dataset', args.dataset,
        '--config', args.config
    ])
    validation_results.append(("Data Pipeline", data_ok))

    # 3. Model Import Test
    logger.info("\n🤖 PHASE 3: Model Import Test")
    logger.info("-" * 40)
    try:
        logger.info("Testing SS4Rec model import...")
        # Test basic imports without full training setup
        exec("""
import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))

# Test SS4Rec import
from models.official_ss4rec.ss4rec_official import SS4Rec
print("✅ SS4Rec model import successful")

# Test RecBole imports
from recbole.config import Config
from recbole.quick_start import run_recbole
print("✅ RecBole imports successful")

# Test SSM imports
from mamba_ssm import Mamba
from s5 import S5
print("✅ SSM library imports successful")
""")
        model_ok = True
        logger.info("✅ Model imports successful")
    except Exception as e:
        logger.error(f"❌ Model import failed: {e}")
        model_ok = False

    validation_results.append(("Model Imports", model_ok))

    # 4. Configuration Validation
    logger.info("\n⚙️ PHASE 4: Configuration Validation")
    logger.info("-" * 40)
    config_ok = True
    try:
        import yaml
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"❌ Config file not found: {config_path}")
            config_ok = False
        else:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Check required SS4Rec fields
            required_fields = [
                'model', 'dataset', 'hidden_size', 'num_layers',
                'd_P', 'd_H', 'model_type', 'TIMESTAMP_FIELD'
            ]

            missing_fields = [field for field in required_fields if field not in config]
            if missing_fields:
                logger.error(f"❌ Config missing required fields: {missing_fields}")
                config_ok = False
            else:
                logger.info(f"✅ Config validation passed")
                logger.info(f"  Model: {config.get('model')}")
                logger.info(f"  Dataset: {config.get('dataset')}")
                logger.info(f"  Hidden Size: {config.get('hidden_size')}")

    except Exception as e:
        logger.error(f"❌ Config validation failed: {e}")
        config_ok = False

    validation_results.append(("Configuration", config_ok))

    # Final Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 FINAL VALIDATION SUMMARY")
    logger.info("=" * 60)

    all_passed = True
    for phase, passed in validation_results:
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status} {phase}")
        if not passed:
            all_passed = False

    logger.info("-" * 60)

    if all_passed:
        logger.info("🎉 ALL VALIDATIONS PASSED!")
        logger.info("🚀 Ready for SS4Rec training!")
        logger.info("")
        logger.info("📋 Next Steps:")
        logger.info("  1. Local training: python training/official/train_ss4rec_official.py --config configs/official/ss4rec_official.yaml")
        logger.info("  2. RunPod training: ./runpod_entrypoint.sh --model ss4rec-official")
        logger.info("  3. Google Cloud: python gcp_training.py")
        return 0
    else:
        logger.error("🚨 VALIDATION FAILED!")
        logger.error("❌ Fix the above issues before training")
        logger.error("")
        logger.error("🔧 Suggested fixes:")

        for phase, passed in validation_results:
            if not passed:
                if phase == "Dependencies":
                    logger.error("  - Run: python validate_dependencies.py --fix-missing")
                    logger.error("  - Or: uv pip install -r requirements_ss4rec.txt")
                elif phase == "Data Pipeline":
                    logger.error("  - Check data file exists: data/recbole_format/ml-25m/ml-25m.inter")
                    logger.error("  - Verify RecBole format with correct column headers")
                elif phase == "Model Imports":
                    logger.error("  - Install missing SSM libraries: uv pip install mamba-ssm s5-pytorch")
                elif phase == "Configuration":
                    logger.error("  - Update config file with required SS4Rec fields")

        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)