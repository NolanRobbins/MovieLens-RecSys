#!/usr/bin/env python3
"""
Dependency Validation Script for SS4Rec Training
===============================================

This script validates all required dependencies before training starts,
preventing runtime failures due to missing or incompatible packages.

Usage:
    python validate_dependencies.py [--fix-missing]
"""

import sys
import importlib
import subprocess
from typing import List, Tuple, Dict
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def check_dependency(package_name: str, import_name: str = None) -> Tuple[bool, str]:
    """Check if a dependency is available and working"""
    if import_name is None:
        import_name = package_name

    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except ImportError as e:
        return False, str(e)

def validate_core_dependencies() -> Dict[str, Tuple[bool, str]]:
    """Validate core ML dependencies"""
    logger = setup_logging()

    core_deps = [
        ('torch', 'torch'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('scipy', 'scipy'),
        ('sklearn', 'sklearn'),
    ]

    results = {}
    logger.info("🔍 Validating core dependencies...")

    for package, import_name in core_deps:
        available, version_or_error = check_dependency(package, import_name)
        results[package] = (available, version_or_error)

        if available:
            logger.info(f"✅ {package}: {version_or_error}")
        else:
            logger.error(f"❌ {package}: {version_or_error}")

    return results

def validate_recbole_dependencies() -> Dict[str, Tuple[bool, str]]:
    """Validate RecBole framework dependencies"""
    logger = setup_logging()

    recbole_deps = [
        ('recbole', 'recbole'),
        ('ray', 'ray'),
        ('hyperopt', 'hyperopt'),
        ('kmeans_pytorch', 'kmeans_pytorch'),
        ('lightgbm', 'lightgbm'),
        ('xgboost', 'xgboost'),
    ]

    results = {}
    logger.info("🤖 Validating RecBole dependencies...")

    for package, import_name in recbole_deps:
        available, version_or_error = check_dependency(package, import_name)
        results[package] = (available, version_or_error)

        if available:
            logger.info(f"✅ {package}: {version_or_error}")
        else:
            logger.error(f"❌ {package}: {version_or_error}")

    return results

def validate_ssm_dependencies() -> Dict[str, Tuple[bool, str]]:
    """Validate State Space Model dependencies"""
    logger = setup_logging()

    ssm_deps = [
        ('causal_conv1d', 'causal_conv1d'),
        ('mamba_ssm', 'mamba_ssm'),
        ('s5', 's5'),
    ]

    results = {}
    logger.info("🌊 Validating State Space Model dependencies...")

    for package, import_name in ssm_deps:
        available, version_or_error = check_dependency(package, import_name)
        results[package] = (available, version_or_error)

        if available:
            logger.info(f"✅ {package}: {version_or_error}")
        else:
            logger.error(f"❌ {package}: {version_or_error}")

    return results

def validate_cuda_support() -> Dict[str, any]:
    """Validate CUDA support and GPU availability"""
    logger = setup_logging()
    logger.info("🔥 Validating CUDA support...")

    results = {}

    try:
        import torch
        results['torch_available'] = True
        results['cuda_available'] = torch.cuda.is_available()
        results['cuda_version'] = torch.version.cuda if torch.cuda.is_available() else None
        results['gpu_count'] = torch.cuda.device_count() if torch.cuda.is_available() else 0

        if torch.cuda.is_available():
            results['gpu_name'] = torch.cuda.get_device_name(0)
            results['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory // (1024**3)  # GB
            logger.info(f"✅ CUDA: {results['cuda_version']}")
            logger.info(f"✅ GPU: {results['gpu_name']} ({results['gpu_memory']}GB)")
        else:
            logger.warning("⚠️ CUDA not available - training will use CPU (very slow)")

    except ImportError:
        results['torch_available'] = False
        logger.error("❌ PyTorch not available")

    return results

def test_model_imports() -> Dict[str, bool]:
    """Test critical model imports"""
    logger = setup_logging()
    logger.info("📦 Testing model imports...")

    results = {}

    # Test SS4Rec model import
    try:
        from models.official_ss4rec.ss4rec_official import SS4Rec
        results['ss4rec_model'] = True
        logger.info("✅ SS4Rec model import successful")
    except ImportError as e:
        results['ss4rec_model'] = False
        logger.error(f"❌ SS4Rec model import failed: {e}")

    # Test RecBole components
    try:
        from recbole.quick_start import run_recbole
        from recbole.config import Config
        from recbole.utils import init_seed, init_logger
        results['recbole_components'] = True
        logger.info("✅ RecBole components import successful")
    except ImportError as e:
        results['recbole_components'] = False
        logger.error(f"❌ RecBole components import failed: {e}")

    # Test SSM components
    try:
        from mamba_ssm import Mamba
        from s5 import S5
        results['ssm_components'] = True
        logger.info("✅ SSM components (Mamba + S5) import successful")
    except ImportError as e:
        results['ssm_components'] = False
        logger.error(f"❌ SSM components import failed: {e}")

    return results

def install_missing_dependency(package: str) -> bool:
    """Attempt to install missing dependency"""
    logger = setup_logging()

    try:
        logger.info(f"📥 Installing {package}...")
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', package
        ], check=True, capture_output=True)
        logger.info(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install {package}: {e}")
        return False

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Validate SS4Rec dependencies')
    parser.add_argument('--fix-missing', action='store_true',
                       help='Attempt to install missing dependencies')
    args = parser.parse_args()

    logger = setup_logging()
    logger.info("🚀 SS4Rec Dependency Validation")
    logger.info("=" * 50)

    # Run all validations
    core_results = validate_core_dependencies()
    recbole_results = validate_recbole_dependencies()
    ssm_results = validate_ssm_dependencies()
    cuda_results = validate_cuda_support()
    import_results = test_model_imports()

    # Count failures
    all_deps = {**core_results, **recbole_results, **ssm_results}
    failed_deps = [pkg for pkg, (available, _) in all_deps.items() if not available]
    failed_imports = [comp for comp, success in import_results.items() if not success]

    logger.info("=" * 50)
    logger.info("📊 VALIDATION SUMMARY")
    logger.info("=" * 50)

    if not failed_deps and not failed_imports and cuda_results.get('cuda_available', False):
        logger.info("🎉 ALL VALIDATIONS PASSED - Ready for training!")
        return 0

    if failed_deps:
        logger.error(f"❌ Missing dependencies: {', '.join(failed_deps)}")

        if args.fix_missing:
            logger.info("🔧 Attempting to fix missing dependencies...")
            for dep in failed_deps:
                install_missing_dependency(dep)

    if failed_imports:
        logger.error(f"❌ Failed imports: {', '.join(failed_imports)}")

    if not cuda_results.get('cuda_available', False):
        logger.warning("⚠️ CUDA not available - training will be very slow")

    if failed_deps or failed_imports:
        logger.error("🚨 VALIDATION FAILED - Fix issues before training")
        return 1
    else:
        logger.info("✅ Dependencies OK - Ready for training (CPU mode)")
        return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)