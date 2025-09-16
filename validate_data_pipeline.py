#!/usr/bin/env python3
"""
Data Pipeline Validation Script for SS4Rec Training
==================================================

This script validates data format, schema, and RecBole integration
before training starts, preventing data-related failures.

Usage:
    python validate_data_pipeline.py --dataset ml-25m
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def validate_file_exists(file_path: Path) -> Tuple[bool, str]:
    """Check if data file exists and get basic info"""
    if not file_path.exists():
        return False, f"File not found: {file_path}"

    try:
        size_mb = file_path.stat().st_size / (1024 * 1024)
        return True, f"File exists ({size_mb:.1f} MB)"
    except Exception as e:
        return False, f"Error accessing file: {e}"

def validate_recbole_format(file_path: Path) -> Tuple[bool, str, Optional[pd.DataFrame]]:
    """Validate RecBole .inter file format"""
    logger = setup_logging()

    try:
        # Read first few lines to check format
        logger.info(f"📖 Reading data from {file_path}...")
        df = pd.read_csv(file_path, sep='\t', nrows=1000)

        # Check required columns
        required_cols = ['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            return False, f"Missing columns: {missing_cols}", None

        # Check data types and ranges
        issues = []

        # Check user_id range
        if 'user_id:token' in df.columns:
            user_range = f"{df['user_id:token'].min()}-{df['user_id:token'].max()}"
            logger.info(f"📊 User IDs: {user_range} ({df['user_id:token'].nunique()} unique)")

        # Check item_id range
        if 'item_id:token' in df.columns:
            item_range = f"{df['item_id:token'].min()}-{df['item_id:token'].max()}"
            logger.info(f"📊 Item IDs: {item_range} ({df['item_id:token'].nunique()} unique)")

        # Check rating range
        if 'rating:float' in df.columns:
            rating_range = f"{df['rating:float'].min():.1f}-{df['rating:float'].max():.1f}"
            logger.info(f"📊 Ratings: {rating_range}")

        # Check timestamp format
        if 'timestamp:float' in df.columns:
            timestamp_range = f"{df['timestamp:float'].min():.0f}-{df['timestamp:float'].max():.0f}"
            logger.info(f"📊 Timestamps: {timestamp_range}")

            # Check for reasonable timestamp values (should be Unix timestamps)
            min_ts, max_ts = df['timestamp:float'].min(), df['timestamp:float'].max()
            if min_ts < 946684800 or max_ts > 2147483647:  # 2000-01-01 to 2038-01-19
                issues.append(f"Unusual timestamp range: {min_ts}-{max_ts}")

        # Check for missing values
        null_counts = df.isnull().sum()
        if null_counts.any():
            issues.append(f"Missing values: {null_counts[null_counts > 0].to_dict()}")

        if issues:
            return False, f"Data issues: {'; '.join(issues)}", df
        else:
            return True, "RecBole format validation passed", df

    except Exception as e:
        return False, f"Error reading file: {e}", None

def validate_sequential_properties(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validate sequential recommendation properties"""
    logger = setup_logging()

    try:
        # Check if data is sorted by user and timestamp
        logger.info("🔍 Validating sequential properties...")

        # Group by user and check timestamp ordering
        user_groups = df.groupby('user_id:token')

        # Sample a few users to check ordering
        sample_users = df['user_id:token'].unique()[:10]
        ordering_issues = 0

        for user_id in sample_users:
            user_data = df[df['user_id:token'] == user_id].sort_index()
            timestamps = user_data['timestamp:float'].values

            if not np.all(timestamps[:-1] <= timestamps[1:]):
                ordering_issues += 1

        if ordering_issues > 0:
            return False, f"Timestamp ordering issues for {ordering_issues}/{len(sample_users)} sampled users"

        # Check sequence length distribution
        seq_lengths = user_groups.size()
        avg_seq_len = seq_lengths.mean()
        max_seq_len = seq_lengths.max()
        min_seq_len = seq_lengths.min()

        logger.info(f"📏 Sequence lengths: avg={avg_seq_len:.1f}, min={min_seq_len}, max={max_seq_len}")

        # Check for very short sequences (problematic for sequential models)
        short_sequences = (seq_lengths < 5).sum()
        if short_sequences > len(seq_lengths) * 0.5:
            return False, f"Too many short sequences: {short_sequences}/{len(seq_lengths)} users have <5 interactions"

        return True, "Sequential properties validation passed"

    except Exception as e:
        return False, f"Error validating sequential properties: {e}"

def test_recbole_integration(data_path: Path, dataset_name: str) -> Tuple[bool, str]:
    """Test RecBole dataset loading"""
    logger = setup_logging()

    try:
        logger.info("🤖 Testing RecBole integration...")

        # Test minimal RecBole config
        config_dict = {
            'dataset': dataset_name,
            'data_path': str(data_path.parent),
            'USER_ID_FIELD': 'user_id',
            'ITEM_ID_FIELD': 'item_id',
            'RATING_FIELD': 'rating',
            'TIME_FIELD': 'timestamp',
            'load_col': {
                'inter': ['user_id', 'item_id', 'rating', 'timestamp']
            },
            'MAX_ITEM_LIST_LENGTH': 50,
            'download': False,
        }

        # Try to import RecBole components
        from recbole.config import Config
        from recbole.data import create_dataset

        # Create config
        config = Config(config_dict=config_dict)

        # Try to create dataset (this will validate the data format)
        logger.info("📊 Creating RecBole dataset...")
        dataset = create_dataset(config)

        # Basic dataset validation
        if hasattr(dataset, 'inter_feat'):
            inter_count = len(dataset.inter_feat)
            logger.info(f"✅ Dataset created successfully: {inter_count} interactions")

        # Check if dataset has required fields
        required_fields = ['user_id', 'item_id', 'rating', 'timestamp']
        missing_fields = [field for field in required_fields if not hasattr(dataset, f'{field}_field')]

        if missing_fields:
            return False, f"Dataset missing required fields: {missing_fields}"

        return True, "RecBole integration test passed"

    except ImportError as e:
        return False, f"RecBole import failed: {e}"
    except Exception as e:
        return False, f"RecBole integration failed: {e}"

def validate_timestamp_field_config(config_path: Path) -> Tuple[bool, str]:
    """Validate timestamp field configuration in config file"""
    logger = setup_logging()

    try:
        import yaml

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Check required timestamp fields for SS4Rec
        required_fields = ['TIME_FIELD', 'TIMESTAMP_FIELD']
        missing_fields = []

        for field in required_fields:
            if field not in config:
                missing_fields.append(field)

        if missing_fields:
            return False, f"Config missing timestamp fields: {missing_fields}"

        # Check LIST_SUFFIX is present
        if 'LIST_SUFFIX' not in config:
            return False, "Config missing LIST_SUFFIX field"

        logger.info(f"✅ Timestamp config: TIME_FIELD={config.get('TIME_FIELD')}, TIMESTAMP_FIELD={config.get('TIMESTAMP_FIELD')}")

        return True, "Timestamp field configuration validated"

    except Exception as e:
        return False, f"Error validating config: {e}"

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Validate SS4Rec data pipeline')
    parser.add_argument('--dataset', default='ml-1m',
                       help='Dataset name (default: ml-1m for debug)')
    parser.add_argument('--data-path',
                       default='data/recbole_format',
                       help='Path to data directory')
    parser.add_argument('--config',
                       default='configs/official/ss4rec_official.yaml',
                       help='Path to config file')
    args = parser.parse_args()

    logger = setup_logging()
    logger.info("🚀 SS4Rec Data Pipeline Validation")
    logger.info("=" * 50)

    # Construct data file path
    data_dir = Path(args.data_path) / args.dataset
    data_file = data_dir / f"{args.dataset}.inter"
    config_file = Path(args.config)

    validation_results = []

    # 1. Check if data file exists
    logger.info("📁 Checking data file existence...")
    exists, msg = validate_file_exists(data_file)
    validation_results.append(("File Exists", exists, msg))

    if not exists:
        logger.error(f"❌ {msg}")
        logger.error("🚨 Cannot proceed without data file")
        return 1

    # 2. Validate RecBole format
    logger.info("📋 Validating RecBole format...")
    format_ok, msg, df = validate_recbole_format(data_file)
    validation_results.append(("RecBole Format", format_ok, msg))

    if not format_ok:
        logger.error(f"❌ {msg}")
        return 1

    # 3. Validate sequential properties
    logger.info("📈 Validating sequential properties...")
    seq_ok, msg = validate_sequential_properties(df)
    validation_results.append(("Sequential Properties", seq_ok, msg))

    # 4. Test RecBole integration
    logger.info("🔗 Testing RecBole integration...")
    recbole_ok, msg = test_recbole_integration(data_file, args.dataset)
    validation_results.append(("RecBole Integration", recbole_ok, msg))

    # 5. Validate config file
    logger.info("⚙️ Validating config file...")
    config_ok, msg = validate_timestamp_field_config(config_file)
    validation_results.append(("Config Validation", config_ok, msg))

    # Summary
    logger.info("=" * 50)
    logger.info("📊 DATA PIPELINE VALIDATION SUMMARY")
    logger.info("=" * 50)

    all_passed = True
    for test_name, passed, message in validation_results:
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status} {test_name}: {message}")
        if not passed:
            all_passed = False

    if all_passed:
        logger.info("🎉 ALL DATA VALIDATIONS PASSED - Ready for training!")
        return 0
    else:
        logger.error("🚨 DATA VALIDATION FAILED - Fix issues before training")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)