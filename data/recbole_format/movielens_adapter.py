#!/usr/bin/env python3
"""
MovieLens to RecBole Format Adapter

This script converts our processed MovieLens data to RecBole standard format
for official SS4Rec training with proper sequential recommendation evaluation.

CRITICAL: RecBole Data Splitting Strategy
==========================================
RecBole uses a SINGLE .inter file containing ALL user interactions, then performs
internal temporal splitting for sequential recommendation evaluation:

1. LEAVE-ONE-OUT PROTOCOL:
   - Last interaction per user → Test set
   - Second-to-last interaction → Validation set  
   - All earlier interactions → Training set

2. TEMPORAL ORDERING MAINTAINED:
   - All interactions sorted by (user_id, timestamp)
   - Sequential patterns preserved within each user's history
   - No random shuffling - respects temporal dependencies

3. ETL PIPELINE INTEGRATION:
   - For "future" data evaluation: append new interactions to existing .inter file
   - RecBole will automatically use NEW data as test set (most recent timestamps)
   - Existing data becomes training/validation (older timestamps)
   - This enables realistic temporal evaluation of model on unseen future data

RecBole Format Requirements:
- .inter file with columns: user_id:token, item_id:token, rating:float, timestamp:float
- Sequential ordering by user and timestamp
- Leave-one-out evaluation protocol
- Standard RecSys evaluation metrics

Example ETL Integration:
- Original data: interactions from 2020-2023 (20M records)
- Future ETL batch: interactions from 2024 (new unseen data)
- Combined .inter file: RecBole uses 2024 data for testing, 2020-2023 for training
- Result: True temporal evaluation showing model performance on future data
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def load_processed_movielens_data(data_dir: str, include_test: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load our processed MovieLens data
    
    Args:
        data_dir: Path to processed data directory
        include_test: Whether to include test data (False for RecBole training, True for full dataset)
        
    Returns:
        combined_data: Combined train/val data for RecBole (test excluded for future ETL pipeline)
        metadata: Data statistics and mappings
    """
    data_path = Path(data_dir)
    
    # Load existing processed data
    train_data = pd.read_csv(data_path / 'train_data.csv')
    
    # Check for validation data
    val_path = data_path / 'val_data.csv'
    if val_path.exists():
        val_data = pd.read_csv(val_path)
        combined_data = pd.concat([train_data, val_data], ignore_index=True)
        logging.info(f"Combined train ({len(train_data)}) + val ({len(val_data)}) = {len(combined_data)} interactions")
    else:
        logging.warning("No validation data found, using only train data")
        combined_data = train_data.copy()
    
    # Optionally include test data (not recommended for RecBole training - kept for future ETL)
    if include_test:
        test_path = data_path / 'test_data.csv'
        if test_path.exists():
            test_data = pd.read_csv(test_path)
            combined_data = pd.concat([combined_data, test_data], ignore_index=True)
            logging.info(f"Added test data: {len(test_data)} interactions")
        else:
            logging.warning("No test data found")
    else:
        logging.info("🎯 Test data EXCLUDED for RecBole training (reserved for future ETL pipeline)")
    
    # Load metadata if available
    metadata = {}
    metadata_path = data_path / 'metadata.json'
    if metadata_path.exists():
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    logging.info(f"Loaded {len(combined_data)} interactions")
    # Check column names and adapt
    if 'user_idx' in combined_data.columns:
        user_col, item_col = 'user_idx', 'movie_idx'
    else:
        user_col, item_col = 'user_id', 'movie_id'
    
    logging.info(f"Users: {combined_data[user_col].nunique()}")
    logging.info(f"Items: {combined_data[item_col].nunique()}")
    
    # Store column names for later use
    combined_data._user_col = user_col
    combined_data._item_col = item_col
    
    return combined_data, metadata


def convert_to_recbole_format(data: pd.DataFrame, output_dir: str, dataset_name: str = 'movielens'):
    """
    Convert MovieLens data to RecBole format
    
    Args:
        data: DataFrame with columns [user_idx, movie_idx, rating, timestamp, ...]
        output_dir: Output directory for RecBole files
        dataset_name: Name of the dataset
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get column names
    user_col = getattr(data, '_user_col', 'user_id')
    item_col = getattr(data, '_item_col', 'movie_id')
    
    # Create RecBole interaction file - ALWAYS include rating column as per NEXT_STEPS.md schema
    rating_col = 'rating' if 'rating' in data.columns else None
    
    # Ensure rating column exists - required for new schema: user_id, item_id, rating, timestamp
    if rating_col is None:
        logging.error("❌ Rating column is required but not found in data")
        logging.error(f"Available columns: {list(data.columns)}")
        raise ValueError("Rating column is required for the new schema: user_id, item_id, rating, timestamp")
    
    if 'timestamp' in data.columns:
        recbole_data = data[[user_col, item_col, rating_col, 'timestamp']].copy()
    else:
        # Create synthetic timestamps based on rating order
        logging.warning("No timestamp column found, creating synthetic timestamps")
        recbole_data = data[[user_col, item_col, rating_col]].copy()
        recbole_data['timestamp'] = range(len(recbole_data))
    
    # Rename columns to RecBole format - NEW SCHEMA: user_id, item_id, rating, timestamp
    recbole_data.columns = ['user_id', 'item_id', 'rating', 'timestamp']
    
    # Clean data - remove NaN and infinite values before type conversion
    initial_count = len(recbole_data)
    recbole_data = recbole_data.dropna()  # Remove NaN values
    recbole_data = recbole_data.replace([np.inf, -np.inf], np.nan).dropna()  # Remove infinite values
    
    if len(recbole_data) < initial_count:
        logging.warning(f"⚠️ Removed {initial_count - len(recbole_data)} rows with NaN/infinite values")
    
    # Ensure proper data types
    recbole_data['user_id'] = recbole_data['user_id'].astype(int)
    recbole_data['item_id'] = recbole_data['item_id'].astype(int)
    recbole_data['rating'] = recbole_data['rating'].astype(float)
    recbole_data['timestamp'] = recbole_data['timestamp'].astype(float)
    
    # Sort by user and timestamp for proper sequential evaluation
    recbole_data = recbole_data.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
    
    # Filter users with sufficient interactions (minimum 3 for leave-one-out)
    user_counts = recbole_data['user_id'].value_counts()
    valid_users = user_counts[user_counts >= 3].index
    recbole_data = recbole_data[recbole_data['user_id'].isin(valid_users)].reset_index(drop=True)
    
    logging.info(f"After filtering: {len(recbole_data)} interactions")
    logging.info(f"Valid users: {len(valid_users)}")
    
    # Save as .inter file (RecBole format)
    inter_file = output_path / f'{dataset_name}.inter'
    recbole_data.to_csv(inter_file, sep='\t', index=False)
    
    logging.info(f"✅ Saved RecBole interaction file: {inter_file}")
    
    # Create dataset statistics file
    stats = {
        'n_users': recbole_data['user_id'].nunique(),
        'n_items': recbole_data['item_id'].nunique(), 
        'n_interactions': len(recbole_data),
        'sparsity': 1 - (len(recbole_data) / (recbole_data['user_id'].nunique() * recbole_data['item_id'].nunique())),
        'avg_seq_length': recbole_data.groupby('user_id').size().mean(),
        'timestamp_range': [recbole_data['timestamp'].min(), recbole_data['timestamp'].max()]
    }
    
    # Save statistics
    stats_file = output_path / f'{dataset_name}_stats.json'
    import json
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    logging.info(f"✅ Saved dataset statistics: {stats_file}")
    logging.info("📊 Dataset Statistics:")
    for key, value in stats.items():
        if key != 'timestamp_range':
            logging.info(f"  - {key}: {value}")
    
    return inter_file, stats


def create_recbole_config_file(output_dir: str, dataset_name: str, stats: Dict[str, Any]):
    """
    Create RecBole-compatible configuration snippet
    
    Args:
        output_dir: Output directory
        dataset_name: Dataset name  
        stats: Dataset statistics
    """
    config_snippet = f"""
# RecBole Dataset Configuration for {dataset_name}
# Generated automatically from MovieLens processed data

# Dataset Information  
n_users: {stats['n_users']}
n_items: {stats['n_items']}  
n_interactions: {stats['n_interactions']}
sparsity: {stats['sparsity']:.6f}

# Sequential Parameters
MAX_ITEM_LIST_LENGTH: {min(50, int(stats['avg_seq_length'] * 2))}  # Adaptive max length

# Field Types (RecBole format)
field_separator: "\\t"
seq_separator: " "

# Load Configuration - NEW SCHEMA: user_id, item_id, rating, timestamp
load_col:
  inter: [user_id, item_id, rating, timestamp]
  
# Evaluation
eval_args:
  split: {{'LS': 'valid_and_test'}}  # Leave-one-out splitting
  order: 'TO'                        # Time-ordered
"""
    
    config_file = Path(output_dir) / f'{dataset_name}_recbole_config.yaml'
    with open(config_file, 'w') as f:
        f.write(config_snippet)
    
    logging.info(f"✅ Created RecBole config snippet: {config_file}")
    return config_file


def validate_recbole_format(inter_file: str) -> bool:
    """
    Validate RecBole format requirements
    
    Args:
        inter_file: Path to .inter file
        
    Returns:
        is_valid: Whether format is valid
    """
    try:
        # Load and check format
        data = pd.read_csv(inter_file, sep='\t')
        
        # Check required columns - NEW SCHEMA: user_id, item_id, rating, timestamp
        required_cols = ['user_id', 'item_id', 'rating', 'timestamp']
        if not all(col in data.columns for col in required_cols):
            logging.error(f"❌ Missing required columns: {required_cols}")
            logging.error(f"Found columns: {list(data.columns)}")
            return False
        
        # Check data types
        if not pd.api.types.is_integer_dtype(data['user_id']):
            logging.error("❌ user_id must be integer type")
            return False
            
        if not pd.api.types.is_integer_dtype(data['item_id']):
            logging.error("❌ item_id must be integer type") 
            return False
        
        if not pd.api.types.is_numeric_dtype(data['rating']):
            logging.error("❌ rating must be numeric type")
            return False
            
        if not pd.api.types.is_numeric_dtype(data['timestamp']):
            logging.error("❌ timestamp must be numeric type")
            return False
        
        # Check temporal ordering
        user_groups = data.groupby('user_id')
        for user_id, group in user_groups:
            if not group['timestamp'].is_monotonic_increasing:
                logging.warning(f"⚠️ Timestamps not ordered for user {user_id}")
        
        # Check minimum interactions per user
        user_counts = data['user_id'].value_counts()
        min_interactions = user_counts.min()
        if min_interactions < 3:
            logging.warning(f"⚠️ Some users have < 3 interactions (min: {min_interactions})")
        
        logging.info("✅ RecBole format validation passed")
        return True
        
    except Exception as e:
        logging.error(f"❌ RecBole format validation failed: {e}")
        return False


def append_future_data_for_etl(new_data: pd.DataFrame, 
                              existing_inter_file: str,
                              output_file: str = None) -> str:
    """
    Append future data to existing RecBole .inter file for ETL evaluation
    
    This function enables temporal evaluation by appending new "future" interactions
    to the existing training data. RecBole will automatically:
    - Use the newest timestamps (future data) as test set
    - Use older timestamps (existing data) as training/validation
    
    Args:
        new_data: DataFrame with new interactions [user_id, item_id, rating, timestamp]
        existing_inter_file: Path to existing .inter file
        output_file: Output file path (optional, defaults to existing_inter_file)
        
    Returns:
        output_file: Path to updated .inter file
        
    Example Usage for ETL Pipeline:
        # Load new batch data (e.g., from daily ETL)
        new_batch = pd.read_csv('etl_batch_day_5.csv')
        
        # Convert to RecBole format
        new_batch_recbole = new_batch[['user_id', 'item_id', 'rating', 'timestamp']].copy()
        
        # Append to existing training data
        updated_file = append_future_data_for_etl(
            new_batch_recbole, 
            'data/recbole_format/movielens.inter'
        )
        
        # Train SS4Rec on combined data - it will automatically evaluate on new_batch
        # Result: True temporal evaluation showing performance on unseen future data
    """
    logging.info(f"📊 Appending {len(new_data)} new interactions for ETL evaluation")
    
    try:
        # Load existing data
        existing_data = pd.read_csv(existing_inter_file, sep='\t')
        logging.info(f"📄 Loaded {len(existing_data)} existing interactions")
        
        # Ensure consistent column names and types - NEW SCHEMA: user_id, item_id, rating, timestamp
        required_cols = ['user_id', 'item_id', 'rating', 'timestamp']
        if not all(col in new_data.columns for col in required_cols):
            logging.error(f"❌ New data missing required columns: {required_cols}")
            logging.error(f"Found columns: {list(new_data.columns)}")
            raise ValueError(f"New data must have columns: {required_cols}")
            
        new_data_clean = new_data[required_cols].copy()
        new_data_clean['user_id'] = new_data_clean['user_id'].astype(int)
        new_data_clean['item_id'] = new_data_clean['item_id'].astype(int)
        new_data_clean['rating'] = new_data_clean['rating'].astype(float)
        new_data_clean['timestamp'] = new_data_clean['timestamp'].astype(float)
        
        # Combine data (new data will have later timestamps)
        combined_data = pd.concat([existing_data, new_data_clean], ignore_index=True)
        
        # Sort by user and timestamp (critical for RecBole)
        combined_data = combined_data.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
        
        # Save updated file
        if output_file is None:
            output_file = existing_inter_file
        
        combined_data.to_csv(output_file, sep='\t', index=False)
        
        logging.info(f"✅ Updated .inter file: {output_file}")
        logging.info(f"📊 Total interactions: {len(combined_data)}")
        logging.info(f"📈 New interactions: {len(new_data_clean)}")
        logging.info(f"🎯 RecBole will use newest timestamps as test set (your ETL data)")
        
        return output_file
        
    except Exception as e:
        logging.error(f"❌ Failed to append future data: {e}")
        raise


def main():
    """Main conversion function"""
    setup_logging()
    
    # Paths
    data_dir = 'data/processed'
    output_dir = 'data/recbole_format'
    dataset_name = 'movielens'
    
    logging.info("🔄 Converting MovieLens data to RecBole format")
    logging.info(f"📂 Input: {data_dir}")
    logging.info(f"📂 Output: {output_dir}")
    
    try:
        # Load processed data
        logging.info("📊 Loading processed MovieLens data...")
        data, metadata = load_processed_movielens_data(data_dir)
        
        # Convert to RecBole format
        logging.info("🔄 Converting to RecBole format...")
        inter_file, stats = convert_to_recbole_format(data, output_dir, dataset_name)
        
        # Create configuration
        logging.info("⚙️ Creating RecBole configuration...")
        config_file = create_recbole_config_file(output_dir, dataset_name, stats)
        
        # Validate format
        logging.info("✅ Validating RecBole format...")
        is_valid = validate_recbole_format(inter_file)
        
        if is_valid:
            logging.info("🎉 MovieLens to RecBole conversion completed successfully!")
            logging.info(f"📄 Files created:")
            logging.info(f"  - Interactions: {inter_file}")
            logging.info(f"  - Statistics: {Path(output_dir) / f'{dataset_name}_stats.json'}")
            logging.info(f"  - Config: {config_file}")
            
            logging.info("🚀 Ready for official SS4Rec training!")
            
        else:
            logging.error("❌ RecBole format validation failed")
            return 1
            
        return 0
        
    except Exception as e:
        logging.error(f"❌ Conversion failed: {e}")
        logging.error("Full error details:", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)