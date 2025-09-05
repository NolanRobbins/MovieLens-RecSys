#!/usr/bin/env python3
"""
Temporal Analysis Script for ML-32M vs ML-20M
==============================================

This script analyzes the difference between ML-32M and ML-20M datasets
to create a realistic temporal split for testing future user behavior prediction.

The strategy:
1. ML-20M (1995-2015) = Past data for training
2. ML-32M - ML-20M (2015-2023) = Future data for testing
3. This creates an 8-year temporal gap for realistic evaluation

Usage:
    python temporal_analysis.py --ml20m-path data/ml-20m --ml32m-path data/ml-32m --output-dir data/temporal_split
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        handlers=[
            logging.FileHandler('logs/temporal_analysis.log', mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def load_movielens_data(file_path, dataset_name):
    """Load MovieLens dataset and return interactions DataFrame"""
    logger = logging.getLogger(__name__)
    
    try:
        # Try different possible file names
        possible_files = [
            f"{file_path}/ratings.csv",
            f"{file_path}/ratings.dat",
            f"{file_path}/ml-{dataset_name}.csv",
            f"{file_path}/ml-{dataset_name}.dat"
        ]
        
        data_file = None
        for file in possible_files:
            if os.path.exists(file):
                data_file = file
                break
        
        if not data_file:
            raise FileNotFoundError(f"No ratings file found in {file_path}")
        
        logger.info(f"Loading {dataset_name} from: {data_file}")
        
        # Load data based on file extension
        if data_file.endswith('.csv'):
            df = pd.read_csv(data_file)
        else:  # .dat file
            df = pd.read_csv(data_file, sep='::', header=None, 
                           names=['user_id', 'item_id', 'rating', 'timestamp'])
        
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df['year'] = df['datetime'].dt.year
        
        logger.info(f"Loaded {dataset_name}: {len(df)} interactions")
        logger.info(f"  Users: {df['user_id'].nunique()}")
        logger.info(f"  Items: {df['item_id'].nunique()}")
        logger.info(f"  Year range: {df['year'].min()} - {df['year'].max()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to load {dataset_name}: {e}")
        raise

def analyze_temporal_split(ml20m_df, ml32m_df):
    """Analyze the temporal split between ML-20M and ML-32M"""
    logger = logging.getLogger(__name__)
    
    logger.info("🔍 Analyzing temporal split between ML-20M and ML-32M...")
    
    # Find common users and items
    ml20m_users = set(ml20m_df['user_id'].unique())
    ml32m_users = set(ml32m_df['user_id'].unique())
    common_users = ml20m_users.intersection(ml32m_users)
    
    ml20m_items = set(ml20m_df['item_id'].unique())
    ml32m_items = set(ml32m_df['item_id'].unique())
    common_items = ml20m_items.intersection(ml32m_items)
    
    logger.info(f"Common users: {len(common_users)}")
    logger.info(f"Common items: {len(common_items)}")
    
    # Filter to common users and items for fair comparison
    ml20m_filtered = ml20m_df[
        (ml20m_df['user_id'].isin(common_users)) & 
        (ml20m_df['item_id'].isin(common_items))
    ].copy()
    
    ml32m_filtered = ml32m_df[
        (ml32m_df['user_id'].isin(common_users)) & 
        (ml32m_df['item_id'].isin(common_items))
    ].copy()
    
    # Find future interactions (in ML-32M but not in ML-20M)
    ml20m_interactions = set(
        zip(ml20m_filtered['user_id'], ml20m_filtered['item_id'], ml20m_filtered['rating'])
    )
    
    future_interactions = []
    for _, row in ml32m_filtered.iterrows():
        interaction = (row['user_id'], row['item_id'], row['rating'])
        if interaction not in ml20m_interactions:
            future_interactions.append(row)
    
    future_df = pd.DataFrame(future_interactions)
    
    logger.info(f"Future interactions (ML-32M - ML-20M): {len(future_df)}")
    
    if len(future_df) > 0:
        logger.info(f"  Future year range: {future_df['year'].min()} - {future_df['year'].max()}")
        logger.info(f"  Future users: {future_df['user_id'].nunique()}")
        logger.info(f"  Future items: {future_df['item_id'].nunique()}")
    
    return {
        'ml20m_filtered': ml20m_filtered,
        'ml32m_filtered': ml32m_filtered,
        'future_interactions': future_df,
        'common_users': common_users,
        'common_items': common_items
    }

def create_recbole_format(data_df, output_path, dataset_name):
    """Convert DataFrame to RecBole .inter format"""
    logger = logging.getLogger(__name__)
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Convert to RecBole format
    recbole_df = data_df[['user_id', 'item_id', 'rating', 'timestamp']].copy()
    
    # Save as .inter file
    inter_file = os.path.join(output_path, f"{dataset_name}.inter")
    recbole_df.to_csv(inter_file, sep='\t', index=False, header=False)
    
    logger.info(f"Saved RecBole format: {inter_file}")
    logger.info(f"  Size: {os.path.getsize(inter_file) / 1024 / 1024:.1f} MB")
    
    return inter_file

def create_temporal_config(analysis_results, output_dir):
    """Create RecBole configuration for temporal split"""
    logger = logging.getLogger(__name__)
    
    config = {
        'model': 'SS4RecOfficial',
        'dataset': 'temporal_split',
        'data_path': str(output_dir),
        
        # Data configuration
        'USER_ID_FIELD': 'user_id',
        'ITEM_ID_FIELD': 'item_id',
        'RATING_FIELD': 'rating',
        'TIME_FIELD': 'timestamp',
        
        # Temporal split configuration
        'temporal_split': True,
        'past_data': 'ml20m_past.inter',
        'future_data': 'ml32m_future.inter',
        
        # Model configuration
        'hidden_size': 128,
        'n_layers': 3,
        'dropout_prob': 0.3,
        'loss_type': 'BPR',
        
        # Training configuration
        'learning_rate': 0.0005,
        'train_batch_size': 4096,
        'eval_batch_size': 4096,
        'epochs': 200,
        'stopping_step': 10,
        
        # Evaluation configuration
        'metrics': ['Recall', 'MRR', 'NDCG', 'Hit'],
        'topk': [1, 5, 10, 20],
        'valid_metric': 'NDCG@10',
        
        # Device configuration
        'device': 'cuda',
        'gpu_id': 0,
        'reproducibility': True,
        'seed': 2023,
        
        # Distributed training
        'nproc': 1,
        'world_size': 1,
        'offset': 0,
        'ip': 'localhost',
        'port': 29500,
        'backend': 'gloo'
    }
    
    config_file = os.path.join(output_dir, 'temporal_split_config.yaml')
    with open(config_file, 'w') as f:
        import yaml
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Created temporal split config: {config_file}")
    return config_file

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Temporal Analysis for ML-32M vs ML-20M')
    parser.add_argument('--ml20m-path', required=True, help='Path to ML-20M dataset')
    parser.add_argument('--ml32m-path', required=True, help='Path to ML-32M dataset')
    parser.add_argument('--output-dir', default='data/temporal_split', help='Output directory')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("🔬 Starting Temporal Analysis: ML-32M vs ML-20M")
    logger.info("=" * 60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    try:
        # Load datasets
        logger.info("📊 Loading ML-20M dataset...")
        ml20m_df = load_movielens_data(args.ml20m_path, '20m')
        
        logger.info("📊 Loading ML-32M dataset...")
        ml32m_df = load_movielens_data(args.ml32m_path, '32m')
        
        # Analyze temporal split
        analysis_results = analyze_temporal_split(ml20m_df, ml32m_df)
        
        # Create RecBole format files
        logger.info("🔄 Creating RecBole format files...")
        
        past_file = create_recbole_format(
            analysis_results['ml20m_filtered'], 
            args.output_dir, 
            'ml20m_past'
        )
        
        future_file = create_recbole_format(
            analysis_results['future_interactions'], 
            args.output_dir, 
            'ml32m_future'
        )
        
        # Create configuration
        config_file = create_temporal_config(analysis_results, args.output_dir)
        
        # Save analysis summary
        summary = {
            'analysis_date': datetime.now().isoformat(),
            'ml20m_interactions': len(analysis_results['ml20m_filtered']),
            'ml32m_interactions': len(analysis_results['ml32m_filtered']),
            'future_interactions': len(analysis_results['future_interactions']),
            'common_users': len(analysis_results['common_users']),
            'common_items': len(analysis_results['common_items']),
            'temporal_gap_years': 8,
            'past_data_file': past_file,
            'future_data_file': future_file,
            'config_file': config_file
        }
        
        summary_file = os.path.join(args.output_dir, 'temporal_analysis_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("✅ Temporal analysis completed successfully!")
        logger.info(f"📁 Output directory: {args.output_dir}")
        logger.info(f"📄 Summary: {summary_file}")
        logger.info(f"⚙️  Config: {config_file}")
        logger.info(f"📊 Past data: {past_file}")
        logger.info(f"🔮 Future data: {future_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Temporal analysis failed: {e}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    import traceback
    exit(main())
