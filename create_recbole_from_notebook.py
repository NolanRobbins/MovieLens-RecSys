#!/usr/bin/env python3
"""
Create RecBole movielens.inter directly from notebook data processing approach

This script replicates the exact data processing from data_loading.ipynb:
1. Load raw MovieLens-32M data
2. Split by timestamp (80% older data for training)
3. Generate proper RecBole format: userId, movieId, rating, timestamp
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_data(path):
    """Read MovieLens data from raw directory"""
    files = {}
    for filename in os.listdir(path):
        stem, suffix = os.path.splitext(filename)
        file_path = os.path.join(path, filename)
        logging.info(f"Reading {filename}")
        if suffix == '.csv':
            files[stem] = pd.read_csv(file_path)
        elif suffix == '.dat':
            if stem == 'ratings':
                columns = ['userId', 'movieId', 'rating', 'timestamp']
            else:
                columns = ['movieId', 'title', 'genres']
            data = pd.read_csv(file_path, sep='::', names=columns, engine='python')
            files[stem] = data
    return files['ratings'], files['movies']

def create_recbole_inter_from_processed():
    """
    Create RecBole movielens.inter file from processed data
    Combines train_data + val_data (following notebook's 80% split)
    Schema: userId, movieId, rating, timestamp
    """
    
    # Paths
    processed_path = Path('data/processed')
    output_dir = Path('data/recbole_format/movielens')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info("🔄 Loading processed MovieLens data...")
    
    # Load train and val data (which represent the 80% older data from notebook)
    train_data = pd.read_csv(processed_path / 'train_data.csv')
    val_data = pd.read_csv(processed_path / 'val_data.csv')
    
    logging.info(f"📊 Train data: {len(train_data):,} interactions")
    logging.info(f"📊 Val data: {len(val_data):,} interactions")
    
    # Use only the core columns we need for RecBole
    train_core = train_data[['userId', 'movieId', 'rating', 'timestamp']].copy()
    val_core = val_data[['userId', 'movieId', 'rating', 'timestamp']].copy()
    
    # Combine train + val data (this is the 80% from notebook)
    recbole_data = pd.concat([train_core, val_core], ignore_index=True)
    logging.info(f"📈 Combined data for RecBole: {len(recbole_data):,} interactions")
    
    # Sort by userId and timestamp (critical for RecBole sequential models)
    recbole_data = recbole_data.sort_values(['userId', 'timestamp']).reset_index(drop=True)
    
    logging.info(f"📋 RecBole data shape: {recbole_data.shape}")
    logging.info(f"👥 Unique users: {recbole_data['userId'].nunique():,}")
    logging.info(f"🎬 Unique movies: {recbole_data['movieId'].nunique():,}")
    
    # Write RecBole format file with proper header
    inter_file = output_dir / 'movielens.inter'
    logging.info(f"💾 Writing RecBole format to: {inter_file}")
    
    with open(inter_file, 'w') as f:
        # Write proper RecBole header: feature_name:feature_type
        f.write('userId:token\tmovieId:token\trating:float\ttimestamp:float\n')
        
        # Write data rows
        recbole_data.to_csv(f, sep='\t', index=False, header=False)
    
    logging.info(f"✅ RecBole movielens.inter created successfully!")
    logging.info(f"📊 Final stats:")
    logging.info(f"  - Interactions: {len(recbole_data):,}")
    logging.info(f"  - Users: {recbole_data['userId'].nunique():,}")
    logging.info(f"  - Movies: {recbole_data['movieId'].nunique():,}")
    logging.info(f"  - Rating range: [{recbole_data['rating'].min()}, {recbole_data['rating'].max()}]")
    logging.info(f"  - Time range: [{recbole_data['timestamp'].min()}, {recbole_data['timestamp'].max()}]")
    
    return inter_file

if __name__ == "__main__":
    logging.info("🚀 Creating RecBole movielens.inter from notebook data processing...")
    inter_file = create_recbole_inter_from_processed()
    logging.info(f"🎉 Complete! File created: {inter_file}")