#!/usr/bin/env python3
"""
Create RecBole format data from fresh ML-32M dataset
Following NEXT_STEPS.md Step 1: Generate missing data files

This script:
1. Loads fresh ML-32M data from data/raw/ml-32m/
2. Creates main movielens.inter file with proper RecBole format
3. Splits into movielens_past.inter (80%) and movielens_future.inter (20%)
4. Follows the exact schema: user_id:token, item_id:token, rating:float, timestamp:float
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

def main():
    print("🚀 Creating RecBole format data from fresh ML-32M dataset...")
    
    # Check if raw data exists
    raw_data_dir = Path("data/raw/ml-32m")
    if not raw_data_dir.exists():
        print(f"❌ Raw data directory not found: {raw_data_dir}")
        return False
    
    print(f"✅ Found raw data directory: {raw_data_dir}")
    
    # Load the ratings data
    ratings_file = raw_data_dir / "ratings.csv"
    if not ratings_file.exists():
        print(f"❌ Ratings file not found: {ratings_file}")
        return False
    
    print(f"📊 Loading ratings from: {ratings_file}")
    ratings = pd.read_csv(ratings_file)
    
    print(f"✅ Loaded {len(ratings):,} ratings")
    print(f"📋 Columns: {list(ratings.columns)}")
    print("📋 Sample data:")
    print(ratings.head())
    
    # Verify the data format
    expected_columns = ['userId', 'movieId', 'rating', 'timestamp']
    if not all(col in ratings.columns for col in expected_columns):
        print(f"❌ Expected columns {expected_columns} not found")
        print(f"Found columns: {list(ratings.columns)}")
        return False
    
    print("✅ Data format verified")
    
    # Create user and movie mappings (0-indexed for RecBole)
    print("🔄 Creating user and movie mappings...")
    unique_users = sorted(ratings['userId'].unique())
    unique_movies = sorted(ratings['movieId'].unique())
    
    user_to_index = {old: new for new, old in enumerate(unique_users)}
    movie_to_index = {old: new for new, old in enumerate(unique_movies)}
    
    print(f"📊 Users: {len(unique_users):,}, Movies: {len(unique_movies):,}")
    
    # Convert to indexed format
    ratings['user_idx'] = ratings['userId'].map(user_to_index)
    ratings['movie_idx'] = ratings['movieId'].map(movie_to_index)
    
    # Create RecBole format data
    print("🔄 Creating RecBole format data...")
    recbole_data = ratings[['user_idx', 'movie_idx', 'rating', 'timestamp']].copy()
    recbole_data.columns = ['user_id', 'item_id', 'rating', 'timestamp']
    
    # Sort by user and timestamp (critical for RecBole sequential recommendation)
    recbole_data = recbole_data.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
    
    print(f"✅ RecBole data prepared: {len(recbole_data):,} interactions")
    
    # Create RecBole directory structure
    recbole_dir = Path("data/recbole_format/movielens")
    recbole_dir.mkdir(parents=True, exist_ok=True)
    
    # Create the main movielens.inter file
    print("📝 Creating main movielens.inter file...")
    inter_file = recbole_dir / "movielens.inter"
    
    with open(inter_file, 'w') as f:
        # Write RecBole header with proper format
        f.write('user_id:token\titem_id:token\trating:float\ttimestamp:float\n')
        # Write data without header
        recbole_data.to_csv(f, sep='\t', index=False, header=False)
    
    print(f"✅ Created {inter_file}: {len(recbole_data):,} interactions")
    
    # Now split chronologically: 80% past / 20% future
    print("🔄 Splitting data chronologically (80% past / 20% future)...")
    
    # Calculate 80th percentile timestamp for chronological split
    split_timestamp = recbole_data['timestamp'].quantile(0.8)
    print(f"📅 Split timestamp (80th percentile): {split_timestamp}")
    
    # Split chronologically
    past_mask = recbole_data['timestamp'] <= split_timestamp
    future_mask = recbole_data['timestamp'] > split_timestamp
    
    past_data = recbole_data[past_mask].copy()
    future_data = recbole_data[future_mask].copy()
    
    print(f"📊 Past data (training): {len(past_data):,} interactions ({len(past_data)/len(recbole_data)*100:.1f}%)")
    print(f"📊 Future data (ETL pipeline): {len(future_data):,} interactions ({len(future_data)/len(recbole_data)*100:.1f}%)")
    
    # Create processed directory
    processed_dir = Path("data/processed")
    processed_dir.mkdir(exist_ok=True)
    
    # Save chronologically split files
    print("📝 Creating chronologically split .inter files...")
    
    past_file = processed_dir / "movielens_past.inter"
    future_file = processed_dir / "movielens_future.inter"
    
    # Write past data file
    with open(past_file, 'w') as f:
        f.write('user_id:token\titem_id:token\trating:float\ttimestamp:float\n')
        past_data.to_csv(f, sep='\t', index=False, header=False)
    
    # Write future data file
    with open(future_file, 'w') as f:
        f.write('user_id:token\titem_id:token\trating:float\ttimestamp:float\n')
        future_data.to_csv(f, sep='\t', index=False, header=False)
    
    print(f"✅ Created {past_file}: {len(past_data):,} interactions")
    print(f"✅ Created {future_file}: {len(future_data):,} interactions")
    
    # Save mappings for reference
    import pickle
    mappings = {
        'user_to_index': user_to_index,
        'movie_to_index': movie_to_index,
        'n_users': len(unique_users),
        'n_movies': len(unique_movies),
        'split_timestamp': split_timestamp
    }
    
    mappings_file = processed_dir / "data_mappings.pkl"
    with open(mappings_file, 'wb') as f:
        pickle.dump(mappings, f)
    
    print(f"✅ Saved mappings to {mappings_file}")
    
    # Verify files were created
    print("\n🔍 File verification:")
    files_to_check = [
        inter_file,
        past_file,
        future_file
    ]
    
    for filepath in files_to_check:
        if filepath.exists():
            file_size = filepath.stat().st_size / (1024*1024)  # MB
            print(f"✅ {filepath}: {file_size:.1f} MB")
        else:
            print(f"❌ Error: {filepath} was not created")
    
    # Show sample data
    print(f"\n📋 Sample of past data (for training):")
    print(past_data.head())
    print(f"\n📋 Sample of future data (for ETL):")
    print(future_data.head())
    
    print("\n🎯 STEP 1 COMPLETE: RecBole format data created!")
    print("✅ movielens.inter - Complete dataset in RecBole format")
    print("✅ movielens_past.inter - 80% chronological past data (for SS4Rec training)")
    print("✅ movielens_future.inter - 20% chronological future data (for ETL pipeline)")
    print("✅ All files have correct schema: user_id:token, item_id:token, rating:float, timestamp:float")
    
    return True

if __name__ == "__main__":
    main()