#!/usr/bin/env python3
"""
Simple script to create RecBole format data from train+val CSV files
Uses only built-in Python modules to avoid dependency issues
"""

import csv
import json
import os
from pathlib import Path


def combine_train_val_data():
    """Combine train and val CSV files into RecBole format"""
    
    # Paths
    data_dir = Path('data/processed')
    output_dir = Path('data/recbole_format')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_file = data_dir / 'train_data.csv'
    val_file = data_dir / 'val_data.csv'
    output_file = output_dir / 'movielens.inter'
    
    print(f"📊 Creating RecBole format data...")
    print(f"📂 Train: {train_file}")
    print(f"📂 Val: {val_file}")
    print(f"📂 Output: {output_file}")
    
    # Check if input files exist
    if not train_file.exists():
        print(f"❌ Train file not found: {train_file}")
        return False
    if not val_file.exists():
        print(f"❌ Val file not found: {val_file}")
        return False
    
    interactions = []
    user_count = set()
    item_count = set()
    
    # Read train data
    with open(train_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Handle different column name formats
            if 'user_idx' in row:
                user_id, item_id = row['user_idx'], row['movie_idx']
            else:
                user_id, item_id = row['user_id'], row['movie_id']
            
            timestamp = row.get('timestamp', len(interactions))  # Use index as timestamp if missing
            
            interactions.append({
                'user_id': int(user_id),
                'item_id': int(item_id), 
                'timestamp': float(timestamp)
            })
            user_count.add(int(user_id))
            item_count.add(int(item_id))
    
    print(f"✅ Loaded {len(interactions)} train interactions")
    
    # Read val data
    with open(val_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Handle different column name formats
            if 'user_idx' in row:
                user_id, item_id = row['user_idx'], row['movie_idx']
            else:
                user_id, item_id = row['user_id'], row['movie_id']
                
            timestamp = row.get('timestamp', len(interactions))  # Use index as timestamp if missing
            
            interactions.append({
                'user_id': int(user_id),
                'item_id': int(item_id),
                'timestamp': float(timestamp)
            })
            user_count.add(int(user_id))
            item_count.add(int(item_id))
    
    print(f"✅ Total interactions: {len(interactions)}")
    print(f"📊 Users: {len(user_count)}")
    print(f"📊 Items: {len(item_count)}")
    
    # Sort by user_id and timestamp (critical for RecBole)
    interactions.sort(key=lambda x: (x['user_id'], x['timestamp']))
    
    # Filter users with at least 3 interactions (required for leave-one-out)
    user_interaction_counts = {}
    for interaction in interactions:
        user_id = interaction['user_id']
        user_interaction_counts[user_id] = user_interaction_counts.get(user_id, 0) + 1
    
    valid_users = {uid for uid, count in user_interaction_counts.items() if count >= 3}
    filtered_interactions = [
        interaction for interaction in interactions 
        if interaction['user_id'] in valid_users
    ]
    
    print(f"📊 After filtering (≥3 interactions): {len(filtered_interactions)}")
    print(f"📊 Valid users: {len(valid_users)}")
    
    # Write RecBole format file (tab-separated)
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        
        # Write header
        writer.writerow(['user_id', 'item_id', 'timestamp'])
        
        # Write data
        for interaction in filtered_interactions:
            writer.writerow([
                interaction['user_id'],
                interaction['item_id'], 
                interaction['timestamp']
            ])
    
    print(f"✅ Created RecBole format file: {output_file}")
    
    # Create statistics file
    final_user_count = len(set(i['user_id'] for i in filtered_interactions))
    final_item_count = len(set(i['item_id'] for i in filtered_interactions))
    
    stats = {
        'n_users': final_user_count,
        'n_items': final_item_count,
        'n_interactions': len(filtered_interactions),
        'sparsity': 1 - (len(filtered_interactions) / (final_user_count * final_item_count)),
        'file_size_mb': os.path.getsize(output_file) / (1024 * 1024)
    }
    
    stats_file = output_dir / 'movielens_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✅ Created statistics file: {stats_file}")
    print(f"📊 Final statistics:")
    for key, value in stats.items():
        if key == 'sparsity':
            print(f"  - {key}: {value:.6f}")
        elif key == 'file_size_mb':
            print(f"  - {key}: {value:.2f} MB")
        else:
            print(f"  - {key}: {value:,}")
    
    # Create simple RecBole config
    config_file = output_dir / 'movielens_recbole_config.yaml'
    config_content = f"""# RecBole Configuration for MovieLens
# Train + Val data combined (test data excluded for future ETL pipeline)

dataset: movielens
model: SS4RecOfficial

# Dataset info
n_users: {stats['n_users']}
n_items: {stats['n_items']}

# Data loading
load_col:
  inter: [user_id, item_id, timestamp]

# Sequential parameters  
MAX_ITEM_LIST_LENGTH: 50

# Evaluation (leave-one-out)
eval_args:
  split: {{'LS': 'valid_and_test'}}
  order: 'TO'

field_separator: "\\t"
"""
    
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print(f"✅ Created config file: {config_file}")
    
    print("\n🎉 RecBole format data ready for upload!")
    print(f"📄 Upload this file to Google Drive: {output_file}")
    print(f"📊 File size: {stats['file_size_mb']:.2f} MB")
    
    return True


if __name__ == "__main__":
    success = combine_train_val_data()
    if not success:
        exit(1)