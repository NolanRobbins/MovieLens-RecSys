#!/usr/bin/env python3
"""
Generate missing data files for SS4Rec training
Step 1: Create movielens_past.inter and movielens_future.inter files
"""

import pandas as pd
import os
import numpy as np

def main():
    print('🚀 Generating missing data files for SS4Rec training...')

    # Check if the complete dataset exists
    inter_file = 'data/recbole_format/movielens/movielens.inter'
    if not os.path.exists(inter_file):
        print(f'❌ Error: {inter_file} not found')
        return False

    print(f'✅ Found complete dataset: {inter_file}')

    # Load the complete dataset
    print('📊 Loading complete dataset...')
    recbole_data = pd.read_csv(inter_file, sep='\t')
    print(f'✅ Loaded {len(recbole_data):,} interactions')

    # Verify the data format
    print(f'📋 Data columns: {list(recbole_data.columns)}')
    print(f'📋 Sample data:')
    print(recbole_data.head())

    # Calculate 80th percentile timestamp for chronological split
    # The column name includes the type annotation: 'timestamp:float'
    timestamp_col = 'timestamp:float'
    split_timestamp = recbole_data[timestamp_col].quantile(0.8)
    print(f'📅 Split timestamp (80th percentile): {split_timestamp}')

    # Split chronologically
    past_data = recbole_data[recbole_data[timestamp_col] <= split_timestamp].copy()
    future_data = recbole_data[recbole_data[timestamp_col] > split_timestamp].copy()

    print(f'📊 Past data (training): {len(past_data):,} interactions ({len(past_data)/len(recbole_data)*100:.1f}%)')
    print(f'📊 Future data (ETL pipeline): {len(future_data):,} interactions ({len(future_data)/len(recbole_data)*100:.1f}%)')

    # Create processed directory if it doesn't exist
    os.makedirs('data/processed', exist_ok=True)

    # Export chronologically split files
    past_file = 'data/processed/movielens_past.inter'
    future_file = 'data/processed/movielens_future.inter'

    past_data.to_csv(past_file, sep='\t', index=False)
    future_data.to_csv(future_file, sep='\t', index=False)

    print(f'✅ Created {past_file}: {len(past_data):,} interactions')
    print(f'✅ Created {future_file}: {len(future_data):,} interactions')

    # Verify files were created
    for filepath in [past_file, future_file]:
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath) / (1024*1024)  # MB
            print(f'✅ {filepath}: {file_size:.1f} MB')
        else:
            print(f'❌ Error: {filepath} was not created')

    print('🎯 Step 1 Complete: Missing data files generated!')
    return True

if __name__ == "__main__":
    main()
