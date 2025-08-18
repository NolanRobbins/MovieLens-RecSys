"""
Data loading utilities for MovieLens RecSys training
Supports both Neural CF and SS4Rec data requirements
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional


class MovieLensDataset(Dataset):
    """Standard MovieLens dataset for Neural CF training"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize dataset from DataFrame
        
        Args:
            df: DataFrame with columns ['user_id', 'movie_id', 'rating']
        """
        self.users = torch.LongTensor(df['user_id'].values)
        self.items = torch.LongTensor(df['movie_id'].values) 
        self.ratings = torch.FloatTensor(df['rating'].values)
    
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]


class SequentialMovieLensDataset(Dataset):
    """Sequential MovieLens dataset for SS4Rec training"""
    
    def __init__(self, 
                 df: pd.DataFrame, 
                 max_seq_len: int = 200,
                 min_seq_len: int = 10):
        """
        Initialize sequential dataset
        
        Args:
            df: DataFrame with columns ['user_id', 'movie_id', 'rating', 'timestamp']
            max_seq_len: Maximum sequence length
            min_seq_len: Minimum sequence length
        """
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        
        # Sort by user and timestamp
        df_sorted = df.sort_values(['user_id', 'timestamp'])
        
        # Group by user to create sequences
        self.user_sequences = []
        self.user_ratings = []
        self.user_timestamps = []
        
        for user_id, group in df_sorted.groupby('user_id'):
            if len(group) >= min_seq_len:
                # Get sequences
                items = group['movie_id'].tolist()
                ratings = group['rating'].tolist()
                timestamps = group['timestamp'].tolist()
                
                # Create sliding windows if sequence is too long
                if len(items) > max_seq_len:
                    for i in range(len(items) - max_seq_len + 1):
                        self.user_sequences.append(items[i:i + max_seq_len])
                        self.user_ratings.append(ratings[i:i + max_seq_len])
                        self.user_timestamps.append(timestamps[i:i + max_seq_len])
                else:
                    self.user_sequences.append(items)
                    self.user_ratings.append(ratings)
                    self.user_timestamps.append(timestamps)
    
    def __len__(self):
        return len(self.user_sequences)
    
    def __getitem__(self, idx):
        sequence = self.user_sequences[idx]
        ratings = self.user_ratings[idx]
        timestamps = self.user_timestamps[idx]
        
        # Pad sequences to max_seq_len
        seq_len = len(sequence)
        
        if seq_len < self.max_seq_len:
            # Pad with zeros
            sequence = sequence + [0] * (self.max_seq_len - seq_len)
            ratings = ratings + [0.0] * (self.max_seq_len - seq_len)
            timestamps = timestamps + [0] * (self.max_seq_len - seq_len)
        
        return {
            'items': torch.LongTensor(sequence),
            'ratings': torch.FloatTensor(ratings),
            'timestamps': torch.LongTensor(timestamps),
            'seq_len': torch.LongTensor([seq_len])
        }


def create_data_loaders(train_df: pd.DataFrame,
                       val_df: pd.DataFrame, 
                       test_df: pd.DataFrame,
                       batch_size: int = 1024,
                       num_workers: int = 4,
                       sequential: bool = False,
                       **kwargs) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for training
    
    Args:
        train_df, val_df, test_df: DataFrames with interaction data
        batch_size: Batch size for training
        num_workers: Number of worker processes
        sequential: Whether to create sequential datasets (for SS4Rec)
        **kwargs: Additional arguments for sequential dataset
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    if sequential:
        # Create sequential datasets for SS4Rec
        train_dataset = SequentialMovieLensDataset(train_df, **kwargs)
        val_dataset = SequentialMovieLensDataset(val_df, **kwargs)
        test_dataset = SequentialMovieLensDataset(test_df, **kwargs)
        
        def collate_fn(batch):
            """Custom collate function for sequential data"""
            items = torch.stack([item['items'] for item in batch])
            ratings = torch.stack([item['ratings'] for item in batch])
            timestamps = torch.stack([item['timestamps'] for item in batch])
            seq_lens = torch.stack([item['seq_len'] for item in batch])
            
            return {
                'items': items,
                'ratings': ratings, 
                'timestamps': timestamps,
                'seq_lens': seq_lens
            }
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False, 
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers, 
            collate_fn=collate_fn,
            pin_memory=True
        )
        
    else:
        # Create standard datasets for Neural CF
        train_dataset = MovieLensDataset(train_df)
        val_dataset = MovieLensDataset(val_df)
        test_dataset = MovieLensDataset(test_df)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return train_loader, val_loader, test_loader


def prepare_movielens_data(data_dir: str, 
                          min_interactions: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare MovieLens data with proper splits
    
    Args:
        data_dir: Directory containing processed data
        min_interactions: Minimum interactions per user/item
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Load data
    train_df = pd.read_csv(f"{data_dir}/train_data.csv")
    val_df = pd.read_csv(f"{data_dir}/val_data.csv") 
    test_df = pd.read_csv(f"{data_dir}/test_data.csv")
    
    # Filter by minimum interactions if needed
    if min_interactions > 0:
        # Count interactions per user/item
        all_df = pd.concat([train_df, val_df, test_df])
        
        user_counts = all_df['user_id'].value_counts()
        item_counts = all_df['movie_id'].value_counts()
        
        valid_users = user_counts[user_counts >= min_interactions].index
        valid_items = item_counts[item_counts >= min_interactions].index
        
        # Filter datasets
        train_df = train_df[
            (train_df['user_id'].isin(valid_users)) & 
            (train_df['movie_id'].isin(valid_items))
        ]
        val_df = val_df[
            (val_df['user_id'].isin(valid_users)) & 
            (val_df['movie_id'].isin(valid_items))
        ]
        test_df = test_df[
            (test_df['user_id'].isin(valid_users)) & 
            (test_df['movie_id'].isin(valid_items))
        ]
    
    return train_df, val_df, test_df