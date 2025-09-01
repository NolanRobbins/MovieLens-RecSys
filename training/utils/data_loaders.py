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
            df: DataFrame with columns ['user_idx', 'movie_idx', 'rating']
        """
        self.users = torch.LongTensor(df['user_idx'].values)
        self.items = torch.LongTensor(df['movie_idx'].values) 
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
            df: DataFrame with columns ['user_idx', 'movie_idx', 'rating', 'timestamp']
            max_seq_len: Maximum sequence length
            min_seq_len: Minimum sequence length
        """
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        
        # Sort by user and timestamp
        df_sorted = df.sort_values(['user_idx', 'timestamp'])
        
        # Group by user to create sequences
        self.user_sequences = []
        self.user_ratings = []
        self.user_timestamps = []
        
        for user_idx, group in df_sorted.groupby('user_idx'):
            if len(group) >= min_seq_len:
                # Get sequences
                items = group['movie_idx'].tolist()
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
                       **kwargs) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
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
        
        # Only create test dataset if test_df is not empty
        if not test_df.empty and len(test_df.columns) > 0:
            test_dataset = MovieLensDataset(test_df)
        else:
            test_dataset = None
        
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
        
        # Only create test_loader if test_dataset exists
        if test_dataset is not None:
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
        else:
            test_loader = None
    
    return train_loader, val_loader, test_loader


def prepare_training_data(data_dir: str = "data/processed") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load ONLY training and validation data for RunPod training
    Test data remains unseen until after training completion
    
    Args:
        data_dir: Directory containing processed data
        
    Returns:
        Tuple of (train_df, val_df) - NO test_df during training!
    """
    import logging
    logging.info("Loading training and validation data only...")
    
    # Load ONLY train and validation data
    train_df = pd.read_csv(f"{data_dir}/train_data.csv")
    val_df = pd.read_csv(f"{data_dir}/val_data.csv")
    
    # Verify column formats
    required_cols = ['user_idx', 'movie_idx', 'rating']
    for df_name, df in [("train", train_df), ("val", val_df)]:
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in {df_name}_data.csv: {missing_cols}")
    
    logging.info(f"Loaded train: {len(train_df):,} samples, val: {len(val_df):,} samples")
    return train_df, val_df