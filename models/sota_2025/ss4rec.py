"""
SS4Rec: Continuous-Time Sequential Recommendation with State Space Models

Based on:
- Paper: "SS4Rec: Continuous-Time Sequential Recommendation with State Space Models" (2025)
- Official Implementation: https://github.com/XiaoWei-i/SS4Rec

Adapted for MovieLens rating prediction task while maintaining
core sequential modeling capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple
import numpy as np

from .components.state_space_models import SSBlock


class SS4Rec(nn.Module):
    """
    SS4Rec model for sequential recommendation
    
    Combines time-aware and relation-aware state space models
    for continuous-time sequential recommendation.
    """
    
    def __init__(self,
                 n_users: int,
                 n_items: int,
                 d_model: int = 64,
                 n_layers: int = 2,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2,
                 dt_min: float = 0.001,
                 dt_max: float = 0.1,
                 dropout: float = 0.1,
                 max_seq_len: int = 200,
                 loss_type: str = 'bpr',
                 rating_prediction: bool = True):
        """
        Initialize SS4Rec model
        
        Args:
            n_users: Number of users
            n_items: Number of items
            d_model: Model dimension (embedding size)
            n_layers: Number of SS blocks
            d_state: State space dimension
            d_conv: Convolution dimension
            expand: Expansion factor for Mamba
            dt_min: Minimum time step
            dt_max: Maximum time step
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
            loss_type: Loss function type ('bpr', 'ce', 'mse')
            rating_prediction: Whether to predict ratings (vs. ranking)
        """
        super(SS4Rec, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.loss_type = loss_type.lower()
        self.rating_prediction = rating_prediction
        
        # Embeddings
        self.item_embedding = nn.Embedding(n_items + 1, d_model, padding_idx=0)  # +1 for padding
        self.user_embedding = nn.Embedding(n_users, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # State space blocks
        self.ss_blocks = nn.ModuleList([
            SSBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dt_min=dt_min,
                dt_max=dt_max,
                dropout=dropout
            ) for _ in range(n_layers)
        ])
        
        # Output layers
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        if rating_prediction:
            # For rating prediction (RMSE evaluation)
            self.rating_head = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1)
            )
        else:
            # For ranking (original SS4Rec approach)
            self.output_layer = nn.Linear(d_model, n_items)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        # Initialize embeddings
        nn.init.xavier_uniform_(self.item_embedding.weight[1:])  # Skip padding
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.position_embedding.weight)
        
        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def compute_time_intervals(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Compute time intervals between consecutive interactions
        
        Args:
            timestamps: Timestamp tensor [batch_size, seq_len]
            
        Returns:
            Time intervals [batch_size, seq_len-1]
        """
        if timestamps is None:
            # If no timestamps, use uniform intervals
            batch_size, seq_len = timestamps.shape if timestamps is not None else (1, self.max_seq_len)
            return torch.ones(batch_size, seq_len - 1, device=self.item_embedding.weight.device)
        
        # Compute time differences (in seconds)
        time_diffs = timestamps[:, 1:] - timestamps[:, :-1]
        
        # Debug: check for extreme values
        print(f"DEBUG: time_diffs range: [{time_diffs.min():.2f}, {time_diffs.max():.2f}]")
        
        # Convert to days first to avoid huge numbers
        time_intervals = time_diffs.float() / 86400.0  # Convert seconds to days
        print(f"DEBUG: time_intervals in days: [{time_intervals.min():.6f}, {time_intervals.max():.6f}]")
        
        # Fix zero time differences (identical timestamps) with small positive value
        time_intervals = torch.where(time_intervals == 0.0, 0.01, time_intervals)
        print(f"DEBUG: after zero fix: [{time_intervals.min():.6f}, {time_intervals.max():.6f}]")
        
        # Clip extreme values (0.01 days = ~14 minutes, 365 days = 1 year)
        time_intervals = torch.clamp(time_intervals, min=0.01, max=365.0)
        print(f"DEBUG: after clamp: [{time_intervals.min():.6f}, {time_intervals.max():.6f}]")
        
        # Normalize to [0, 1] range for numerical stability
        time_intervals = time_intervals / 365.0
        print(f"DEBUG: after normalize: [{time_intervals.min():.6f}, {time_intervals.max():.6f}]")
        
        # Additional safety: clamp to prevent any remaining extreme values
        time_intervals = torch.clamp(time_intervals, min=1e-6, max=1.0)
        print(f"DEBUG: final time_intervals: [{time_intervals.min():.6f}, {time_intervals.max():.6f}]")
        
        return time_intervals
    
    def encode_sequence(self, 
                       item_seq: torch.Tensor,
                       timestamps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode item sequence using state space models
        
        Args:
            item_seq: Item sequence [batch_size, seq_len]
            timestamps: Timestamp sequence [batch_size, seq_len]
            
        Returns:
            Encoded sequence [batch_size, seq_len, d_model]
        """
        batch_size, seq_len = item_seq.shape
        
        # Item embeddings
        item_emb = self.item_embedding(item_seq)  # [batch_size, seq_len, d_model]
        
        # Position embeddings
        positions = torch.arange(seq_len, device=item_seq.device).unsqueeze(0).expand(batch_size, seq_len)
        pos_emb = self.position_embedding(positions)
        
        # Combine embeddings
        x = item_emb + pos_emb
        x = self.dropout(x)
        
        # Compute time intervals
        time_intervals = self.compute_time_intervals(timestamps) if timestamps is not None else None
        
        # Process through state space blocks
        for ss_block in self.ss_blocks:
            x = ss_block(x, time_intervals)
        
        # Final normalization
        x = self.layer_norm(x)
        
        return x
    
    def forward(self, 
                users: torch.Tensor,
                item_seq: torch.Tensor,
                target_items: Optional[torch.Tensor] = None,
                timestamps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for SS4Rec
        
        Args:
            users: User IDs [batch_size]
            item_seq: Item sequence [batch_size, seq_len]
            target_items: Target items for prediction [batch_size] or [batch_size, n_candidates]
            timestamps: Timestamp sequence [batch_size, seq_len]
            
        Returns:
            Predictions (ratings or scores) [batch_size] or [batch_size, n_candidates]
        """
        # Encode sequence
        seq_output = self.encode_sequence(item_seq, timestamps)
        if torch.isnan(seq_output).any():
            print(f"NaN in seq_output after encode_sequence!")
            print(f"seq_output stats: min={seq_output.min()}, max={seq_output.max()}, nan_count={torch.isnan(seq_output).sum()}")
        
        # Get user embeddings
        user_emb = self.user_embedding(users)  # [batch_size, d_model]
        if torch.isnan(user_emb).any():
            print(f"NaN in user_emb!")
            print(f"user_emb stats: min={user_emb.min()}, max={user_emb.max()}, nan_count={torch.isnan(user_emb).sum()}")
        
        if self.rating_prediction:
            # Rating prediction mode
            # Use last item's representation + user embedding
            seq_repr = seq_output[:, -1, :]  # [batch_size, d_model]
            
            if target_items is not None:
                # Predict rating for specific target items
                target_emb = self.item_embedding(target_items)  # [batch_size, d_model]
                
                # Combine user, sequence, and target item representations
                combined = torch.cat([
                    user_emb + seq_repr,  # User context
                    target_emb  # Target item
                ], dim=-1)  # [batch_size, d_model * 2]
                
                # Predict rating
                if torch.isnan(combined).any():
                    print(f"NaN in combined tensor before rating_head!")
                    print(f"combined stats: min={combined.min()}, max={combined.max()}, nan_count={torch.isnan(combined).sum()}")
                    print(f"user_emb + seq_repr stats: min={(user_emb + seq_repr).min()}, max={(user_emb + seq_repr).max()}")
                    print(f"target_emb stats: min={target_emb.min()}, max={target_emb.max()}")
                
                ratings = self.rating_head(combined).squeeze(-1)  # [batch_size]
                
                if torch.isnan(ratings).any():
                    print(f"NaN in ratings after rating_head!")
                    print(f"ratings stats: min={ratings.min()}, max={ratings.max()}, nan_count={torch.isnan(ratings).sum()}")
                return ratings
            else:
                # Return sequence representation for further processing
                return seq_repr
        
        else:
            # Ranking mode (original SS4Rec)
            seq_repr = seq_output[:, -1, :]  # [batch_size, d_model]
            
            # Score all items
            item_scores = self.output_layer(seq_repr)  # [batch_size, n_items]
            
            return item_scores
    
    def predict_all_items(self, 
                         users: torch.Tensor,
                         item_seq: torch.Tensor,
                         timestamps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict ratings for all items (for evaluation)
        
        Args:
            users: User IDs [batch_size]
            item_seq: Item sequence [batch_size, seq_len]
            timestamps: Timestamp sequence [batch_size, seq_len]
            
        Returns:
            Predicted ratings for all items [batch_size, n_items]
        """
        batch_size = users.shape[0]
        
        # Get sequence representation
        seq_repr = self.encode_sequence(item_seq, timestamps)[:, -1, :]  # [batch_size, d_model]
        user_emb = self.user_embedding(users)  # [batch_size, d_model]
        
        # Predict ratings for all items
        all_ratings = []
        
        # Process in chunks to avoid memory issues
        chunk_size = 1000
        for start_idx in range(0, self.n_items, chunk_size):
            end_idx = min(start_idx + chunk_size, self.n_items)
            item_ids = torch.arange(start_idx, end_idx, device=users.device)
            
            # Expand for batch processing
            item_ids_batch = item_ids.unsqueeze(0).expand(batch_size, -1)  # [batch_size, chunk_size]
            item_emb_batch = self.item_embedding(item_ids_batch)  # [batch_size, chunk_size, d_model]
            
            # Expand user and sequence representations
            user_seq_repr = (user_emb + seq_repr).unsqueeze(1).expand(-1, item_ids_batch.shape[1], -1)
            
            # Combine representations
            combined = torch.cat([
                user_seq_repr,  # [batch_size, chunk_size, d_model]
                item_emb_batch  # [batch_size, chunk_size, d_model]
            ], dim=-1)  # [batch_size, chunk_size, d_model * 2]
            
            # Predict ratings
            chunk_ratings = self.rating_head(combined).squeeze(-1)  # [batch_size, chunk_size]
            all_ratings.append(chunk_ratings)
        
        return torch.cat(all_ratings, dim=1)  # [batch_size, n_items]


def create_ss4rec_model(n_users: int, n_items: int, config: dict = None) -> SS4Rec:
    """
    Factory function to create SS4Rec model with default or custom config
    
    Args:
        n_users: Number of users
        n_items: Number of items
        config: Optional configuration dictionary
        
    Returns:
        Configured SS4Rec model
    """
    default_config = {
        'd_model': 64,
        'n_layers': 2,
        'd_state': 16,
        'd_conv': 4,
        'expand': 2,
        'dt_min': 0.001,
        'dt_max': 0.1,
        'dropout': 0.1,
        'max_seq_len': 200,
        'loss_type': 'mse',  # Use MSE for rating prediction
        'rating_prediction': True
    }
    
    if config:
        default_config.update(config)
    
    return SS4Rec(
        n_users=n_users,
        n_items=n_items,
        **default_config
    )