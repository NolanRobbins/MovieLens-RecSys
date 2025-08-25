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
import logging

from .components.state_space_models import SSBlock

# Set up logger - level controlled by training script
logger = logging.getLogger(__name__)

# Check if debug mode is enabled via environment variable
import os
DEBUG_MODE = os.getenv('SS4REC_DEBUG_LOGGING', '0') == '1'

def log_tensor_stats(tensor: torch.Tensor, name: str, step: str = "") -> None:
    """Log comprehensive tensor statistics for debugging (only in debug mode)"""
    if not DEBUG_MODE:
        return  # Skip all logging in production mode
    
    if tensor is None:
        logger.debug(f"🔍 [{step}] {name}: None")
        return
    
    # Handle different tensor types
    if tensor.dtype in [torch.float32, torch.float64, torch.float16]:
        # Float tensors - full statistics
        has_nan = torch.isnan(tensor).any()
        has_inf = torch.isinf(tensor).any()
        
        logger.debug(f"🔍 [{step}] {name}: shape={tensor.shape}, dtype={tensor.dtype}, "
                    f"range=[{tensor.min():.6f}, {tensor.max():.6f}], "
                    f"mean={tensor.mean():.6f}, std={tensor.std():.6f}, "
                    f"nan_count={torch.isnan(tensor).sum()}, inf_count={torch.isinf(tensor).sum()}")
        
        if has_nan:
            logger.error(f"🚨 NaN detected in {name} at step {step}!")
        if has_inf:
            logger.error(f"🚨 Inf detected in {name} at step {step}!")
    else:
        # Integer tensors - basic statistics only
        logger.debug(f"🔍 [{step}] {name}: shape={tensor.shape}, dtype={tensor.dtype}, "
                    f"range=[{tensor.min()}, {tensor.max()}], "
                    f"unique_values={tensor.unique().numel()}")

def check_numerical_stability(tensor: torch.Tensor, name: str, max_val: float = 1e6) -> torch.Tensor:
    """Check and fix numerical stability issues (always active for safety)"""
    # Only check float tensors for NaN/Inf
    if tensor.dtype in [torch.float32, torch.float64, torch.float16]:
        if torch.isnan(tensor).any():
            if DEBUG_MODE:
                logger.error(f"🚨 NaN in {name} - replacing with zeros")
            tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
        
        if torch.isinf(tensor).any():
            if DEBUG_MODE:
                logger.error(f"🚨 Inf in {name} - clamping to [{-max_val}, {max_val}]")
            tensor = torch.clamp(tensor, min=-max_val, max=max_val)
    
    return tensor


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
        Compute time intervals between consecutive interactions (OPTIMIZED)
        
        Args:
            timestamps: Timestamp tensor [batch_size, seq_len]
            
        Returns:
            Time intervals [batch_size, seq_len-1]
        """
        if timestamps is None:
            batch_size, seq_len = timestamps.shape if timestamps is not None else (1, self.max_seq_len)
            return torch.ones(batch_size, seq_len - 1, device=self.item_embedding.weight.device) * 0.1
        
        # Fast computation without excessive logging
        time_diffs = timestamps[:, 1:] - timestamps[:, :-1]
        time_intervals = time_diffs.float() / 86400.0  # Convert to days
        
        # Fix negative/zero intervals efficiently 
        time_intervals = torch.where(time_intervals <= 0, 0.01, time_intervals)
        
        # Clip extreme values silently (only warn if many extremes)
        extreme_mask = (time_intervals > 365.0)
        extreme_count = extreme_mask.sum()
        if extreme_count > 1000:  # Only warn for major issues
            logger.warning(f"🚨 Clipping {extreme_count} extreme time intervals")
        
        time_intervals = torch.clamp(time_intervals, min=0.01, max=365.0)
        time_intervals = time_intervals / 365.0  # Normalize [0,1]
        time_intervals = torch.clamp(time_intervals, min=1e-6, max=1.0)
        
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
        logger.debug(f"🔄 encode_sequence: batch_size={batch_size}, seq_len={seq_len}")
        
        # Log input tensors
        log_tensor_stats(item_seq, "input_item_seq", "encode_start")
        if timestamps is not None:
            log_tensor_stats(timestamps, "input_timestamps", "encode_start")
        
        # Item embeddings
        item_emb = self.item_embedding(item_seq)  # [batch_size, seq_len, d_model]
        item_emb = check_numerical_stability(item_emb, "item_embeddings")
        log_tensor_stats(item_emb, "item_embeddings", "encode_embeddings")
        
        # Position embeddings
        positions = torch.arange(seq_len, device=item_seq.device).unsqueeze(0).expand(batch_size, seq_len)
        log_tensor_stats(positions, "position_indices", "encode_embeddings")
        
        pos_emb = self.position_embedding(positions)
        pos_emb = check_numerical_stability(pos_emb, "position_embeddings")
        log_tensor_stats(pos_emb, "position_embeddings", "encode_embeddings")
        
        # Combine embeddings
        x = item_emb + pos_emb
        x = check_numerical_stability(x, "combined_embeddings")
        log_tensor_stats(x, "combined_embeddings", "encode_embeddings")
        
        x = self.dropout(x)
        log_tensor_stats(x, "embeddings_after_dropout", "encode_embeddings")
        
        # Compute time intervals
        time_intervals = self.compute_time_intervals(timestamps) if timestamps is not None else None
        if time_intervals is not None:
            log_tensor_stats(time_intervals, "computed_time_intervals", "encode_time")
        
        # Process through state space blocks
        for i, ss_block in enumerate(self.ss_blocks):
            logger.debug(f"🔍 Processing SS block {i+1}/{len(self.ss_blocks)}")
            x_before = x.clone()
            x = ss_block(x, time_intervals)
            x = check_numerical_stability(x, f"ss_block_{i}_output")
            log_tensor_stats(x, f"ss_block_{i}_output", "encode_ss_blocks")
            
            # Check for vanishing/exploding gradients
            change_magnitude = (x - x_before).abs().mean()
            logger.debug(f"🔍 SS block {i} change magnitude: {change_magnitude:.6f}")
        
        # Final normalization
        x = self.layer_norm(x)
        x = check_numerical_stability(x, "final_layer_norm")
        log_tensor_stats(x, "final_normalized_sequence", "encode_finalization")
        
        logger.debug(f"✅ encode_sequence completed successfully")
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
        logger.debug(f"🔄 SS4Rec.forward: users.shape={users.shape}, item_seq.shape={item_seq.shape}")
        log_tensor_stats(users, "input_users", "forward_start")
        log_tensor_stats(item_seq, "input_item_seq", "forward_start")
        if target_items is not None:
            log_tensor_stats(target_items, "input_target_items", "forward_start")
        if timestamps is not None:
            log_tensor_stats(timestamps, "input_timestamps", "forward_start")
        
        # Encode sequence
        seq_output = self.encode_sequence(item_seq, timestamps)
        seq_output = check_numerical_stability(seq_output, "sequence_output")
        log_tensor_stats(seq_output, "sequence_output", "forward_encoding")
        
        # Get user embeddings
        user_emb = self.user_embedding(users)  # [batch_size, d_model]
        user_emb = check_numerical_stability(user_emb, "user_embeddings")
        log_tensor_stats(user_emb, "user_embeddings", "forward_user_emb")
        
        if self.rating_prediction:
            # Rating prediction mode
            # Use last item's representation + user embedding
            seq_repr = seq_output[:, -1, :]  # [batch_size, d_model]
            log_tensor_stats(seq_repr, "sequence_representation", "forward_rating_prediction")
            
            if target_items is not None:
                logger.debug(f"🔍 Predicting ratings for specific target items")
                
                # Predict rating for specific target items
                target_emb = self.item_embedding(target_items)  # [batch_size, d_model]
                target_emb = check_numerical_stability(target_emb, "target_item_embeddings")
                log_tensor_stats(target_emb, "target_item_embeddings", "forward_rating_prediction")
                
                # Combine user, sequence, and target item representations
                user_context = user_emb + seq_repr
                log_tensor_stats(user_context, "user_context", "forward_rating_prediction")
                
                combined = torch.cat([
                    user_context,  # User context
                    target_emb  # Target item
                ], dim=-1)  # [batch_size, d_model * 2]
                
                combined = check_numerical_stability(combined, "combined_features")
                log_tensor_stats(combined, "combined_features", "forward_rating_prediction")
                
                # Predict rating
                ratings = self.rating_head(combined).squeeze(-1)  # [batch_size]
                ratings = check_numerical_stability(ratings, "predicted_ratings")
                log_tensor_stats(ratings, "predicted_ratings", "forward_rating_prediction")
                
                logger.debug(f"✅ Rating prediction completed successfully")
                return ratings
            else:
                # Return sequence representation for further processing
                return seq_repr
        
        else:
            logger.debug(f"🔍 Ranking mode - scoring all items")
            
            # Ranking mode (original SS4Rec)
            seq_repr = seq_output[:, -1, :]  # [batch_size, d_model]
            log_tensor_stats(seq_repr, "sequence_representation_ranking", "forward_ranking")
            
            # Score all items
            item_scores = self.output_layer(seq_repr)  # [batch_size, n_items]
            item_scores = check_numerical_stability(item_scores, "item_scores")
            log_tensor_stats(item_scores, "item_scores", "forward_ranking")
            
            logger.debug(f"✅ Ranking completed successfully")
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
        logger.debug(f"🔄 predict_all_items: batch_size={batch_size}, n_items={self.n_items}")
        
        log_tensor_stats(users, "predict_all_users", "predict_all_start")
        log_tensor_stats(item_seq, "predict_all_item_seq", "predict_all_start")
        if timestamps is not None:
            log_tensor_stats(timestamps, "predict_all_timestamps", "predict_all_start")
        
        # Get sequence representation
        seq_repr = self.encode_sequence(item_seq, timestamps)[:, -1, :]  # [batch_size, d_model]
        seq_repr = check_numerical_stability(seq_repr, "predict_all_seq_repr")
        log_tensor_stats(seq_repr, "predict_all_seq_repr", "predict_all_encoding")
        
        user_emb = self.user_embedding(users)  # [batch_size, d_model]
        user_emb = check_numerical_stability(user_emb, "predict_all_user_emb")
        log_tensor_stats(user_emb, "predict_all_user_emb", "predict_all_encoding")
        
        # Predict ratings for all items
        all_ratings = []
        
        # Process in chunks to avoid memory issues
        chunk_size = 1000
        logger.debug(f"🔍 Processing {self.n_items} items in chunks of {chunk_size}")
        
        for chunk_idx, start_idx in enumerate(range(0, self.n_items, chunk_size)):
            end_idx = min(start_idx + chunk_size, self.n_items)
            current_chunk_size = end_idx - start_idx
            
            logger.debug(f"🔍 Processing chunk {chunk_idx+1}: items {start_idx} to {end_idx-1} ({current_chunk_size} items)")
            
            item_ids = torch.arange(start_idx, end_idx, device=users.device)
            log_tensor_stats(item_ids, f"chunk_{chunk_idx}_item_ids", "predict_all_chunk")
            
            # Expand for batch processing
            item_ids_batch = item_ids.unsqueeze(0).expand(batch_size, -1)  # [batch_size, chunk_size]
            item_emb_batch = self.item_embedding(item_ids_batch)  # [batch_size, chunk_size, d_model]
            item_emb_batch = check_numerical_stability(item_emb_batch, f"chunk_{chunk_idx}_item_emb")
            log_tensor_stats(item_emb_batch, f"chunk_{chunk_idx}_item_emb", "predict_all_chunk")
            
            # Expand user and sequence representations
            user_seq_repr = (user_emb + seq_repr).unsqueeze(1).expand(-1, item_ids_batch.shape[1], -1)
            log_tensor_stats(user_seq_repr, f"chunk_{chunk_idx}_user_seq_repr", "predict_all_chunk")
            
            # Combine representations
            combined = torch.cat([
                user_seq_repr,  # [batch_size, chunk_size, d_model]
                item_emb_batch  # [batch_size, chunk_size, d_model]
            ], dim=-1)  # [batch_size, chunk_size, d_model * 2]
            
            combined = check_numerical_stability(combined, f"chunk_{chunk_idx}_combined")
            log_tensor_stats(combined, f"chunk_{chunk_idx}_combined", "predict_all_chunk")
            
            # Predict ratings
            chunk_ratings = self.rating_head(combined).squeeze(-1)  # [batch_size, chunk_size]
            chunk_ratings = check_numerical_stability(chunk_ratings, f"chunk_{chunk_idx}_ratings")
            log_tensor_stats(chunk_ratings, f"chunk_{chunk_idx}_ratings", "predict_all_chunk")
            
            all_ratings.append(chunk_ratings)
        
        final_ratings = torch.cat(all_ratings, dim=1)  # [batch_size, n_items]
        log_tensor_stats(final_ratings, "predict_all_final_ratings", "predict_all_completion")
        
        logger.debug(f"✅ predict_all_items completed successfully")
        return final_ratings


def create_ss4rec_model(n_users: int, n_items: int, config: Optional[Dict] = None) -> SS4Rec:
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