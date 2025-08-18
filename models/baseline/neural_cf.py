"""
Neural Collaborative Filtering (NCF) - Baseline Model
Based on: "Neural Collaborative Filtering" (WWW 2017)

Clean implementation for baseline comparison with 2025 SOTA models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class NeuralCollaborativeFiltering(nn.Module):
    """
    Neural Collaborative Filtering (NCF) model
    
    Combines Matrix Factorization (MF) and Multi-Layer Perceptron (MLP)
    for collaborative filtering with implicit feedback.
    
    Architecture:
    - MF path: Element-wise product of user/item embeddings
    - MLP path: Concatenated embeddings through neural network
    - Final: Concatenate MF + MLP outputs → prediction layer
    """
    
    def __init__(self, 
                 n_users: int, 
                 n_items: int,
                 mf_dim: int = 64,
                 mlp_dims: list = [128, 64, 32],
                 dropout_rate: float = 0.2):
        """
        Initialize NCF model
        
        Args:
            n_users: Number of users
            n_items: Number of items (movies)
            mf_dim: Dimension for MF embeddings
            mlp_dims: List of hidden dimensions for MLP path
            dropout_rate: Dropout probability
        """
        super(NeuralCollaborativeFiltering, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.mf_dim = mf_dim
        self.mlp_dims = mlp_dims
        
        # Matrix Factorization embeddings
        self.user_mf_embedding = nn.Embedding(n_users, mf_dim)
        self.item_mf_embedding = nn.Embedding(n_items, mf_dim)
        
        # MLP embeddings (separate from MF)
        mlp_input_dim = mlp_dims[0] // 2  # Split between user and item
        self.user_mlp_embedding = nn.Embedding(n_users, mlp_input_dim)
        self.item_mlp_embedding = nn.Embedding(n_items, mlp_input_dim)
        
        # MLP layers
        mlp_layers = []
        input_dim = mlp_dims[0]
        
        for hidden_dim in mlp_dims[1:]:
            mlp_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(hidden_dim)
            ])
            input_dim = hidden_dim
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Final prediction layer
        final_input_dim = mf_dim + mlp_dims[-1]
        self.prediction_layer = nn.Linear(final_input_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        # MF embeddings with smaller variance
        nn.init.normal_(self.user_mf_embedding.weight, std=0.01)
        nn.init.normal_(self.item_mf_embedding.weight, std=0.01)
        
        # MLP embeddings 
        nn.init.xavier_uniform_(self.user_mlp_embedding.weight)
        nn.init.xavier_uniform_(self.item_mlp_embedding.weight)
        
        # Linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining MF and MLP paths
        
        Args:
            users: User indices [batch_size]
            items: Item indices [batch_size]
            
        Returns:
            Predicted ratings [batch_size]
        """
        # Matrix Factorization path
        user_mf_emb = self.user_mf_embedding(users)
        item_mf_emb = self.item_mf_embedding(items)
        mf_vector = user_mf_emb * item_mf_emb  # Element-wise product
        
        # MLP path
        user_mlp_emb = self.user_mlp_embedding(users)
        item_mlp_emb = self.item_mlp_embedding(items)
        mlp_input = torch.cat([user_mlp_emb, item_mlp_emb], dim=1)
        mlp_vector = self.mlp(mlp_input)
        
        # Concatenate MF and MLP vectors
        combined = torch.cat([mf_vector, mlp_vector], dim=1)
        
        # Final prediction
        prediction = self.prediction_layer(combined).squeeze()
        
        return prediction
    
    def get_user_embeddings(self) -> torch.Tensor:
        """Get combined user embeddings for inference"""
        mf_emb = self.user_mf_embedding.weight
        mlp_emb = self.user_mlp_embedding.weight
        return torch.cat([mf_emb, mlp_emb], dim=1)
    
    def get_item_embeddings(self) -> torch.Tensor:
        """Get combined item embeddings for inference"""
        mf_emb = self.item_mf_embedding.weight
        mlp_emb = self.item_mlp_embedding.weight
        return torch.cat([mf_emb, mlp_emb], dim=1)
    
    def predict_all(self, user_ids: torch.Tensor) -> torch.Tensor:
        """
        Predict ratings for given users against all items
        
        Args:
            user_ids: User indices [batch_size]
            
        Returns:
            Predicted ratings [batch_size, n_items]
        """
        batch_size = user_ids.size(0)
        all_items = torch.arange(self.n_items, device=user_ids.device)
        
        # Expand for batch computation
        users_expanded = user_ids.unsqueeze(1).expand(batch_size, self.n_items)
        items_expanded = all_items.unsqueeze(0).expand(batch_size, self.n_items)
        
        # Flatten for forward pass
        users_flat = users_expanded.reshape(-1)
        items_flat = items_expanded.reshape(-1)
        
        # Get predictions
        predictions = self.forward(users_flat, items_flat)
        
        # Reshape back to [batch_size, n_items]
        return predictions.reshape(batch_size, self.n_items)


def create_ncf_model(n_users: int, n_items: int, config: dict = None) -> NeuralCollaborativeFiltering:
    """
    Factory function to create NCF model with default or custom config
    
    Args:
        n_users: Number of users
        n_items: Number of items
        config: Optional configuration dictionary
        
    Returns:
        Configured NCF model
    """
    default_config = {
        'mf_dim': 64,
        'mlp_dims': [128, 64, 32],
        'dropout_rate': 0.2
    }
    
    if config:
        default_config.update(config)
    
    return NeuralCollaborativeFiltering(
        n_users=n_users,
        n_items=n_items,
        **default_config
    )