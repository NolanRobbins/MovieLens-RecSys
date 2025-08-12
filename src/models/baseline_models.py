"""
Baseline Models for A/B Testing Against Hybrid VAE
Implements Matrix Factorization, Neural CF, and Two-Tower architectures
Compatible with existing training infrastructure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MatrixFactorizationBaseline(nn.Module):
    """
    Classic Matrix Factorization baseline
    Simple but effective collaborative filtering approach
    """
    
    def __init__(self, n_users: int, n_movies: int, n_factors: int = 100, 
                 dropout_rate: float = 0.2):
        super(MatrixFactorizationBaseline, self).__init__()
        
        self.n_users = n_users
        self.n_movies = n_movies
        self.n_factors = n_factors
        
        # Embedding layers
        self.user_embedding = nn.Embedding(n_users, n_factors)
        self.movie_embedding = nn.Embedding(n_movies, n_factors)
        
        # Bias terms
        self.user_bias = nn.Embedding(n_users, 1)
        self.movie_bias = nn.Embedding(n_movies, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform"""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.movie_embedding.weight)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.movie_bias.weight)
        nn.init.constant_(self.global_bias, 3.5)  # Average rating
    
    def forward(self, users: torch.Tensor, movies: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for rating prediction
        
        Args:
            users: User indices [batch_size]
            movies: Movie indices [batch_size]
            
        Returns:
            Predicted ratings [batch_size]
        """
        # Get embeddings
        user_emb = self.user_embedding(users)  # [batch_size, n_factors]
        movie_emb = self.movie_embedding(movies)  # [batch_size, n_factors]
        
        # Apply dropout
        user_emb = self.dropout(user_emb)
        movie_emb = self.dropout(movie_emb)
        
        # Dot product for interaction
        interaction = (user_emb * movie_emb).sum(dim=1)  # [batch_size]
        
        # Add bias terms
        user_bias = self.user_bias(users).squeeze()
        movie_bias = self.movie_bias(movies).squeeze()
        
        # Final prediction
        prediction = interaction + user_bias + movie_bias + self.global_bias
        
        return prediction
    
    def get_user_embeddings(self) -> torch.Tensor:
        """Get all user embeddings for recommendation generation"""
        return self.user_embedding.weight
    
    def get_movie_embeddings(self) -> torch.Tensor:
        """Get all movie embeddings for recommendation generation"""
        return self.movie_embedding.weight

class NeuralCollaborativeFiltering(nn.Module):
    """
    Neural Collaborative Filtering (NCF) model
    Combines matrix factorization with neural networks
    Based on "Neural Collaborative Filtering" paper
    """
    
    def __init__(self, n_users: int, n_movies: int, 
                 mf_dim: int = 64, mlp_dims: list = [128, 64, 32], 
                 dropout_rate: float = 0.2):
        super(NeuralCollaborativeFiltering, self).__init__()
        
        self.n_users = n_users
        self.n_movies = n_movies
        self.mf_dim = mf_dim
        
        # Matrix Factorization embeddings
        self.user_mf_embedding = nn.Embedding(n_users, mf_dim)
        self.movie_mf_embedding = nn.Embedding(n_movies, mf_dim)
        
        # MLP embeddings (typically larger)
        mlp_input_dim = mlp_dims[0] // 2
        self.user_mlp_embedding = nn.Embedding(n_users, mlp_input_dim)
        self.movie_mlp_embedding = nn.Embedding(n_movies, mlp_input_dim)
        
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
        """Initialize weights"""
        # MF embeddings with smaller variance
        nn.init.normal_(self.user_mf_embedding.weight, std=0.01)
        nn.init.normal_(self.movie_mf_embedding.weight, std=0.01)
        
        # MLP embeddings 
        nn.init.xavier_uniform_(self.user_mlp_embedding.weight)
        nn.init.xavier_uniform_(self.movie_mlp_embedding.weight)
        
        # Linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, users: torch.Tensor, movies: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining MF and MLP paths
        
        Args:
            users: User indices [batch_size]
            movies: Movie indices [batch_size]
            
        Returns:
            Predicted ratings [batch_size]
        """
        # Matrix Factorization path
        user_mf_emb = self.user_mf_embedding(users)
        movie_mf_emb = self.movie_mf_embedding(movies)
        mf_vector = user_mf_emb * movie_mf_emb  # Element-wise product
        
        # MLP path
        user_mlp_emb = self.user_mlp_embedding(users)
        movie_mlp_emb = self.movie_mlp_embedding(movies)
        mlp_input = torch.cat([user_mlp_emb, movie_mlp_emb], dim=1)
        mlp_vector = self.mlp(mlp_input)
        
        # Concatenate MF and MLP vectors
        combined = torch.cat([mf_vector, mlp_vector], dim=1)
        
        # Final prediction
        prediction = self.prediction_layer(combined).squeeze()
        
        return prediction
    
    def get_user_embeddings(self) -> torch.Tensor:
        """Get combined user embeddings for recommendations"""
        mf_emb = self.user_mf_embedding.weight
        mlp_emb = self.user_mlp_embedding.weight
        return torch.cat([mf_emb, mlp_emb], dim=1)

class TwoTowerModel(nn.Module):
    """
    Two-Tower Architecture for large-scale recommendations
    Separate towers for users and items with dot-product interaction
    Efficient for candidate generation in production
    """
    
    def __init__(self, n_users: int, n_movies: int, 
                 embedding_dim: int = 128, tower_dims: list = [256, 128, 64],
                 dropout_rate: float = 0.3):
        super(TwoTowerModel, self).__init__()
        
        self.n_users = n_users
        self.n_movies = n_movies
        self.embedding_dim = embedding_dim
        self.output_dim = tower_dims[-1]
        
        # User and movie embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.movie_embedding = nn.Embedding(n_movies, embedding_dim)
        
        # User tower
        user_layers = []
        input_dim = embedding_dim
        
        for hidden_dim in tower_dims:
            user_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim
        
        self.user_tower = nn.Sequential(*user_layers)
        
        # Movie tower (same architecture)
        movie_layers = []
        input_dim = embedding_dim
        
        for hidden_dim in tower_dims:
            movie_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim
            
        self.movie_tower = nn.Sequential(*movie_layers)
        
        # Temperature parameter for calibration
        self.temperature = nn.Parameter(torch.ones(1))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.movie_embedding.weight)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, users: torch.Tensor, movies: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with two-tower architecture
        
        Args:
            users: User indices [batch_size]
            movies: Movie indices [batch_size]
            
        Returns:
            Predicted ratings/scores [batch_size]
        """
        # Get embeddings
        user_emb = self.user_embedding(users)
        movie_emb = self.movie_embedding(movies)
        
        # Pass through towers
        user_repr = self.user_tower(user_emb)  # [batch_size, output_dim]
        movie_repr = self.movie_tower(movie_emb)  # [batch_size, output_dim]
        
        # Normalize representations for better training stability
        user_repr = F.normalize(user_repr, p=2, dim=1)
        movie_repr = F.normalize(movie_repr, p=2, dim=1)
        
        # Compute similarity (dot product)
        similarity = (user_repr * movie_repr).sum(dim=1)  # [batch_size]
        
        # Apply temperature scaling
        prediction = similarity / self.temperature
        
        return prediction
    
    def get_user_representations(self, user_ids: torch.Tensor) -> torch.Tensor:
        """Get user representations from user tower"""
        user_emb = self.user_embedding(user_ids)
        user_repr = self.user_tower(user_emb)
        return F.normalize(user_repr, p=2, dim=1)
    
    def get_movie_representations(self, movie_ids: torch.Tensor) -> torch.Tensor:
        """Get movie representations from movie tower"""
        movie_emb = self.movie_embedding(movie_ids)
        movie_repr = self.movie_tower(movie_emb)
        return F.normalize(movie_repr, p=2, dim=1)
    
    def get_all_movie_representations(self) -> torch.Tensor:
        """Get all movie representations for efficient retrieval"""
        all_movie_ids = torch.arange(self.n_movies, device=self.movie_embedding.weight.device)
        return self.get_movie_representations(all_movie_ids)

class ModelFactory:
    """Factory class for creating baseline models"""
    
    @staticmethod
    def create_model(model_type: str, n_users: int, n_movies: int, 
                    **kwargs) -> nn.Module:
        """
        Create a model of specified type
        
        Args:
            model_type: Type of model to create
            n_users: Number of users
            n_movies: Number of movies
            **kwargs: Additional model parameters
            
        Returns:
            Initialized model
        """
        model_type = model_type.lower()
        
        if model_type == 'matrix_factorization' or model_type == 'mf':
            return MatrixFactorizationBaseline(
                n_users=n_users,
                n_movies=n_movies,
                n_factors=kwargs.get('n_factors', 100),
                dropout_rate=kwargs.get('dropout_rate', 0.2)
            )
        
        elif model_type == 'neural_cf' or model_type == 'ncf':
            return NeuralCollaborativeFiltering(
                n_users=n_users,
                n_movies=n_movies,
                mf_dim=kwargs.get('mf_dim', 64),
                mlp_dims=kwargs.get('mlp_dims', [128, 64, 32]),
                dropout_rate=kwargs.get('dropout_rate', 0.2)
            )
        
        elif model_type == 'two_tower':
            return TwoTowerModel(
                n_users=n_users,
                n_movies=n_movies,
                embedding_dim=kwargs.get('embedding_dim', 128),
                tower_dims=kwargs.get('tower_dims', [256, 128, 64]),
                dropout_rate=kwargs.get('dropout_rate', 0.3)
            )
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def get_model_info(model_type: str) -> Dict:
        """Get model information and default parameters"""
        info = {
            'matrix_factorization': {
                'name': 'Matrix Factorization',
                'description': 'Classic collaborative filtering baseline',
                'strengths': ['Simple', 'Interpretable', 'Fast training'],
                'weaknesses': ['Limited expressiveness', 'Cold start issues'],
                'default_params': {'n_factors': 100, 'dropout_rate': 0.2}
            },
            'neural_cf': {
                'name': 'Neural Collaborative Filtering',
                'description': 'Combines MF with neural networks', 
                'strengths': ['More expressive', 'Handles non-linearity', 'SOTA performance'],
                'weaknesses': ['More complex', 'Slower training', 'More parameters'],
                'default_params': {'mf_dim': 64, 'mlp_dims': [128, 64, 32], 'dropout_rate': 0.2}
            },
            'two_tower': {
                'name': 'Two-Tower Model',
                'description': 'Separate user and item towers with dot-product',
                'strengths': ['Scalable', 'Good for retrieval', 'Production-friendly'],
                'weaknesses': ['Limited interaction modeling', 'Requires large datasets'],
                'default_params': {'embedding_dim': 128, 'tower_dims': [256, 128, 64], 'dropout_rate': 0.3}
            }
        }
        
        return info.get(model_type.lower(), {})

def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    """Test baseline models creation"""
    print("🔧 Testing baseline models...")
    
    # Mock parameters
    n_users, n_movies = 1000, 5000
    
    # Test all models
    models = ['matrix_factorization', 'neural_cf', 'two_tower']
    
    for model_type in models:
        print(f"\n📊 Creating {model_type}...")
        
        model = ModelFactory.create_model(model_type, n_users, n_movies)
        param_count = count_parameters(model)
        info = ModelFactory.get_model_info(model_type)
        
        print(f"   Parameters: {param_count:,}")
        print(f"   Description: {info.get('description', 'N/A')}")
        print(f"   Strengths: {', '.join(info.get('strengths', []))}")
        
        # Test forward pass
        batch_size = 32
        users = torch.randint(0, n_users, (batch_size,))
        movies = torch.randint(0, n_movies, (batch_size,))
        
        with torch.no_grad():
            predictions = model(users, movies)
            print(f"   Output shape: {predictions.shape}")
            print(f"   Sample predictions: {predictions[:3].tolist()}")
    
    print("\n✅ All baseline models created successfully!")

if __name__ == "__main__":
    main()