"""
Advanced Contextual Recommendation System
Implements context-aware personalization with temporal, demographic, and behavioral features
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContextType(Enum):
    """Types of contextual information"""
    TEMPORAL = "temporal"
    DEMOGRAPHIC = "demographic" 
    BEHAVIORAL = "behavioral"
    ENVIRONMENTAL = "environmental"
    SOCIAL = "social"

@dataclass
class UserContext:
    """User contextual information"""
    user_id: int
    timestamp: datetime
    
    # Temporal context
    hour_of_day: int
    day_of_week: int
    is_weekend: bool
    season: str
    
    # Behavioral context
    session_length: Optional[float] = None
    recent_genres: Optional[List[str]] = None
    avg_rating: Optional[float] = None
    rating_count: Optional[int] = None
    last_activity: Optional[datetime] = None
    
    # Environmental context
    device_type: Optional[str] = None
    platform: Optional[str] = None
    
    # Derived features
    time_since_last_activity: Optional[float] = None
    activity_level: Optional[str] = None  # low, medium, high
    preference_stability: Optional[float] = None

@dataclass
class ItemContext:
    """Item contextual information"""
    item_id: int
    
    # Content features
    genres: List[str]
    release_year: int
    popularity_score: float
    
    # Temporal features
    age_in_days: int
    trending_score: float
    
    # Social features
    avg_rating: float
    rating_count: int
    
    # Derived features
    novelty_score: Optional[float] = None
    diversity_contribution: Optional[float] = None

class ContextualFeatureExtractor:
    """
    Extract and engineer contextual features for personalization
    """
    
    def __init__(self):
        self.genre_categories = [
            'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
            'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
            'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
        ]
        
        # Seasonal mappings
        self.season_mapping = {
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'fall', 10: 'fall', 11: 'fall'
        }
    
    def extract_user_context(self, user_id: int, timestamp: datetime,
                           user_history: pd.DataFrame) -> UserContext:
        """Extract comprehensive user context"""
        
        # Temporal features
        hour_of_day = timestamp.hour
        day_of_week = timestamp.weekday()
        is_weekend = day_of_week >= 5
        season = self.season_mapping[timestamp.month]
        
        # Initialize context
        context = UserContext(
            user_id=user_id,
            timestamp=timestamp,
            hour_of_day=hour_of_day,
            day_of_week=day_of_week,
            is_weekend=is_weekend,
            season=season
        )
        
        # Behavioral features from history
        if not user_history.empty:
            context.avg_rating = user_history['rating'].mean()
            context.rating_count = len(user_history)
            context.last_activity = pd.to_datetime(user_history['timestamp']).max()
            
            # Recent genre preferences (last 30 days)
            recent_cutoff = timestamp - timedelta(days=30)
            recent_ratings = user_history[
                pd.to_datetime(user_history['timestamp']) >= recent_cutoff
            ]
            
            if not recent_ratings.empty and 'genres' in recent_ratings.columns:
                genre_counts = {}
                for genres_str in recent_ratings['genres'].dropna():
                    for genre in genres_str.split('|'):
                        genre_counts[genre] = genre_counts.get(genre, 0) + 1
                
                context.recent_genres = sorted(genre_counts.keys(), 
                                             key=genre_counts.get, reverse=True)[:5]
            
            # Time since last activity
            if context.last_activity:
                time_delta = timestamp - context.last_activity
                context.time_since_last_activity = time_delta.total_seconds() / 3600  # hours
            
            # Activity level classification
            context.activity_level = self._classify_activity_level(context.rating_count)
            
            # Preference stability (variance in ratings)
            if len(user_history) > 1:
                context.preference_stability = 1.0 / (1.0 + user_history['rating'].var())
        
        return context
    
    def extract_item_context(self, item_id: int, item_info: pd.Series,
                           all_ratings: pd.DataFrame, timestamp: datetime) -> ItemContext:
        """Extract comprehensive item context"""
        
        # Basic item information
        genres = item_info.get('genres', '').split('|') if item_info.get('genres') else []
        release_year = item_info.get('year', datetime.now().year)
        
        # Calculate item statistics from ratings
        item_ratings = all_ratings[all_ratings['movieId'] == item_id]
        avg_rating = item_ratings['rating'].mean() if not item_ratings.empty else 3.5
        rating_count = len(item_ratings)
        
        # Popularity score (normalized rating count)
        max_ratings = all_ratings.groupby('movieId').size().max()
        popularity_score = rating_count / max_ratings if max_ratings > 0 else 0.0
        
        # Temporal features
        release_date = datetime(year=release_year, month=1, day=1)
        age_in_days = (timestamp - release_date).days
        
        # Trending score (recent rating activity)
        recent_cutoff = timestamp - timedelta(days=30)
        recent_ratings = item_ratings[
            pd.to_datetime(item_ratings['timestamp']) >= recent_cutoff
        ]
        trending_score = len(recent_ratings) / max(1, len(item_ratings))
        
        context = ItemContext(
            item_id=item_id,
            genres=genres,
            release_year=release_year,
            popularity_score=popularity_score,
            age_in_days=age_in_days,
            trending_score=trending_score,
            avg_rating=avg_rating,
            rating_count=rating_count
        )
        
        # Novelty score (how different from user's typical preferences)
        context.novelty_score = self._calculate_novelty_score(genres)
        
        return context
    
    def _classify_activity_level(self, rating_count: int) -> str:
        """Classify user activity level"""
        if rating_count < 20:
            return 'low'
        elif rating_count < 100:
            return 'medium'
        else:
            return 'high'
    
    def _calculate_novelty_score(self, genres: List[str]) -> float:
        """Calculate novelty score based on genre diversity"""
        if not genres:
            return 0.5
        
        # Simple novelty based on number of genres
        return min(1.0, len(genres) / 3.0)

class ContextualRecommenderModel(nn.Module):
    """
    Neural network model for context-aware recommendations
    Incorporates user, item, and contextual features
    """
    
    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 128,
                 context_dim: int = 64, hidden_dims: List[int] = [256, 128, 64]):
        super().__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.context_dim = context_dim
        
        # User and item embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # Context feature processing
        self.temporal_features = 24 + 7 + 4 + 1  # hour + day + season + weekend
        self.behavioral_features = 8  # avg_rating, count, activity_level, etc.
        self.item_context_features = 10  # popularity, age, trending, etc.
        
        total_context_features = (self.temporal_features + 
                                self.behavioral_features + 
                                self.item_context_features)
        
        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(total_context_features, context_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(context_dim * 2, context_dim),
            nn.ReLU()
        )
        
        # Main prediction network
        input_dim = embedding_dim * 2 + context_dim
        layers = []
        
        for i, hidden_dim in enumerate(hidden_dims):
            if i == 0:
                layers.extend([
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3)
                ])
            else:
                layers.extend([
                    nn.Linear(hidden_dims[i-1], hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                ])
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], 1))
        
        self.prediction_network = nn.Sequential(*layers)
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
    
    def encode_context_features(self, user_context: UserContext, 
                              item_context: ItemContext) -> torch.Tensor:
        """Encode contextual features into vector representation"""
        features = []
        
        # Temporal features (one-hot encoded)
        hour_onehot = torch.zeros(24)
        hour_onehot[user_context.hour_of_day] = 1.0
        features.append(hour_onehot)
        
        day_onehot = torch.zeros(7)
        day_onehot[user_context.day_of_week] = 1.0
        features.append(day_onehot)
        
        # Season one-hot
        seasons = ['spring', 'summer', 'fall', 'winter']
        season_onehot = torch.zeros(4)
        if user_context.season in seasons:
            season_onehot[seasons.index(user_context.season)] = 1.0
        features.append(season_onehot)
        
        # Weekend indicator
        features.append(torch.tensor([1.0 if user_context.is_weekend else 0.0]))
        
        # Behavioral features
        behavioral = torch.tensor([
            user_context.avg_rating / 5.0 if user_context.avg_rating else 0.7,
            min(1.0, (user_context.rating_count or 0) / 1000.0),
            user_context.time_since_last_activity / 168.0 if user_context.time_since_last_activity else 0.5,  # normalize by week
            1.0 if user_context.activity_level == 'high' else 0.0,
            1.0 if user_context.activity_level == 'medium' else 0.0,
            1.0 if user_context.activity_level == 'low' else 0.0,
            user_context.preference_stability or 0.5,
            len(user_context.recent_genres or []) / 5.0
        ])
        features.append(behavioral)
        
        # Item context features
        item_features = torch.tensor([
            item_context.popularity_score,
            min(1.0, item_context.age_in_days / 10000.0),  # normalize by ~27 years
            item_context.trending_score,
            item_context.avg_rating / 5.0,
            min(1.0, item_context.rating_count / 10000.0),
            len(item_context.genres) / 5.0,  # normalize by max expected genres
            item_context.novelty_score or 0.5,
            1.0 if item_context.release_year >= 2010 else 0.0,  # recent movie
            1.0 if item_context.release_year >= 2000 else 0.0,  # modern movie
            item_context.diversity_contribution or 0.5
        ])
        features.append(item_features)
        
        # Concatenate all features
        context_vector = torch.cat(features, dim=0)
        return context_vector
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor,
                context_features: torch.Tensor) -> torch.Tensor:
        """Forward pass with contextual features"""
        
        # Get embeddings
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)
        
        # Process context
        context_encoded = self.context_encoder(context_features)
        
        # Combine all features
        combined_features = torch.cat([user_embeds, item_embeds, context_encoded], dim=1)
        
        # Predict rating
        prediction = self.prediction_network(combined_features)
        
        return prediction.squeeze()

class ContextualRecommendationSystem:
    """
    Complete contextual recommendation system with advanced personalization
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.feature_extractor = ContextualFeatureExtractor()
        self.user_id_map = {}
        self.item_id_map = {}
        self.reverse_user_map = {}
        self.reverse_item_map = {}
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
    
    def train(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame,
              epochs: int = 50, batch_size: int = 1024, 
              learning_rate: float = 0.001, validation_split: float = 0.1) -> Dict[str, Any]:
        """Train the contextual recommendation model"""
        
        logger.info("🚀 Training contextual recommendation model...")
        
        # Create user and item mappings
        unique_users = sorted(ratings_df['userId'].unique())
        unique_items = sorted(ratings_df['movieId'].unique())
        
        self.user_id_map = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.item_id_map = {item_id: idx for idx, item_id in enumerate(unique_items)}
        self.reverse_user_map = {idx: user_id for user_id, idx in self.user_id_map.items()}
        self.reverse_item_map = {idx: item_id for item_id, idx in self.item_id_map.items()}
        
        # Initialize model
        n_users = len(unique_users)
        n_items = len(unique_items)
        self.model = ContextualRecommenderModel(n_users, n_items)
        
        # Prepare training data with context
        train_data = self._prepare_contextual_data(ratings_df, movies_df)
        
        # Split data
        n_train = int(len(train_data) * (1 - validation_split))
        train_subset = train_data[:n_train]
        val_subset = train_data[n_train:]
        
        # Training setup
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        train_losses = []
        val_losses = []
        
        logger.info(f"Training with {len(train_subset)} samples, validating with {len(val_subset)}")
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for i in range(0, len(train_subset), batch_size):
                batch = train_subset[i:i+batch_size]
                
                user_ids = torch.tensor([self.user_id_map[x['user_id']] for x in batch])
                item_ids = torch.tensor([self.item_id_map[x['item_id']] for x in batch])
                context_features = torch.stack([x['context_features'] for x in batch])
                ratings = torch.tensor([x['rating'] for x in batch], dtype=torch.float32)
                
                optimizer.zero_grad()
                predictions = self.model(user_ids, item_ids, context_features)
                loss = criterion(predictions, ratings)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / (len(train_subset) // batch_size + 1)
            train_losses.append(avg_train_loss)
            
            # Validation
            if val_subset:
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for i in range(0, len(val_subset), batch_size):
                        batch = val_subset[i:i+batch_size]
                        
                        user_ids = torch.tensor([self.user_id_map[x['user_id']] for x in batch])
                        item_ids = torch.tensor([self.item_id_map[x['item_id']] for x in batch])
                        context_features = torch.stack([x['context_features'] for x in batch])
                        ratings = torch.tensor([x['rating'] for x in batch], dtype=torch.float32)
                        
                        predictions = self.model(user_ids, item_ids, context_features)
                        loss = criterion(predictions, ratings)
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / (len(val_subset) // batch_size + 1)
                val_losses.append(avg_val_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        training_results = {
            'epochs': epochs,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1] if val_losses else None,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'n_users': n_users,
            'n_items': n_items
        }
        
        logger.info("✅ Training completed!")
        return training_results
    
    def _prepare_contextual_data(self, ratings_df: pd.DataFrame, 
                               movies_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Prepare training data with contextual features"""
        
        logger.info("📊 Preparing contextual training data...")
        
        # Merge ratings with movie info
        merged_df = ratings_df.merge(movies_df, on='movieId', how='left')
        
        contextual_data = []
        
        # Group by user for efficient context extraction
        for user_id, user_ratings in merged_df.groupby('userId'):
            user_ratings = user_ratings.sort_values('timestamp')
            
            for idx, row in user_ratings.iterrows():
                timestamp = pd.to_datetime(row['timestamp'], unit='s')
                
                # Get user history up to this point
                history_mask = user_ratings['timestamp'] < row['timestamp']
                user_history = user_ratings[history_mask]
                
                # Extract contexts
                user_context = self.feature_extractor.extract_user_context(
                    user_id, timestamp, user_history
                )
                
                item_context = self.feature_extractor.extract_item_context(
                    row['movieId'], row, merged_df, timestamp
                )
                
                # Encode features
                context_features = self.model.encode_context_features(
                    user_context, item_context
                ) if self.model else torch.zeros(50)  # placeholder during initialization
                
                contextual_data.append({
                    'user_id': user_id,
                    'item_id': row['movieId'],
                    'rating': row['rating'],
                    'context_features': context_features,
                    'user_context': user_context,
                    'item_context': item_context
                })
        
        logger.info(f"📋 Prepared {len(contextual_data)} contextual training samples")
        return contextual_data
    
    def get_contextual_recommendations(self, user_id: int, n_recommendations: int = 10,
                                     candidate_items: Optional[List[int]] = None,
                                     context_timestamp: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Generate contextual recommendations for a user"""
        
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        context_timestamp = context_timestamp or datetime.now()
        
        # Get user history for context
        # Note: In a real system, this would come from a database
        user_history = pd.DataFrame()  # Placeholder
        
        # Extract user context
        user_context = self.feature_extractor.extract_user_context(
            user_id, context_timestamp, user_history
        )
        
        recommendations = []
        
        # Use all items as candidates if none provided
        if candidate_items is None:
            candidate_items = list(self.reverse_item_map.keys())
        
        self.model.eval()
        with torch.no_grad():
            for item_id in candidate_items:
                if item_id not in self.item_id_map:
                    continue
                
                # Create mock item context (in production, load from database)
                item_context = ItemContext(
                    item_id=item_id,
                    genres=['Unknown'],
                    release_year=2020,
                    popularity_score=0.5,
                    age_in_days=365,
                    trending_score=0.3,
                    avg_rating=3.5,
                    rating_count=100
                )
                
                # Encode features
                context_features = self.model.encode_context_features(user_context, item_context)
                
                # Predict rating
                user_idx = torch.tensor([self.user_id_map[user_id]])
                item_idx = torch.tensor([self.item_id_map[item_id]])
                context_tensor = context_features.unsqueeze(0)
                
                predicted_rating = self.model(user_idx, item_idx, context_tensor).item()
                
                recommendations.append({
                    'item_id': item_id,
                    'predicted_rating': predicted_rating,
                    'user_context': user_context,
                    'item_context': item_context
                })
        
        # Sort by predicted rating
        recommendations.sort(key=lambda x: x['predicted_rating'], reverse=True)
        
        return recommendations[:n_recommendations]
    
    def save_model(self, path: str):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'n_users': self.model.n_users,
                'n_items': self.model.n_items,
                'embedding_dim': self.model.embedding_dim,
                'context_dim': self.model.context_dim
            },
            'user_id_map': self.user_id_map,
            'item_id_map': self.item_id_map,
            'reverse_user_map': self.reverse_user_map,
            'reverse_item_map': self.reverse_item_map
        }
        
        torch.save(model_data, path)
        logger.info(f"💾 Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model"""
        model_data = torch.load(path, map_location='cpu')
        
        config = model_data['model_config']
        self.model = ContextualRecommenderModel(
            n_users=config['n_users'],
            n_items=config['n_items'],
            embedding_dim=config['embedding_dim'],
            context_dim=config['context_dim']
        )
        
        self.model.load_state_dict(model_data['model_state_dict'])
        self.user_id_map = model_data['user_id_map']
        self.item_id_map = model_data['item_id_map']
        self.reverse_user_map = model_data['reverse_user_map']
        self.reverse_item_map = model_data['reverse_item_map']
        
        logger.info(f"📁 Model loaded from {path}")

def main():
    """Test contextual recommendation system"""
    print("🧠 Testing Contextual Recommendation System...")
    
    # Create sample data
    np.random.seed(42)
    
    # Sample ratings data
    n_users = 100
    n_items = 50
    n_ratings = 1000
    
    ratings_data = {
        'userId': np.random.choice(range(1, n_users + 1), n_ratings),
        'movieId': np.random.choice(range(1, n_items + 1), n_ratings),
        'rating': np.random.choice([1, 2, 3, 4, 5], n_ratings),
        'timestamp': np.random.randint(1000000000, 1700000000, n_ratings)
    }
    ratings_df = pd.DataFrame(ratings_data)
    
    # Sample movies data
    movies_data = {
        'movieId': range(1, n_items + 1),
        'title': [f'Movie {i}' for i in range(1, n_items + 1)],
        'genres': [np.random.choice(['Action', 'Comedy', 'Drama'], 1)[0] for _ in range(n_items)],
        'year': np.random.choice(range(1990, 2024), n_items)
    }
    movies_df = pd.DataFrame(movies_data)
    
    # Initialize and train system
    rec_system = ContextualRecommendationSystem()
    
    try:
        training_results = rec_system.train(
            ratings_df, movies_df, 
            epochs=5, batch_size=64
        )
        
        print(f"✅ Training completed!")
        print(f"Final train loss: {training_results['final_train_loss']:.4f}")
        print(f"Final val loss: {training_results['final_val_loss']:.4f}")
        
        # Test recommendations
        recommendations = rec_system.get_contextual_recommendations(
            user_id=1, n_recommendations=5
        )
        
        print(f"\n📋 Top 5 contextual recommendations for user 1:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. Item {rec['item_id']}: {rec['predicted_rating']:.3f}")
        
        # Save model
        rec_system.save_model("data/models/contextual_recommender.pt")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Contextual recommendation system ready!")

if __name__ == "__main__":
    main()