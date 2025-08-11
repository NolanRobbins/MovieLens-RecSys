"""
Production Feature Store for MovieLens RecSys
SQLite-based feature storage with timestamp versioning and fast lookups
Integrates with existing ETL pipeline and data structure
"""

import sqlite3
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass, asdict
import torch
import hashlib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UserFeatures:
    """User feature schema"""
    user_id: int
    embedding: np.ndarray  # VAE learned embedding
    avg_rating: float
    rating_count: int
    genres_preference: Dict[str, float]  # Genre preference scores
    rating_variance: float
    favorite_decade: int
    last_activity: datetime
    created_at: datetime
    updated_at: datetime

@dataclass
class ItemFeatures:
    """Movie feature schema"""
    movie_id: int
    title: str
    genres: List[str]
    release_year: int
    avg_rating: float
    rating_count: int
    popularity_score: float  # Rolling window popularity
    genre_diversity: float
    recency_boost: float  # Boost for recent movies
    created_at: datetime
    updated_at: datetime

@dataclass
class InteractionFeatures:
    """User-item interaction features"""
    user_id: int
    movie_id: int
    rating: float
    timestamp: datetime
    session_id: str
    interaction_type: str  # 'explicit', 'implicit'
    context: Dict[str, Any]  # Time of day, device, etc.
    created_at: datetime

class FeatureStore:
    """
    Production-grade feature store using SQLite
    Optimized for fast lookups and temporal versioning
    """
    
    def __init__(self, db_path: str = "data/processed/feature_store.db", 
                 data_dir: str = "data/processed"):
        self.db_path = Path(db_path)
        self.data_dir = Path(data_dir)
        
        # Create directory if it doesn't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._initialize_db()
        
        # Load existing data mappings if available
        self.data_mappings = self._load_data_mappings()
        
        logger.info(f"FeatureStore initialized at {self.db_path}")
    
    def _initialize_db(self):
        """Initialize SQLite database with optimized schema"""
        with sqlite3.connect(self.db_path) as conn:
            # User features table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_features (
                    user_id INTEGER PRIMARY KEY,
                    embedding BLOB,
                    avg_rating REAL,
                    rating_count INTEGER,
                    genres_preference TEXT,
                    rating_variance REAL,
                    favorite_decade INTEGER,
                    last_activity TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            """)
            
            # Movie features table  
            conn.execute("""
                CREATE TABLE IF NOT EXISTS item_features (
                    movie_id INTEGER PRIMARY KEY,
                    title TEXT,
                    genres TEXT,
                    release_year INTEGER,
                    avg_rating REAL,
                    rating_count INTEGER,
                    popularity_score REAL,
                    genre_diversity REAL,
                    recency_boost REAL,
                    created_at TEXT,
                    updated_at TEXT
                )
            """)
            
            # Interaction features table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS interaction_features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    movie_id INTEGER,
                    rating REAL,
                    timestamp TEXT,
                    session_id TEXT,
                    interaction_type TEXT,
                    context TEXT,
                    created_at TEXT,
                    FOREIGN KEY (user_id) REFERENCES user_features (user_id),
                    FOREIGN KEY (movie_id) REFERENCES item_features (movie_id)
                )
            """)
            
            # Create indexes for fast lookups
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_features_user_id ON user_features(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_item_features_movie_id ON item_features(movie_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_interactions_user_id ON interaction_features(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_interactions_movie_id ON interaction_features(movie_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_interactions_timestamp ON interaction_features(timestamp)")
            
            conn.commit()
    
    def _load_data_mappings(self) -> Dict:
        """Load existing data mappings from ETL pipeline"""
        mappings_path = self.data_dir / "data_mappings.pkl"
        if mappings_path.exists():
            try:
                with open(mappings_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Could not load data mappings: {e}")
                return {}
        return {}
    
    def populate_from_existing_data(self):
        """
        Populate feature store from existing processed data
        Integrates with current data structure
        """
        logger.info("Populating feature store from existing data...")
        
        # Load processed data
        train_data = pd.read_csv(self.data_dir / "train_data.csv")
        movies_data = pd.read_csv(self.data_dir / "movies_processed.csv")
        
        # Generate user features from training data
        self._generate_user_features(train_data, movies_data)
        
        # Generate item features
        self._generate_item_features(movies_data, train_data)
        
        # Store interaction features
        self._store_interaction_features(train_data)
        
        logger.info("Feature store population completed")
    
    def _generate_user_features(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
        """Generate and store user features"""
        logger.info("Generating user features...")
        
        # Calculate user statistics
        user_stats = ratings_df.groupby('user_id').agg({
            'rating': ['mean', 'count', 'std'],
            'movie_id': 'count'
        }).reset_index()
        
        user_stats.columns = ['user_id', 'avg_rating', 'rating_count', 'rating_variance', 'total_movies']
        user_stats['rating_variance'].fillna(0, inplace=True)
        
        # Calculate genre preferences for each user
        user_genre_prefs = self._calculate_user_genre_preferences(ratings_df, movies_df)
        
        # Generate mock embeddings (in production, these would come from your VAE model)
        user_embeddings = self._generate_user_embeddings(user_stats['user_id'].values)
        
        # Store user features
        with sqlite3.connect(self.db_path) as conn:
            for _, user_row in user_stats.iterrows():
                user_id = int(user_row['user_id'])
                
                user_features = UserFeatures(
                    user_id=user_id,
                    embedding=user_embeddings.get(user_id, np.random.randn(64)),
                    avg_rating=float(user_row['avg_rating']),
                    rating_count=int(user_row['rating_count']),
                    genres_preference=user_genre_prefs.get(user_id, {}),
                    rating_variance=float(user_row['rating_variance']),
                    favorite_decade=2000,  # Default
                    last_activity=datetime.now(),
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                
                self._store_user_features(conn, user_features)
    
    def _calculate_user_genre_preferences(self, ratings_df: pd.DataFrame, 
                                        movies_df: pd.DataFrame) -> Dict[int, Dict[str, float]]:
        """Calculate user genre preference scores"""
        # Merge ratings with movie genres
        ratings_with_genres = ratings_df.merge(movies_df, left_on='movie_id', right_on='movieId')
        
        user_genre_prefs = {}
        
        for user_id, user_ratings in ratings_with_genres.groupby('user_id'):
            genre_scores = {}
            
            for _, row in user_ratings.iterrows():
                genres = eval(row['genres_list'])  # Convert string list to actual list
                rating = row['rating']
                
                for genre in genres:
                    if genre not in genre_scores:
                        genre_scores[genre] = []
                    genre_scores[genre].append(rating)
            
            # Calculate average preference for each genre
            user_genre_prefs[user_id] = {
                genre: np.mean(scores) for genre, scores in genre_scores.items()
            }
        
        return user_genre_prefs
    
    def _generate_user_embeddings(self, user_ids: np.ndarray) -> Dict[int, np.ndarray]:
        """Generate user embeddings (placeholder - in production use VAE embeddings)"""
        embeddings = {}
        for user_id in user_ids:
            # Generate reproducible random embedding
            np.random.seed(int(user_id))
            embeddings[user_id] = np.random.randn(64).astype(np.float32)
        return embeddings
    
    def _generate_item_features(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame):
        """Generate and store movie features"""
        logger.info("Generating item features...")
        
        # Calculate movie statistics
        movie_stats = ratings_df.groupby('movie_id').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        movie_stats.columns = ['movie_id', 'avg_rating', 'rating_count']
        
        # Merge with movie metadata
        movies_with_stats = movies_df.merge(movie_stats, left_on='movieId', right_on='movie_id', how='left')
        movies_with_stats['avg_rating'].fillna(3.0, inplace=True)
        movies_with_stats['rating_count'].fillna(0, inplace=True)
        
        with sqlite3.connect(self.db_path) as conn:
            for _, movie_row in movies_with_stats.iterrows():
                movie_id = int(movie_row['movieId'])
                genres_list = eval(movie_row['genres_list'])
                
                item_features = ItemFeatures(
                    movie_id=movie_id,
                    title=str(movie_row['title']),
                    genres=genres_list,
                    release_year=int(movie_row['release_year']) if pd.notna(movie_row['release_year']) else 1990,
                    avg_rating=float(movie_row['avg_rating']),
                    rating_count=int(movie_row['rating_count']),
                    popularity_score=self._calculate_popularity_score(movie_row),
                    genre_diversity=len(genres_list),
                    recency_boost=self._calculate_recency_boost(movie_row['release_year']),
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                
                self._store_item_features(conn, item_features)
    
    def _calculate_popularity_score(self, movie_row) -> float:
        """Calculate popularity score based on rating count and average"""
        rating_count = movie_row['rating_count']
        avg_rating = movie_row['avg_rating']
        
        # Wilson score for popularity
        if rating_count == 0:
            return 0.0
        
        # Simple popularity score (can be enhanced)
        return float(np.log1p(rating_count) * avg_rating / 5.0)
    
    def _calculate_recency_boost(self, release_year) -> float:
        """Calculate boost for recent movies"""
        if pd.isna(release_year):
            return 0.0
        
        current_year = datetime.now().year
        years_old = current_year - release_year
        
        # Exponential decay for recency boost
        return float(np.exp(-years_old / 10.0))
    
    def _store_interaction_features(self, ratings_df: pd.DataFrame):
        """Store interaction features"""
        logger.info("Storing interaction features...")
        
        with sqlite3.connect(self.db_path) as conn:
            for _, rating_row in ratings_df.iterrows():
                interaction = InteractionFeatures(
                    user_id=int(rating_row['user_id']),
                    movie_id=int(rating_row['movie_id']),
                    rating=float(rating_row['rating']),
                    timestamp=datetime.now(),  # In production, use actual timestamp
                    session_id=self._generate_session_id(rating_row['user_id']),
                    interaction_type='explicit',
                    context={},
                    created_at=datetime.now()
                )
                
                self._store_interaction_features_single(conn, interaction)
    
    def _generate_session_id(self, user_id: int) -> str:
        """Generate session ID for user"""
        return hashlib.md5(f"{user_id}_{datetime.now().date()}".encode()).hexdigest()[:8]
    
    def _store_user_features(self, conn, user_features: UserFeatures):
        """Store user features in database"""
        conn.execute("""
            INSERT OR REPLACE INTO user_features 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_features.user_id,
            pickle.dumps(user_features.embedding),
            user_features.avg_rating,
            user_features.rating_count,
            json.dumps(user_features.genres_preference),
            user_features.rating_variance,
            user_features.favorite_decade,
            user_features.last_activity.isoformat(),
            user_features.created_at.isoformat(),
            user_features.updated_at.isoformat()
        ))
    
    def _store_item_features(self, conn, item_features: ItemFeatures):
        """Store item features in database"""
        conn.execute("""
            INSERT OR REPLACE INTO item_features 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            item_features.movie_id,
            item_features.title,
            json.dumps(item_features.genres),
            item_features.release_year,
            item_features.avg_rating,
            item_features.rating_count,
            item_features.popularity_score,
            item_features.genre_diversity,
            item_features.recency_boost,
            item_features.created_at.isoformat(),
            item_features.updated_at.isoformat()
        ))
    
    def _store_interaction_features_single(self, conn, interaction: InteractionFeatures):
        """Store single interaction feature"""
        conn.execute("""
            INSERT INTO interaction_features 
            (user_id, movie_id, rating, timestamp, session_id, interaction_type, context, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            interaction.user_id,
            interaction.movie_id,
            interaction.rating,
            interaction.timestamp.isoformat(),
            interaction.session_id,
            interaction.interaction_type,
            json.dumps(interaction.context),
            interaction.created_at.isoformat()
        ))
    
    def get_user_features(self, user_id: int) -> Optional[UserFeatures]:
        """Fast user features lookup"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM user_features WHERE user_id = ?
            """, (user_id,))
            
            row = cursor.fetchone()
            if row is None:
                return None
            
            return UserFeatures(
                user_id=row[0],
                embedding=pickle.loads(row[1]),
                avg_rating=row[2],
                rating_count=row[3],
                genres_preference=json.loads(row[4]),
                rating_variance=row[5],
                favorite_decade=row[6],
                last_activity=datetime.fromisoformat(row[7]),
                created_at=datetime.fromisoformat(row[8]),
                updated_at=datetime.fromisoformat(row[9])
            )
    
    def get_item_features(self, movie_id: int) -> Optional[ItemFeatures]:
        """Fast item features lookup"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM item_features WHERE movie_id = ?
            """, (movie_id,))
            
            row = cursor.fetchone()
            if row is None:
                return None
            
            return ItemFeatures(
                movie_id=row[0],
                title=row[1],
                genres=json.loads(row[2]),
                release_year=row[3],
                avg_rating=row[4],
                rating_count=row[5],
                popularity_score=row[6],
                genre_diversity=row[7],
                recency_boost=row[8],
                created_at=datetime.fromisoformat(row[9]),
                updated_at=datetime.fromisoformat(row[10])
            )
    
    def get_user_interactions(self, user_id: int, limit: int = 100) -> List[InteractionFeatures]:
        """Get recent user interactions"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT user_id, movie_id, rating, timestamp, session_id, 
                       interaction_type, context, created_at
                FROM interaction_features 
                WHERE user_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (user_id, limit))
            
            interactions = []
            for row in cursor.fetchall():
                interactions.append(InteractionFeatures(
                    user_id=row[0],
                    movie_id=row[1],
                    rating=row[2],
                    timestamp=datetime.fromisoformat(row[3]),
                    session_id=row[4],
                    interaction_type=row[5],
                    context=json.loads(row[6]),
                    created_at=datetime.fromisoformat(row[7])
                ))
            
            return interactions
    
    def get_feature_store_stats(self) -> Dict[str, Any]:
        """Get feature store statistics for monitoring"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM user_features")
            user_count = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM item_features")
            item_count = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM interaction_features")
            interaction_count = cursor.fetchone()[0]
            
            # Get latest interaction timestamp
            cursor = conn.execute("""
                SELECT MAX(timestamp) FROM interaction_features
            """)
            latest_interaction = cursor.fetchone()[0]
        
        return {
            'user_count': user_count,
            'item_count': item_count,
            'interaction_count': interaction_count,
            'latest_interaction': latest_interaction,
            'db_size_mb': self.db_path.stat().st_size / 1024 / 1024 if self.db_path.exists() else 0,
            'last_updated': datetime.now().isoformat()
        }

def main():
    """Initialize and populate feature store"""
    feature_store = FeatureStore()
    feature_store.populate_from_existing_data()
    
    # Print statistics
    stats = feature_store.get_feature_store_stats()
    print("Feature Store Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()