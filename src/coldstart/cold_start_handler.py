"""
Cold Start Problem Handler
Comprehensive solution for new user and new item recommendation challenges
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path
import json
from collections import defaultdict, Counter
import torch
import torch.nn as nn
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ColdStartType(Enum):
    """Types of cold start scenarios"""
    NEW_USER = "new_user"           # User with no interaction history
    NEW_ITEM = "new_item"           # Item with no ratings/interactions
    NEW_USER_NEW_ITEM = "new_user_new_item"  # Both user and item are new
    SPARSE_USER = "sparse_user"     # User with very few interactions
    SPARSE_ITEM = "sparse_item"     # Item with very few interactions

class ColdStartStrategy(Enum):
    """Cold start recommendation strategies"""
    POPULARITY = "popularity"                   # Most popular items
    DEMOGRAPHIC = "demographic"                # Demographic-based filtering
    CONTENT_BASED = "content_based"            # Content similarity
    ONBOARDING = "onboarding"                  # Interactive onboarding
    HYBRID = "hybrid"                          # Combination of strategies
    CLUSTERING = "clustering"                  # User/item clustering
    KNOWLEDGE_BASED = "knowledge_based"        # Rule-based recommendations

@dataclass
class UserProfile:
    """User profile for cold start scenarios"""
    user_id: int
    
    # Demographic information
    age: Optional[int] = None
    gender: Optional[str] = None
    occupation: Optional[str] = None
    location: Optional[str] = None
    
    # Behavioral signals
    signup_date: Optional[datetime] = None
    device_type: Optional[str] = None
    platform: Optional[str] = None
    
    # Explicit preferences (from onboarding)
    preferred_genres: Optional[List[str]] = None
    disliked_genres: Optional[List[str]] = None
    rating_preferences: Optional[Dict[str, float]] = None
    
    # Inferred characteristics
    interaction_count: int = 0
    avg_session_duration: Optional[float] = None
    time_of_day_preference: Optional[str] = None

@dataclass
class ItemProfile:
    """Item profile for cold start scenarios"""
    item_id: int
    
    # Content features
    title: str
    genres: List[str]
    release_year: int
    duration_minutes: Optional[int] = None
    language: Optional[str] = None
    country: Optional[str] = None
    
    # Metadata
    director: Optional[str] = None
    cast: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    description: Optional[str] = None
    
    # Computed features
    popularity_score: float = 0.0
    quality_score: float = 0.0
    trending_score: float = 0.0
    novelty_score: float = 0.0
    
    # Statistics
    rating_count: int = 0
    avg_rating: Optional[float] = None
    rating_distribution: Optional[Dict[int, int]] = None

@dataclass
class ColdStartRecommendation:
    """Cold start recommendation with explanation"""
    item_id: int
    score: float
    strategy: ColdStartStrategy
    confidence: float
    explanation: str
    metadata: Dict[str, Any] = None

class PopularityRecommender:
    """Popularity-based recommendations for cold start"""
    
    def __init__(self):
        self.global_popularity = {}
        self.demographic_popularity = {}
        self.genre_popularity = {}
        self.temporal_popularity = {}
    
    def fit(self, ratings_df: pd.DataFrame, users_df: Optional[pd.DataFrame] = None,
            items_df: Optional[pd.DataFrame] = None):
        """Fit popularity models"""
        logger.info("📊 Computing popularity statistics...")
        
        # Global popularity
        item_stats = ratings_df.groupby('movieId').agg({
            'rating': ['count', 'mean'],
            'timestamp': 'max'
        }).round(3)
        
        item_stats.columns = ['rating_count', 'avg_rating', 'last_rating']
        
        # Popularity score (weighted rating count and average)
        max_count = item_stats['rating_count'].max()
        for item_id, row in item_stats.iterrows():
            normalized_count = row['rating_count'] / max_count
            popularity = 0.7 * normalized_count + 0.3 * (row['avg_rating'] / 5.0)
            self.global_popularity[item_id] = popularity
        
        # Demographic popularity
        if users_df is not None:
            merged_df = ratings_df.merge(users_df, on='userId', how='left')
            
            for demographic in ['gender', 'age_group', 'occupation']:
                if demographic in merged_df.columns:
                    demo_stats = merged_df.groupby([demographic, 'movieId'])['rating'].mean()
                    self.demographic_popularity[demographic] = demo_stats.to_dict()
        
        # Genre popularity
        if items_df is not None:
            ratings_with_genres = ratings_df.merge(items_df, on='movieId', how='left')
            
            genre_ratings = defaultdict(list)
            for _, row in ratings_with_genres.iterrows():
                if pd.notna(row.get('genres')):
                    genres = row['genres'].split('|')
                    for genre in genres:
                        genre_ratings[genre].append((row['movieId'], row['rating']))
            
            # Compute genre-specific popularity
            for genre, item_ratings in genre_ratings.items():
                genre_df = pd.DataFrame(item_ratings, columns=['movieId', 'rating'])
                genre_pop = genre_df.groupby('movieId')['rating'].agg(['count', 'mean'])
                self.genre_popularity[genre] = genre_pop.to_dict('index')
        
        # Temporal popularity (recent trends)
        recent_cutoff = ratings_df['timestamp'].max() - 30 * 24 * 3600  # Last 30 days
        recent_ratings = ratings_df[ratings_df['timestamp'] >= recent_cutoff]
        
        if not recent_ratings.empty:
            recent_stats = recent_ratings.groupby('movieId')['rating'].agg(['count', 'mean'])
            for item_id, row in recent_stats.iterrows():
                self.temporal_popularity[item_id] = row['count'] * row['mean'] / 5.0
        
        logger.info(f"✅ Computed popularity for {len(self.global_popularity)} items")
    
    def get_popular_recommendations(self, n_recommendations: int = 10,
                                  user_profile: Optional[UserProfile] = None) -> List[ColdStartRecommendation]:
        """Get popularity-based recommendations"""
        
        recommendations = []
        
        # Choose popularity source based on user profile
        if user_profile and user_profile.preferred_genres:
            # Genre-based popularity
            genre_scores = defaultdict(float)
            for genre in user_profile.preferred_genres:
                if genre in self.genre_popularity:
                    for item_id, stats in self.genre_popularity[genre].items():
                        weight = 0.7 * (stats['count'] / 100) + 0.3 * (stats['mean'] / 5.0)
                        genre_scores[item_id] += weight / len(user_profile.preferred_genres)
            
            # Convert to recommendations
            sorted_items = sorted(genre_scores.items(), key=lambda x: x[1], reverse=True)
            for item_id, score in sorted_items[:n_recommendations]:
                recommendations.append(ColdStartRecommendation(
                    item_id=item_id,
                    score=score,
                    strategy=ColdStartStrategy.POPULARITY,
                    confidence=0.7,
                    explanation=f"Popular in preferred genres: {', '.join(user_profile.preferred_genres)}"
                ))
        
        else:
            # Global popularity
            sorted_items = sorted(self.global_popularity.items(), 
                                key=lambda x: x[1], reverse=True)
            
            for item_id, score in sorted_items[:n_recommendations]:
                recommendations.append(ColdStartRecommendation(
                    item_id=item_id,
                    score=score,
                    strategy=ColdStartStrategy.POPULARITY,
                    confidence=0.6,
                    explanation="Globally popular item"
                ))
        
        return recommendations

class ContentBasedRecommender:
    """Content-based recommendations using item features"""
    
    def __init__(self):
        self.item_features = {}
        self.feature_vectors = None
        self.item_ids = []
        self.scaler = StandardScaler()
    
    def fit(self, items_df: pd.DataFrame):
        """Fit content-based model"""
        logger.info("🎬 Building content-based model...")
        
        # Extract content features
        for _, row in items_df.iterrows():
            features = self._extract_item_features(row)
            self.item_features[row['movieId']] = features
        
        # Create feature matrix
        self.item_ids = list(self.item_features.keys())
        feature_matrix = np.array([self.item_features[item_id] for item_id in self.item_ids])
        
        # Normalize features
        self.feature_vectors = self.scaler.fit_transform(feature_matrix)
        
        logger.info(f"✅ Content model built for {len(self.item_ids)} items")
    
    def _extract_item_features(self, item_row: pd.Series) -> np.ndarray:
        """Extract numerical features from item"""
        features = []
        
        # Genre one-hot encoding
        genres = item_row.get('genres', '').split('|') if pd.notna(item_row.get('genres')) else []
        genre_features = [1 if genre in genres else 0 for genre in [
            'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
            'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
            'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
        ]]
        features.extend(genre_features)
        
        # Year features
        year = item_row.get('year', 2000)
        current_year = datetime.now().year
        age = current_year - year
        features.extend([
            year / current_year,  # Normalized year
            age / 100,           # Normalized age
            1 if year >= 2010 else 0,  # Recent movie
            1 if year >= 2000 else 0,  # Modern movie
        ])
        
        # Additional numerical features if available
        features.extend([
            item_row.get('popularity_score', 0.5),
            item_row.get('quality_score', 0.5),
            item_row.get('avg_rating', 3.5) / 5.0
        ])
        
        return np.array(features)
    
    def get_similar_items(self, reference_items: List[int], 
                         n_recommendations: int = 10) -> List[ColdStartRecommendation]:
        """Get content-based recommendations"""
        
        if not reference_items or self.feature_vectors is None:
            return []
        
        # Get reference item features
        reference_vectors = []
        valid_references = []
        
        for item_id in reference_items:
            if item_id in self.item_features:
                idx = self.item_ids.index(item_id)
                reference_vectors.append(self.feature_vectors[idx])
                valid_references.append(item_id)
        
        if not reference_vectors:
            return []
        
        # Compute average reference vector
        reference_vector = np.mean(reference_vectors, axis=0)
        
        # Compute similarities
        similarities = cosine_similarity([reference_vector], self.feature_vectors)[0]
        
        # Get top recommendations (excluding reference items)
        item_scores = [(self.item_ids[i], similarities[i]) 
                      for i in range(len(similarities))
                      if self.item_ids[i] not in reference_items]
        
        item_scores.sort(key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for item_id, score in item_scores[:n_recommendations]:
            recommendations.append(ColdStartRecommendation(
                item_id=item_id,
                score=score,
                strategy=ColdStartStrategy.CONTENT_BASED,
                confidence=0.8,
                explanation=f"Similar content to reference items"
            ))
        
        return recommendations

class OnboardingRecommender:
    """Interactive onboarding for new users"""
    
    def __init__(self):
        self.onboarding_items = {}
        self.genre_representatives = {}
    
    def setup_onboarding(self, items_df: pd.DataFrame, ratings_df: pd.DataFrame):
        """Setup onboarding item candidates"""
        logger.info("🎯 Setting up onboarding system...")
        
        # Select representative items for each genre
        merged_df = ratings_df.merge(items_df, on='movieId', how='left')
        
        for genre in ['Action', 'Comedy', 'Drama', 'Romance', 'Sci-Fi', 'Horror']:
            # Find popular, well-rated items in each genre
            genre_items = merged_df[merged_df['genres'].str.contains(genre, na=False)]
            
            if not genre_items.empty:
                item_stats = genre_items.groupby('movieId').agg({
                    'rating': ['count', 'mean'],
                    'title': 'first'
                })
                
                item_stats.columns = ['count', 'mean_rating', 'title']
                
                # Filter for quality and popularity
                quality_items = item_stats[
                    (item_stats['count'] >= 50) & 
                    (item_stats['mean_rating'] >= 3.5)
                ].sort_values(['mean_rating', 'count'], ascending=False)
                
                if not quality_items.empty:
                    top_item = quality_items.index[0]
                    self.genre_representatives[genre] = {
                        'item_id': top_item,
                        'title': quality_items.loc[top_item, 'title'],
                        'rating': quality_items.loc[top_item, 'mean_rating']
                    }
        
        logger.info(f"✅ Onboarding setup with {len(self.genre_representatives)} genre representatives")
    
    def get_onboarding_items(self, n_items: int = 12) -> List[Dict[str, Any]]:
        """Get diverse items for onboarding questionnaire"""
        
        onboarding_items = []
        
        # Include representative items from different genres
        for genre, item_info in self.genre_representatives.items():
            onboarding_items.append({
                'item_id': item_info['item_id'],
                'title': item_info['title'],
                'genre': genre,
                'purpose': 'genre_discovery'
            })
            
            if len(onboarding_items) >= n_items:
                break
        
        # Add some popular recent items
        # (In real implementation, would query recent popular items)
        
        return onboarding_items[:n_items]
    
    def process_onboarding_feedback(self, feedback: Dict[int, float]) -> UserProfile:
        """Process user feedback from onboarding"""
        
        # Analyze user preferences from feedback
        preferred_genres = []
        disliked_genres = []
        
        for item_id, rating in feedback.items():
            # Find item's genres (simplified)
            for genre, item_info in self.genre_representatives.items():
                if item_info['item_id'] == item_id:
                    if rating >= 4.0:
                        preferred_genres.append(genre)
                    elif rating <= 2.0:
                        disliked_genres.append(genre)
        
        # Create user profile
        user_profile = UserProfile(
            user_id=0,  # Will be set by caller
            preferred_genres=preferred_genres,
            disliked_genres=disliked_genres,
            rating_preferences=feedback,
            signup_date=datetime.now()
        )
        
        return user_profile

class UserClusteringRecommender:
    """User clustering for cold start recommendations"""
    
    def __init__(self, n_clusters: int = 10):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.user_clusters = {}
        self.cluster_recommendations = {}
    
    def fit(self, ratings_df: pd.DataFrame, users_df: Optional[pd.DataFrame] = None):
        """Fit user clustering model"""
        logger.info("👥 Building user clustering model...")
        
        # Create user-item interaction matrix
        user_item_matrix = ratings_df.pivot_table(
            index='userId', columns='movieId', values='rating', fill_value=0
        )
        
        # Fit clustering model
        user_features = user_item_matrix.values
        self.kmeans.fit(user_features)
        
        # Assign users to clusters
        for i, user_id in enumerate(user_item_matrix.index):
            cluster_id = self.kmeans.labels_[i]
            self.user_clusters[user_id] = cluster_id
        
        # Compute cluster-based recommendations
        for cluster_id in range(self.n_clusters):
            cluster_users = [uid for uid, cid in self.user_clusters.items() 
                           if cid == cluster_id]
            
            if cluster_users:
                # Get ratings from cluster users
                cluster_ratings = ratings_df[ratings_df['userId'].isin(cluster_users)]
                
                # Compute cluster preferences
                cluster_prefs = cluster_ratings.groupby('movieId')['rating'].agg([
                    'mean', 'count'
                ]).sort_values(['mean', 'count'], ascending=False)
                
                # Store top recommendations for cluster
                top_items = cluster_prefs.head(50).index.tolist()
                self.cluster_recommendations[cluster_id] = top_items
        
        logger.info(f"✅ User clustering completed with {self.n_clusters} clusters")
    
    def assign_user_to_cluster(self, user_profile: UserProfile) -> int:
        """Assign new user to cluster based on profile"""
        
        # Simplified cluster assignment based on preferences
        # In practice, would use demographic or initial preference features
        
        if user_profile.preferred_genres:
            # Use genre preferences to find similar cluster
            genre_vector = np.zeros(18)  # Assume 18 genres
            for i, genre in enumerate(['Action', 'Adventure', 'Animation', 'Children', 
                                     'Comedy', 'Crime', 'Documentary', 'Drama', 
                                     'Fantasy', 'Film-Noir', 'Horror', 'Musical',
                                     'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 
                                     'War', 'Western']):
                if genre in user_profile.preferred_genres:
                    genre_vector[i] = 1.0
            
            # Find most similar cluster centroid
            cluster_id = self.kmeans.predict([genre_vector])[0]
            return cluster_id
        
        # Default to random assignment
        return np.random.randint(0, self.n_clusters)
    
    def get_cluster_recommendations(self, cluster_id: int, 
                                  n_recommendations: int = 10) -> List[ColdStartRecommendation]:
        """Get recommendations for a user cluster"""
        
        if cluster_id not in self.cluster_recommendations:
            return []
        
        cluster_items = self.cluster_recommendations[cluster_id]
        recommendations = []
        
        for item_id in cluster_items[:n_recommendations]:
            recommendations.append(ColdStartRecommendation(
                item_id=item_id,
                score=0.6,  # Fixed confidence for cluster-based recs
                strategy=ColdStartStrategy.CLUSTERING,
                confidence=0.7,
                explanation=f"Popular in your user cluster (#{cluster_id})"
            ))
        
        return recommendations

class ColdStartHandler:
    """
    Comprehensive cold start handler combining multiple strategies
    """
    
    def __init__(self):
        self.popularity_recommender = PopularityRecommender()
        self.content_recommender = ContentBasedRecommender()
        self.onboarding_recommender = OnboardingRecommender()
        self.clustering_recommender = UserClusteringRecommender()
        
        # Interaction thresholds
        self.new_user_threshold = 5      # < 5 interactions = new user
        self.sparse_user_threshold = 20  # < 20 interactions = sparse user
        self.new_item_threshold = 10     # < 10 ratings = new item
        
        logger.info("🆕 Cold Start Handler initialized")
    
    def fit(self, ratings_df: pd.DataFrame, items_df: pd.DataFrame,
            users_df: Optional[pd.DataFrame] = None):
        """Fit all cold start models"""
        logger.info("🔧 Training cold start models...")
        
        # Fit popularity model
        self.popularity_recommender.fit(ratings_df, users_df, items_df)
        
        # Fit content-based model
        self.content_recommender.fit(items_df)
        
        # Setup onboarding
        self.onboarding_recommender.setup_onboarding(items_df, ratings_df)
        
        # Fit clustering model
        self.clustering_recommender.fit(ratings_df, users_df)
        
        logger.info("✅ All cold start models trained")
    
    def identify_cold_start_type(self, user_id: int, item_id: Optional[int] = None,
                               user_history: Optional[pd.DataFrame] = None,
                               item_history: Optional[pd.DataFrame] = None) -> ColdStartType:
        """Identify the type of cold start scenario"""
        
        user_interaction_count = len(user_history) if user_history is not None else 0
        item_rating_count = len(item_history) if item_history is not None else 0
        
        # Check user status
        is_new_user = user_interaction_count < self.new_user_threshold
        is_sparse_user = user_interaction_count < self.sparse_user_threshold
        
        # Check item status (if provided)
        if item_id is not None:
            is_new_item = item_rating_count < self.new_item_threshold
            
            if is_new_user and is_new_item:
                return ColdStartType.NEW_USER_NEW_ITEM
            elif is_new_user:
                return ColdStartType.NEW_USER
            elif is_new_item:
                return ColdStartType.NEW_ITEM
        
        # User-only analysis
        if is_new_user:
            return ColdStartType.NEW_USER
        elif is_sparse_user:
            return ColdStartType.SPARSE_USER
        
        # Not a cold start scenario
        return None
    
    def get_cold_start_recommendations(self, user_id: int, 
                                     user_profile: Optional[UserProfile] = None,
                                     user_history: Optional[pd.DataFrame] = None,
                                     n_recommendations: int = 10,
                                     strategy: ColdStartStrategy = ColdStartStrategy.HYBRID) -> List[ColdStartRecommendation]:
        """Get cold start recommendations using specified strategy"""
        
        recommendations = []
        
        if strategy == ColdStartStrategy.POPULARITY:
            recommendations = self.popularity_recommender.get_popular_recommendations(
                n_recommendations, user_profile
            )
        
        elif strategy == ColdStartStrategy.CONTENT_BASED and user_history is not None:
            # Use user's limited history for content-based recommendations
            reference_items = user_history['movieId'].tolist() if not user_history.empty else []
            recommendations = self.content_recommender.get_similar_items(
                reference_items, n_recommendations
            )
        
        elif strategy == ColdStartStrategy.ONBOARDING:
            # Return onboarding items (not recommendations)
            onboarding_items = self.onboarding_recommender.get_onboarding_items(n_recommendations)
            recommendations = [
                ColdStartRecommendation(
                    item_id=item['item_id'],
                    score=0.5,
                    strategy=ColdStartStrategy.ONBOARDING,
                    confidence=0.9,
                    explanation=f"Onboarding item for {item['genre']} genre discovery"
                ) for item in onboarding_items
            ]
        
        elif strategy == ColdStartStrategy.CLUSTERING and user_profile:
            cluster_id = self.clustering_recommender.assign_user_to_cluster(user_profile)
            recommendations = self.clustering_recommender.get_cluster_recommendations(
                cluster_id, n_recommendations
            )
        
        elif strategy == ColdStartStrategy.HYBRID:
            # Combine multiple strategies
            recommendations = self._get_hybrid_recommendations(
                user_id, user_profile, user_history, n_recommendations
            )
        
        return recommendations
    
    def _get_hybrid_recommendations(self, user_id: int, 
                                  user_profile: Optional[UserProfile],
                                  user_history: Optional[pd.DataFrame],
                                  n_recommendations: int) -> List[ColdStartRecommendation]:
        """Get hybrid recommendations combining multiple strategies"""
        
        all_recommendations = []
        
        # Get popularity-based recommendations (40%)
        pop_recs = self.popularity_recommender.get_popular_recommendations(
            max(1, n_recommendations // 2), user_profile
        )
        all_recommendations.extend(pop_recs[:max(1, n_recommendations // 2)])
        
        # Get content-based recommendations if possible (30%)
        if user_history is not None and not user_history.empty:
            reference_items = user_history['movieId'].tolist()
            content_recs = self.content_recommender.get_similar_items(
                reference_items, max(1, n_recommendations // 3)
            )
            all_recommendations.extend(content_recs)
        
        # Get clustering-based recommendations (30%)
        if user_profile:
            cluster_id = self.clustering_recommender.assign_user_to_cluster(user_profile)
            cluster_recs = self.clustering_recommender.get_cluster_recommendations(
                cluster_id, max(1, n_recommendations // 3)
            )
            all_recommendations.extend(cluster_recs)
        
        # Remove duplicates and re-rank
        seen_items = set()
        unique_recommendations = []
        
        for rec in all_recommendations:
            if rec.item_id not in seen_items:
                seen_items.add(rec.item_id)
                unique_recommendations.append(rec)
        
        # Sort by score and return top N
        unique_recommendations.sort(key=lambda x: x.score, reverse=True)
        
        return unique_recommendations[:n_recommendations]
    
    def handle_onboarding_flow(self, user_id: int) -> Dict[str, Any]:
        """Handle complete onboarding flow for new user"""
        
        # Get onboarding items
        onboarding_items = self.onboarding_recommender.get_onboarding_items(12)
        
        return {
            'user_id': user_id,
            'onboarding_items': onboarding_items,
            'instructions': 'Please rate these items to help us understand your preferences',
            'min_ratings_required': 5
        }
    
    def process_onboarding_completion(self, user_id: int, 
                                    ratings_feedback: Dict[int, float]) -> UserProfile:
        """Process completed onboarding and create user profile"""
        
        user_profile = self.onboarding_recommender.process_onboarding_feedback(ratings_feedback)
        user_profile.user_id = user_id
        
        logger.info(f"👤 Created user profile for {user_id} with {len(user_profile.preferred_genres or [])} preferred genres")
        
        return user_profile
    
    def get_explanation(self, recommendation: ColdStartRecommendation) -> str:
        """Get detailed explanation for a recommendation"""
        
        base_explanation = recommendation.explanation
        confidence_text = f" (Confidence: {recommendation.confidence:.0%})"
        
        strategy_context = {
            ColdStartStrategy.POPULARITY: "This recommendation is based on what's popular with other users.",
            ColdStartStrategy.CONTENT_BASED: "This recommendation matches the content of items you've shown interest in.",
            ColdStartStrategy.CLUSTERING: "This recommendation is popular among users with similar preferences to you.",
            ColdStartStrategy.HYBRID: "This recommendation combines multiple approaches for the best match."
        }
        
        context = strategy_context.get(recommendation.strategy, "")
        
        return f"{base_explanation} {context}{confidence_text}"

def main():
    """Test the cold start handler"""
    print("🆕 Testing Cold Start Handler...")
    
    # Create sample data
    np.random.seed(42)
    
    # Sample ratings data
    n_users = 200
    n_items = 100
    n_ratings = 2000
    
    ratings_data = {
        'userId': np.random.choice(range(1, n_users + 1), n_ratings),
        'movieId': np.random.choice(range(1, n_items + 1), n_ratings),
        'rating': np.random.choice([1, 2, 3, 4, 5], n_ratings, p=[0.1, 0.1, 0.2, 0.3, 0.3]),
        'timestamp': np.random.randint(1000000000, 1700000000, n_ratings)
    }
    ratings_df = pd.DataFrame(ratings_data)
    
    # Sample movies data
    genres_list = ['Action', 'Comedy', 'Drama', 'Romance', 'Sci-Fi', 'Horror', 'Thriller', 'Animation']
    movies_data = {
        'movieId': range(1, n_items + 1),
        'title': [f'Movie {i}' for i in range(1, n_items + 1)],
        'genres': [np.random.choice(genres_list, np.random.randint(1, 4)).tolist() for _ in range(n_items)],
        'year': np.random.choice(range(1990, 2024), n_items)
    }
    
    # Convert genre lists to strings
    movies_data['genres'] = ['|'.join(genres) for genres in movies_data['genres']]
    movies_df = pd.DataFrame(movies_data)
    
    # Initialize and fit cold start handler
    cold_start_handler = ColdStartHandler()
    cold_start_handler.fit(ratings_df, movies_df)
    
    # Test new user scenario
    new_user_id = 999
    print(f"\n👤 Testing new user scenario (ID: {new_user_id})")
    
    # Create user profile
    user_profile = UserProfile(
        user_id=new_user_id,
        preferred_genres=['Action', 'Sci-Fi'],
        disliked_genres=['Horror'],
        signup_date=datetime.now()
    )
    
    # Get cold start recommendations
    recommendations = cold_start_handler.get_cold_start_recommendations(
        user_id=new_user_id,
        user_profile=user_profile,
        n_recommendations=5,
        strategy=ColdStartStrategy.HYBRID
    )
    
    print(f"📋 Cold start recommendations:")
    for i, rec in enumerate(recommendations, 1):
        explanation = cold_start_handler.get_explanation(rec)
        print(f"{i}. Item {rec.item_id}: {rec.score:.3f} - {explanation}")
    
    # Test onboarding flow
    print(f"\n🎯 Testing onboarding flow")
    onboarding_data = cold_start_handler.handle_onboarding_flow(new_user_id)
    print(f"Onboarding items: {len(onboarding_data['onboarding_items'])}")
    
    # Simulate onboarding feedback
    feedback = {}
    for item in onboarding_data['onboarding_items'][:5]:
        feedback[item['item_id']] = np.random.uniform(1, 5)
    
    # Process onboarding
    processed_profile = cold_start_handler.process_onboarding_completion(new_user_id, feedback)
    print(f"User profile created with preferences: {processed_profile.preferred_genres}")
    
    # Test cold start type identification
    user_history = ratings_df[ratings_df['userId'] == 1].head(3)  # Sparse user
    cold_start_type = cold_start_handler.identify_cold_start_type(
        user_id=1, user_history=user_history
    )
    print(f"\n🔍 Cold start type for user 1: {cold_start_type}")
    
    print("\n✅ Cold start handler ready!")

if __name__ == "__main__":
    main()