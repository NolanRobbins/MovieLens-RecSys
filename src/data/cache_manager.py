"""
Production Caching System for MovieLens RecSys
Diskcache-based persistent caching (Redis alternative)
Optimized for recommendation system access patterns
"""

import diskcache as dc
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging
import hashlib
import json
from functools import wraps

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CacheManager:
    """
    Production-grade caching system using diskcache
    Handles user profiles, recommendations, and popular items
    """
    
    def __init__(self, cache_dir: str = "data/processed/cache", 
                 default_ttl: int = 3600):  # 1 hour default TTL
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.default_ttl = default_ttl
        
        # Initialize separate cache stores for different data types
        self.user_cache = dc.Cache(str(self.cache_dir / "users"), size_limit=100 * 1024**2)  # 100MB
        self.item_cache = dc.Cache(str(self.cache_dir / "items"), size_limit=100 * 1024**2)  # 100MB  
        self.recommendation_cache = dc.Cache(str(self.cache_dir / "recommendations"), size_limit=500 * 1024**2)  # 500MB
        self.popular_cache = dc.Cache(str(self.cache_dir / "popular"), size_limit=50 * 1024**2)  # 50MB
        
        logger.info(f"CacheManager initialized at {self.cache_dir}")
    
    def cache_user_profile(self, user_id: int, profile_data: Dict[str, Any], 
                          ttl: Optional[int] = None) -> bool:
        """Cache user profile with TTL"""
        try:
            cache_key = f"user_profile_{user_id}"
            ttl = ttl or self.default_ttl
            
            cache_data = {
                'data': profile_data,
                'cached_at': datetime.now().isoformat(),
                'ttl': ttl
            }
            
            return self.user_cache.set(cache_key, cache_data, expire=ttl)
        except Exception as e:
            logger.error(f"Error caching user profile {user_id}: {e}")
            return False
    
    def get_user_profile(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve cached user profile"""
        try:
            cache_key = f"user_profile_{user_id}"
            cached_data = self.user_cache.get(cache_key)
            
            if cached_data:
                return cached_data['data']
            return None
        except Exception as e:
            logger.error(f"Error retrieving user profile {user_id}: {e}")
            return None
    
    def cache_user_embedding(self, user_id: int, embedding: np.ndarray, 
                           ttl: Optional[int] = None) -> bool:
        """Cache user embedding vector"""
        try:
            cache_key = f"user_embedding_{user_id}"
            ttl = ttl or (self.default_ttl * 24)  # Embeddings cached longer
            
            # Serialize numpy array
            embedding_bytes = pickle.dumps(embedding.astype(np.float32))
            
            cache_data = {
                'embedding': embedding_bytes,
                'shape': embedding.shape,
                'dtype': str(embedding.dtype),
                'cached_at': datetime.now().isoformat()
            }
            
            return self.user_cache.set(cache_key, cache_data, expire=ttl)
        except Exception as e:
            logger.error(f"Error caching user embedding {user_id}: {e}")
            return False
    
    def get_user_embedding(self, user_id: int) -> Optional[np.ndarray]:
        """Retrieve cached user embedding"""
        try:
            cache_key = f"user_embedding_{user_id}"
            cached_data = self.user_cache.get(cache_key)
            
            if cached_data:
                return pickle.loads(cached_data['embedding'])
            return None
        except Exception as e:
            logger.error(f"Error retrieving user embedding {user_id}: {e}")
            return None
    
    def cache_recommendations(self, user_id: int, recommendations: List[Dict], 
                            model_version: str = "v1", ttl: Optional[int] = None) -> bool:
        """Cache user recommendations with model version"""
        try:
            cache_key = f"recs_{user_id}_{model_version}"
            ttl = ttl or 1800  # 30 minutes for recommendations
            
            cache_data = {
                'recommendations': recommendations,
                'model_version': model_version,
                'generated_at': datetime.now().isoformat(),
                'user_id': user_id,
                'count': len(recommendations)
            }
            
            return self.recommendation_cache.set(cache_key, cache_data, expire=ttl)
        except Exception as e:
            logger.error(f"Error caching recommendations for user {user_id}: {e}")
            return False
    
    def get_recommendations(self, user_id: int, model_version: str = "v1") -> Optional[List[Dict]]:
        """Retrieve cached recommendations"""
        try:
            cache_key = f"recs_{user_id}_{model_version}"
            cached_data = self.recommendation_cache.get(cache_key)
            
            if cached_data:
                return cached_data['recommendations']
            return None
        except Exception as e:
            logger.error(f"Error retrieving recommendations for user {user_id}: {e}")
            return None
    
    def cache_popular_items(self, items: List[Dict], category: str = "global", 
                          ttl: Optional[int] = None) -> bool:
        """Cache popular items by category (global, genre, decade, etc.)"""
        try:
            cache_key = f"popular_{category}"
            ttl = ttl or (self.default_ttl * 6)  # 6 hours for popular items
            
            cache_data = {
                'items': items,
                'category': category,
                'cached_at': datetime.now().isoformat(),
                'count': len(items)
            }
            
            return self.popular_cache.set(cache_key, cache_data, expire=ttl)
        except Exception as e:
            logger.error(f"Error caching popular items for {category}: {e}")
            return False
    
    def get_popular_items(self, category: str = "global") -> Optional[List[Dict]]:
        """Retrieve cached popular items"""
        try:
            cache_key = f"popular_{category}"
            cached_data = self.popular_cache.get(cache_key)
            
            if cached_data:
                return cached_data['items']
            return None
        except Exception as e:
            logger.error(f"Error retrieving popular items for {category}: {e}")
            return None
    
    def cache_item_features(self, movie_id: int, features: Dict[str, Any], 
                          ttl: Optional[int] = None) -> bool:
        """Cache movie features"""
        try:
            cache_key = f"item_features_{movie_id}"
            ttl = ttl or (self.default_ttl * 48)  # 48 hours for item features
            
            cache_data = {
                'features': features,
                'movie_id': movie_id,
                'cached_at': datetime.now().isoformat()
            }
            
            return self.item_cache.set(cache_key, cache_data, expire=ttl)
        except Exception as e:
            logger.error(f"Error caching item features {movie_id}: {e}")
            return False
    
    def get_item_features(self, movie_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve cached item features"""
        try:
            cache_key = f"item_features_{movie_id}"
            cached_data = self.item_cache.get(cache_key)
            
            if cached_data:
                return cached_data['features']
            return None
        except Exception as e:
            logger.error(f"Error retrieving item features {movie_id}: {e}")
            return None
    
    def invalidate_user_cache(self, user_id: int) -> bool:
        """Invalidate all cached data for a user"""
        try:
            patterns = [
                f"user_profile_{user_id}",
                f"user_embedding_{user_id}",
                f"recs_{user_id}_*"
            ]
            
            success = True
            for pattern in patterns:
                if pattern.endswith('*'):
                    # Handle wildcard patterns
                    prefix = pattern[:-1]
                    for key in list(self.user_cache.iterkeys()):
                        if key.startswith(prefix):
                            success &= self.user_cache.delete(key)
                    for key in list(self.recommendation_cache.iterkeys()):
                        if key.startswith(prefix):
                            success &= self.recommendation_cache.delete(key)
                else:
                    success &= self.user_cache.delete(pattern)
            
            logger.info(f"Invalidated cache for user {user_id}")
            return success
        except Exception as e:
            logger.error(f"Error invalidating user cache {user_id}: {e}")
            return False
    
    def warm_popular_cache(self, feature_store) -> bool:
        """Warm up popular items cache from feature store"""
        try:
            logger.info("Warming popular items cache...")
            
            # Get popular movies overall
            popular_movies = self._get_popular_movies_from_db(feature_store, limit=100)
            self.cache_popular_items(popular_movies, "global")
            
            # Get popular by genre
            genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi']
            for genre in genres:
                genre_popular = self._get_popular_by_genre(feature_store, genre, limit=50)
                self.cache_popular_items(genre_popular, f"genre_{genre}")
            
            # Get popular by decade
            decades = [1990, 2000, 2010, 2020]
            for decade in decades:
                decade_popular = self._get_popular_by_decade(feature_store, decade, limit=50)
                self.cache_popular_items(decade_popular, f"decade_{decade}")
            
            logger.info("Popular items cache warmed successfully")
            return True
        except Exception as e:
            logger.error(f"Error warming popular cache: {e}")
            return False
    
    def _get_popular_movies_from_db(self, feature_store, limit: int = 100) -> List[Dict]:
        """Get popular movies from feature store"""
        import sqlite3
        
        with sqlite3.connect(feature_store.db_path) as conn:
            cursor = conn.execute("""
                SELECT movie_id, title, genres, avg_rating, rating_count, popularity_score
                FROM item_features
                ORDER BY popularity_score DESC, rating_count DESC
                LIMIT ?
            """, (limit,))
            
            movies = []
            for row in cursor.fetchall():
                movies.append({
                    'movie_id': row[0],
                    'title': row[1],
                    'genres': json.loads(row[2]),
                    'avg_rating': row[3],
                    'rating_count': row[4],
                    'popularity_score': row[5]
                })
            
            return movies
    
    def _get_popular_by_genre(self, feature_store, genre: str, limit: int = 50) -> List[Dict]:
        """Get popular movies by genre"""
        import sqlite3
        
        with sqlite3.connect(feature_store.db_path) as conn:
            cursor = conn.execute("""
                SELECT movie_id, title, genres, avg_rating, rating_count, popularity_score
                FROM item_features
                WHERE genres LIKE ?
                ORDER BY popularity_score DESC, rating_count DESC
                LIMIT ?
            """, (f'%"{genre}"%', limit))
            
            movies = []
            for row in cursor.fetchall():
                movies.append({
                    'movie_id': row[0],
                    'title': row[1],
                    'genres': json.loads(row[2]),
                    'avg_rating': row[3],
                    'rating_count': row[4],
                    'popularity_score': row[5]
                })
            
            return movies
    
    def _get_popular_by_decade(self, feature_store, decade: int, limit: int = 50) -> List[Dict]:
        """Get popular movies by decade"""
        import sqlite3
        
        decade_start = decade
        decade_end = decade + 9
        
        with sqlite3.connect(feature_store.db_path) as conn:
            cursor = conn.execute("""
                SELECT movie_id, title, genres, avg_rating, rating_count, popularity_score, release_year
                FROM item_features
                WHERE release_year BETWEEN ? AND ?
                ORDER BY popularity_score DESC, rating_count DESC
                LIMIT ?
            """, (decade_start, decade_end, limit))
            
            movies = []
            for row in cursor.fetchall():
                movies.append({
                    'movie_id': row[0],
                    'title': row[1],
                    'genres': json.loads(row[2]),
                    'avg_rating': row[3],
                    'rating_count': row[4],
                    'popularity_score': row[5],
                    'release_year': row[6]
                })
            
            return movies
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        try:
            stats = {
                'user_cache': {
                    'size': len(self.user_cache),
                    'volume': self.user_cache.volume(),
                    'hits': self.user_cache.stats(enable=True)[0],
                    'misses': self.user_cache.stats(enable=True)[1]
                },
                'item_cache': {
                    'size': len(self.item_cache),
                    'volume': self.item_cache.volume(),
                    'hits': self.item_cache.stats(enable=True)[0],
                    'misses': self.item_cache.stats(enable=True)[1]
                },
                'recommendation_cache': {
                    'size': len(self.recommendation_cache),
                    'volume': self.recommendation_cache.volume(),
                    'hits': self.recommendation_cache.stats(enable=True)[0],
                    'misses': self.recommendation_cache.stats(enable=True)[1]
                },
                'popular_cache': {
                    'size': len(self.popular_cache),
                    'volume': self.popular_cache.volume(),
                    'hits': self.popular_cache.stats(enable=True)[0],
                    'misses': self.popular_cache.stats(enable=True)[1]
                },
                'total_size_mb': sum([
                    self.user_cache.volume(),
                    self.item_cache.volume(), 
                    self.recommendation_cache.volume(),
                    self.popular_cache.volume()
                ]) / 1024 / 1024,
                'last_updated': datetime.now().isoformat()
            }
            
            return stats
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}
    
    def clear_all_caches(self) -> bool:
        """Clear all caches - use with caution"""
        try:
            self.user_cache.clear()
            self.item_cache.clear()
            self.recommendation_cache.clear()
            self.popular_cache.clear()
            
            logger.info("All caches cleared")
            return True
        except Exception as e:
            logger.error(f"Error clearing caches: {e}")
            return False
    
    def close(self):
        """Close all cache connections"""
        try:
            self.user_cache.close()
            self.item_cache.close()
            self.recommendation_cache.close()
            self.popular_cache.close()
            logger.info("All cache connections closed")
        except Exception as e:
            logger.error(f"Error closing cache connections: {e}")

# Decorator for caching function results
def cache_result(cache_manager: CacheManager, cache_type: str = "general", 
                ttl: Optional[int] = None):
    """Decorator to cache function results"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}_{hashlib.md5(str(args + tuple(sorted(kwargs.items()))).encode()).hexdigest()[:8]}"
            
            # Try to get from cache
            if cache_type == "recommendation":
                cached_result = cache_manager.recommendation_cache.get(cache_key)
            elif cache_type == "popular":
                cached_result = cache_manager.popular_cache.get(cache_key)
            else:
                cached_result = cache_manager.user_cache.get(cache_key)
            
            if cached_result:
                return cached_result['result']
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            
            cache_data = {
                'result': result,
                'cached_at': datetime.now().isoformat(),
                'function': func.__name__
            }
            
            ttl_value = ttl or cache_manager.default_ttl
            
            if cache_type == "recommendation":
                cache_manager.recommendation_cache.set(cache_key, cache_data, expire=ttl_value)
            elif cache_type == "popular":
                cache_manager.popular_cache.set(cache_key, cache_data, expire=ttl_value)
            else:
                cache_manager.user_cache.set(cache_key, cache_data, expire=ttl_value)
            
            return result
        return wrapper
    return decorator

def main():
    """Test cache manager functionality"""
    from feature_store import FeatureStore
    
    # Initialize cache and feature store
    cache_manager = CacheManager()
    feature_store = FeatureStore()
    
    # Test caching
    test_user_id = 1
    test_profile = {'avg_rating': 3.5, 'rating_count': 100}
    
    # Cache user profile
    cache_manager.cache_user_profile(test_user_id, test_profile)
    
    # Retrieve and verify
    cached_profile = cache_manager.get_user_profile(test_user_id)
    print(f"Cached profile: {cached_profile}")
    
    # Warm popular cache
    cache_manager.warm_popular_cache(feature_store)
    
    # Print cache stats
    stats = cache_manager.get_cache_stats()
    print(f"Cache statistics: {stats}")

if __name__ == "__main__":
    main()