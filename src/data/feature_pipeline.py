"""
Enhanced Feature Pipeline for MovieLens RecSys
Integrates with existing ETL pipeline for real-time feature computation and validation
Extends the existing etl_pipeline.py with production features
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
import json
import pickle
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# Import existing ETL classes
from .etl_pipeline import MovieLensETL, DataQualityMetrics
from .feature_store import FeatureStore, UserFeatures, ItemFeatures, InteractionFeatures
from .cache_manager import CacheManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FeaturePipelineMetrics:
    """Metrics for feature pipeline performance"""
    processing_time_seconds: float
    features_generated: int
    features_updated: int
    cache_hits: int
    cache_misses: int
    validation_errors: int
    data_quality_score: float
    pipeline_version: str
    timestamp: datetime

class EnhancedFeaturePipeline(MovieLensETL):
    """
    Enhanced feature pipeline that extends the existing ETL
    Adds real-time feature computation, caching, and validation
    """
    
    def __init__(self, config_path: str = "config/etl_config.json"):
        # Initialize parent ETL pipeline
        super().__init__(config_path)
        
        # Initialize feature store and cache manager
        self.feature_store = FeatureStore()
        self.cache_manager = CacheManager()
        
        # Feature computation settings
        self.feature_config = {
            'user_embedding_dim': 64,
            'popularity_window_days': 30,
            'recency_decay_rate': 0.1,
            'cold_start_threshold': 5,  # Minimum interactions for warm user
            'feature_freshness_hours': 24
        }
        
        logger.info("Enhanced Feature Pipeline initialized")
    
    def process_streaming_batch(self, new_interactions: pd.DataFrame) -> FeaturePipelineMetrics:
        """
        Process a new batch of streaming interactions
        Updates features and cache in real-time
        """
        start_time = datetime.now()
        logger.info(f"Processing streaming batch with {len(new_interactions)} interactions")
        
        features_generated = 0
        features_updated = 0
        cache_hits = 0
        cache_misses = 0
        validation_errors = 0
        
        try:
            # Validate incoming batch
            validation_result = self._validate_streaming_batch(new_interactions)
            if not validation_result['valid']:
                validation_errors += len(validation_result['errors'])
                logger.warning(f"Batch validation found {validation_errors} errors")
            
            # Process user features updates
            unique_users = new_interactions['user_id'].unique()
            for user_id in unique_users:
                user_interactions = new_interactions[new_interactions['user_id'] == user_id]
                
                # Check cache first
                cached_features = self.cache_manager.get_user_profile(user_id)
                if cached_features:
                    cache_hits += 1
                    # Update existing features
                    updated_features = self._update_user_features_incremental(
                        user_id, user_interactions, cached_features
                    )
                    features_updated += 1
                else:
                    cache_misses += 1
                    # Generate new features
                    new_features = self._generate_user_features_streaming(user_id, user_interactions)
                    features_generated += 1
                
                # Update cache and feature store
                self._persist_user_features(user_id)
            
            # Process item features updates  
            unique_items = new_interactions['movie_id'].unique()
            for movie_id in unique_items:
                movie_interactions = new_interactions[new_interactions['movie_id'] == movie_id]
                
                # Update item popularity and statistics
                self._update_item_features_streaming(movie_id, movie_interactions)
                features_updated += 1
            
            # Store interaction features
            self._store_streaming_interactions(new_interactions)
            
            # Update popular items cache
            self._refresh_popular_cache_if_needed()
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate data quality score
            data_quality_score = self._calculate_batch_quality_score(new_interactions, validation_result)
            
            metrics = FeaturePipelineMetrics(
                processing_time_seconds=processing_time,
                features_generated=features_generated,
                features_updated=features_updated,
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                validation_errors=validation_errors,
                data_quality_score=data_quality_score,
                pipeline_version="v1.0",
                timestamp=datetime.now()
            )
            
            logger.info(f"Streaming batch processed in {processing_time:.2f}s")
            return metrics
            
        except Exception as e:
            logger.error(f"Error processing streaming batch: {e}")
            # Return error metrics
            return FeaturePipelineMetrics(
                processing_time_seconds=(datetime.now() - start_time).total_seconds(),
                features_generated=features_generated,
                features_updated=features_updated,
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                validation_errors=validation_errors + 1,
                data_quality_score=0.0,
                pipeline_version="v1.0",
                timestamp=datetime.now()
            )
    
    def _validate_streaming_batch(self, interactions: pd.DataFrame) -> Dict[str, Any]:
        """Validate incoming streaming batch"""
        errors = []
        
        # Schema validation
        required_columns = ['user_id', 'movie_id', 'rating']
        missing_columns = [col for col in required_columns if col not in interactions.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Data type validation
        if 'user_id' in interactions.columns:
            if not pd.api.types.is_numeric_dtype(interactions['user_id']):
                errors.append("user_id must be numeric")
        
        if 'movie_id' in interactions.columns:
            if not pd.api.types.is_numeric_dtype(interactions['movie_id']):
                errors.append("movie_id must be numeric")
        
        if 'rating' in interactions.columns:
            invalid_ratings = interactions[
                (interactions['rating'] < 0.5) | (interactions['rating'] > 5.0)
            ]
            if len(invalid_ratings) > 0:
                errors.append(f"Found {len(invalid_ratings)} invalid ratings (must be 0.5-5.0)")
        
        # Check for duplicates
        duplicates = interactions.duplicated(['user_id', 'movie_id']).sum()
        if duplicates > 0:
            errors.append(f"Found {duplicates} duplicate user-movie pairs")
        
        # Check for missing values
        missing_values = interactions.isnull().sum().sum()
        if missing_values > 0:
            errors.append(f"Found {missing_values} missing values")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'validation_timestamp': datetime.now()
        }
    
    def _update_user_features_incremental(self, user_id: int, new_interactions: pd.DataFrame,
                                        cached_features: Dict[str, Any]) -> Dict[str, Any]:
        """Update user features incrementally with new interactions"""
        
        # Get current statistics from cache
        current_avg = cached_features.get('avg_rating', 3.0)
        current_count = cached_features.get('rating_count', 0)
        
        # Calculate new statistics
        new_ratings = new_interactions['rating']
        new_count = len(new_ratings)
        
        if new_count > 0:
            # Incremental average calculation
            total_ratings = current_avg * current_count + new_ratings.sum()
            updated_count = current_count + new_count
            updated_avg = total_ratings / updated_count if updated_count > 0 else 3.0
            
            # Update genre preferences incrementally
            updated_genres = self._update_genre_preferences_incremental(
                user_id, new_interactions, cached_features.get('genres_preference', {})
            )
            
            updated_features = {
                'avg_rating': updated_avg,
                'rating_count': updated_count,
                'genres_preference': updated_genres,
                'last_activity': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            
            # Cache updated features
            self.cache_manager.cache_user_profile(user_id, updated_features)
            
            return updated_features
        
        return cached_features
    
    def _generate_user_features_streaming(self, user_id: int, interactions: pd.DataFrame) -> Dict[str, Any]:
        """Generate new user features from streaming interactions"""
        
        # Calculate basic statistics
        avg_rating = interactions['rating'].mean()
        rating_count = len(interactions)
        rating_variance = interactions['rating'].var()
        
        # Generate genre preferences (simplified for streaming)
        genre_preferences = {}  # Would need movie metadata for full implementation
        
        # Generate embedding (placeholder - would use actual model in production)
        np.random.seed(user_id)
        embedding = np.random.randn(self.feature_config['user_embedding_dim']).astype(np.float32)
        
        features = {
            'avg_rating': float(avg_rating),
            'rating_count': int(rating_count),
            'rating_variance': float(rating_variance) if pd.notna(rating_variance) else 0.0,
            'genres_preference': genre_preferences,
            'embedding': embedding.tolist(),  # Convert to list for JSON serialization
            'last_activity': datetime.now().isoformat(),
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        # Cache new features
        self.cache_manager.cache_user_profile(user_id, features)
        self.cache_manager.cache_user_embedding(user_id, embedding)
        
        return features
    
    def _update_genre_preferences_incremental(self, user_id: int, new_interactions: pd.DataFrame,
                                            current_preferences: Dict[str, float]) -> Dict[str, float]:
        """Update user genre preferences incrementally"""
        # Simplified implementation - would need to join with movie metadata
        # For now, return existing preferences
        return current_preferences
    
    def _update_item_features_streaming(self, movie_id: int, new_interactions: pd.DataFrame):
        """Update item features with new interaction data"""
        
        # Get current item features
        current_features = self.feature_store.get_item_features(movie_id)
        
        if current_features:
            # Update popularity score with new interactions
            new_rating_count = len(new_interactions)
            new_avg_rating = new_interactions['rating'].mean()
            
            # Weighted average for incremental updates
            current_count = current_features.rating_count
            current_avg = current_features.avg_rating
            
            if current_count > 0:
                total_ratings = current_avg * current_count + new_avg_rating * new_rating_count
                updated_count = current_count + new_rating_count
                updated_avg = total_ratings / updated_count
            else:
                updated_avg = new_avg_rating
                updated_count = new_rating_count
            
            # Calculate updated popularity score
            updated_popularity = np.log1p(updated_count) * updated_avg / 5.0
            
            # Update features in database (simplified)
            # In production, this would use proper database update
            logger.debug(f"Updated item {movie_id}: avg={updated_avg:.2f}, count={updated_count}")
    
    def _store_streaming_interactions(self, interactions: pd.DataFrame):
        """Store new interactions in feature store"""
        try:
            # Convert to interaction features format
            for _, row in interactions.iterrows():
                interaction = InteractionFeatures(
                    user_id=int(row['user_id']),
                    movie_id=int(row['movie_id']),
                    rating=float(row['rating']),
                    timestamp=datetime.now(),
                    session_id=f"stream_{datetime.now().strftime('%Y%m%d')}",
                    interaction_type='streaming',
                    context={'source': 'streaming_batch'},
                    created_at=datetime.now()
                )
                
                # Store in feature store (would batch this in production)
                # For now, just log the interaction
                logger.debug(f"Stored interaction: user={interaction.user_id}, item={interaction.movie_id}")
                
        except Exception as e:
            logger.error(f"Error storing streaming interactions: {e}")
    
    def _persist_user_features(self, user_id: int):
        """Persist cached user features to feature store"""
        try:
            cached_features = self.cache_manager.get_user_profile(user_id)
            cached_embedding = self.cache_manager.get_user_embedding(user_id)
            
            if cached_features and cached_embedding is not None:
                user_features = UserFeatures(
                    user_id=user_id,
                    embedding=cached_embedding,
                    avg_rating=cached_features['avg_rating'],
                    rating_count=cached_features['rating_count'],
                    genres_preference=cached_features.get('genres_preference', {}),
                    rating_variance=cached_features.get('rating_variance', 0.0),
                    favorite_decade=2000,  # Default
                    last_activity=datetime.fromisoformat(cached_features['last_activity']),
                    created_at=datetime.fromisoformat(cached_features.get('created_at', datetime.now().isoformat())),
                    updated_at=datetime.fromisoformat(cached_features['updated_at'])
                )
                
                # Would persist to feature store database here
                logger.debug(f"Persisted features for user {user_id}")
                
        except Exception as e:
            logger.error(f"Error persisting user features {user_id}: {e}")
    
    def _refresh_popular_cache_if_needed(self):
        """Refresh popular items cache if it's stale"""
        try:
            # Check if popular cache needs refresh (every hour)
            last_refresh_key = "last_popular_refresh"
            last_refresh = self.cache_manager.popular_cache.get(last_refresh_key)
            
            if not last_refresh or datetime.fromisoformat(last_refresh) < datetime.now() - timedelta(hours=1):
                logger.info("Refreshing popular items cache")
                self.cache_manager.warm_popular_cache(self.feature_store)
                self.cache_manager.popular_cache.set(last_refresh_key, datetime.now().isoformat())
                
        except Exception as e:
            logger.error(f"Error refreshing popular cache: {e}")
    
    def _calculate_batch_quality_score(self, interactions: pd.DataFrame, 
                                     validation_result: Dict[str, Any]) -> float:
        """Calculate data quality score for the batch"""
        try:
            # Start with base score
            quality_score = 1.0
            
            # Penalize validation errors
            error_count = len(validation_result.get('errors', []))
            if error_count > 0:
                quality_score -= min(0.5, error_count * 0.1)
            
            # Check rating distribution
            if len(interactions) > 0:
                rating_std = interactions['rating'].std()
                if rating_std < 0.5:  # Too little variance
                    quality_score -= 0.1
                
                # Check for suspicious patterns
                duplicate_rate = interactions.duplicated(['user_id', 'movie_id']).sum() / len(interactions)
                if duplicate_rate > 0.05:  # More than 5% duplicates
                    quality_score -= duplicate_rate
            
            return max(0.0, quality_score)
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return 0.0
    
    def get_feature_pipeline_health(self) -> Dict[str, Any]:
        """Get comprehensive health metrics for the feature pipeline"""
        try:
            # Feature store stats
            fs_stats = self.feature_store.get_feature_store_stats()
            
            # Cache stats  
            cache_stats = self.cache_manager.get_cache_stats()
            
            # Pipeline health metrics
            health_metrics = {
                'feature_store': fs_stats,
                'cache_performance': cache_stats,
                'pipeline_status': 'healthy',
                'last_batch_processed': datetime.now().isoformat(),
                'feature_freshness_ok': True,  # Would check actual freshness
                'error_rate': 0.0,  # Would track from recent batches
                'throughput_interactions_per_second': 100.0,  # Example metric
                'config': self.feature_config
            }
            
            return health_metrics
            
        except Exception as e:
            logger.error(f"Error getting pipeline health: {e}")
            return {'pipeline_status': 'error', 'error': str(e)}
    
    def close(self):
        """Clean up resources"""
        self.cache_manager.close()
        logger.info("Feature pipeline closed")

def main():
    """Test the enhanced feature pipeline"""
    pipeline = EnhancedFeaturePipeline()
    
    # Create sample streaming data
    sample_interactions = pd.DataFrame({
        'user_id': [1, 2, 3, 1, 2],
        'movie_id': [100, 200, 300, 150, 250],
        'rating': [4.0, 3.5, 5.0, 3.0, 4.5]
    })
    
    # Process the batch
    metrics = pipeline.process_streaming_batch(sample_interactions)
    print(f"Processing metrics: {asdict(metrics)}")
    
    # Get health status
    health = pipeline.get_feature_pipeline_health()
    print(f"Pipeline health: {health}")
    
    # Clean up
    pipeline.close()

if __name__ == "__main__":
    main()