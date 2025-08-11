"""
ETL Pipeline for MovieLens Recommendation System
Handles new data ingestion, validation, and model updates
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, Tuple, Optional, List
import hashlib
import json
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DataQualityMetrics:
    """Data quality metrics for monitoring"""
    total_ratings: int
    unique_users: int
    unique_movies: int
    rating_range: Tuple[float, float]
    duplicate_count: int
    missing_values: Dict[str, int]
    data_freshness_days: float
    validation_passed: bool
    timestamp: datetime

class MovieLensETL:
    """
    Production ETL pipeline for MovieLens data
    Handles incremental updates, data validation, and feature engineering
    """
    
    def __init__(self, config_path: str = "etl_config.json"):
        self.config = self._load_config(config_path)
        self.data_dir = Path(self.config.get('data_dir', './data'))
        self.output_dir = Path(self.config.get('output_dir', './processed_data'))
        self.output_dir.mkdir(exist_ok=True)
        
        # Data quality thresholds
        self.min_ratings_per_user = self.config.get('min_ratings_per_user', 10)
        self.min_ratings_per_movie = self.config.get('min_ratings_per_movie', 5)
        self.max_duplicate_rate = self.config.get('max_duplicate_rate', 0.01)
        
        # Initialize mappings
        self.user_mappings = {}
        self.movie_mappings = {}
        self.reverse_user_mappings = {}
        self.reverse_movie_mappings = {}
        
    def _load_config(self, config_path: str) -> Dict:
        """Load ETL configuration"""
        default_config = {
            'data_dir': './ml-32m',
            'output_dir': './processed_data',
            'min_ratings_per_user': 10,
            'min_ratings_per_movie': 5,
            'max_duplicate_rate': 0.01,
            'train_ratio': 0.64,
            'val_ratio': 0.16,
            'test_ratio': 0.20
        }
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return {**default_config, **config}
        except FileNotFoundError:
            logger.info(f"Config file {config_path} not found. Using defaults.")
            return default_config
    
    def extract_data(self, data_source: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Extract data from source (local files or API)
        
        Args:
            data_source: Path to data directory or API endpoint
            
        Returns:
            ratings_df, movies_df, tags_df
        """
        logger.info("Starting data extraction...")
        
        source_path = Path(data_source) if data_source else self.data_dir
        
        # Load core datasets
        ratings_df = pd.read_csv(source_path / 'ratings.csv')
        movies_df = pd.read_csv(source_path / 'movies.csv')
        
        # Tags are optional
        tags_path = source_path / 'tags.csv'
        if tags_path.exists():
            tags_df = pd.read_csv(tags_path)
        else:
            tags_df = pd.DataFrame(columns=['userId', 'movieId', 'tag', 'timestamp'])
        
        logger.info(f"Extracted {len(ratings_df):,} ratings, {len(movies_df):,} movies, {len(tags_df):,} tags")
        return ratings_df, movies_df, tags_df
    
    def validate_data_quality(self, ratings_df: pd.DataFrame, 
                            movies_df: pd.DataFrame) -> DataQualityMetrics:
        """
        Comprehensive data quality validation
        
        Args:
            ratings_df: Ratings dataframe
            movies_df: Movies dataframe
            
        Returns:
            DataQualityMetrics object
        """
        logger.info("Validating data quality...")
        
        # Basic statistics
        total_ratings = len(ratings_df)
        unique_users = ratings_df['userId'].nunique()
        unique_movies = ratings_df['movieId'].nunique()
        rating_range = (ratings_df['rating'].min(), ratings_df['rating'].max())
        
        # Check for duplicates
        duplicate_count = ratings_df.duplicated(['userId', 'movieId']).sum()
        duplicate_rate = duplicate_count / total_ratings if total_ratings > 0 else 0
        
        # Missing values
        missing_values = {
            'ratings': ratings_df.isnull().sum().to_dict(),
            'movies': movies_df.isnull().sum().to_dict()
        }
        
        # Data freshness (assuming timestamp is in seconds)
        if 'timestamp' in ratings_df.columns:
            latest_timestamp = ratings_df['timestamp'].max()
            latest_date = datetime.fromtimestamp(latest_timestamp)
            data_freshness_days = (datetime.now() - latest_date).days
        else:
            data_freshness_days = float('inf')
        
        # Validation checks
        validation_passed = all([
            rating_range[0] >= 0.5 and rating_range[1] <= 5.0,  # Valid rating range
            duplicate_rate <= self.max_duplicate_rate,  # Acceptable duplicate rate
            unique_users >= 1000,  # Minimum user base
            unique_movies >= 100,  # Minimum movie catalog
        ])
        
        metrics = DataQualityMetrics(
            total_ratings=total_ratings,
            unique_users=unique_users,
            unique_movies=unique_movies,
            rating_range=rating_range,
            duplicate_count=duplicate_count,
            missing_values=missing_values,
            data_freshness_days=data_freshness_days,
            validation_passed=validation_passed,
            timestamp=datetime.now()
        )
        
        # Log validation results
        if validation_passed:
            logger.info("✅ Data quality validation PASSED")
        else:
            logger.warning("❌ Data quality validation FAILED")
            
        logger.info(f"Data metrics: {unique_users:,} users, {unique_movies:,} movies, {total_ratings:,} ratings")
        
        return metrics
    
    def transform_data(self, ratings_df: pd.DataFrame, 
                      movies_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Transform and clean data for model training
        
        Args:
            ratings_df: Raw ratings data
            movies_df: Movies metadata
            
        Returns:
            Cleaned ratings_df and metadata dict
        """
        logger.info("Starting data transformation...")
        
        # Remove duplicates
        original_count = len(ratings_df)
        ratings_df = ratings_df.drop_duplicates(['userId', 'movieId'])
        logger.info(f"Removed {original_count - len(ratings_df):,} duplicate ratings")
        
        # Filter users and movies with minimum interactions
        user_counts = ratings_df['userId'].value_counts()
        movie_counts = ratings_df['movieId'].value_counts()
        
        valid_users = user_counts[user_counts >= self.min_ratings_per_user].index
        valid_movies = movie_counts[movie_counts >= self.min_ratings_per_movie].index
        
        ratings_df = ratings_df[
            ratings_df['userId'].isin(valid_users) & 
            ratings_df['movieId'].isin(valid_movies)
        ]
        
        logger.info(f"After filtering: {len(valid_users):,} users, {len(valid_movies):,} movies")
        
        # Create ID mappings for model training
        unique_users = sorted(ratings_df['userId'].unique())
        unique_movies = sorted(ratings_df['movieId'].unique())
        
        self.user_mappings = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.movie_mappings = {movie_id: idx for idx, movie_id in enumerate(unique_movies)}
        self.reverse_user_mappings = {idx: user_id for user_id, idx in self.user_mappings.items()}
        self.reverse_movie_mappings = {idx: movie_id for movie_id, idx in self.movie_mappings.items()}
        
        # Apply mappings
        ratings_df['user_idx'] = ratings_df['userId'].map(self.user_mappings)
        ratings_df['movie_idx'] = ratings_df['movieId'].map(self.movie_mappings)
        
        # Feature engineering
        if 'timestamp' in ratings_df.columns:
            ratings_df['rating_date'] = pd.to_datetime(ratings_df['timestamp'], unit='s')
            ratings_df['rating_year'] = ratings_df['rating_date'].dt.year
            ratings_df['rating_month'] = ratings_df['rating_date'].dt.month
            ratings_df['rating_weekday'] = ratings_df['rating_date'].dt.weekday
        
        # Movie features
        movies_df['release_year'] = movies_df['title'].str.extract(r'\((\d{4})\)').astype(float)
        movies_df['genres_list'] = movies_df['genres'].str.split('|')
        
        # Create metadata
        metadata = {
            'n_users': len(self.user_mappings),
            'n_movies': len(self.movie_mappings),
            'n_ratings': len(ratings_df),
            'rating_stats': {
                'mean': ratings_df['rating'].mean(),
                'std': ratings_df['rating'].std(),
                'min': ratings_df['rating'].min(),
                'max': ratings_df['rating'].max()
            },
            'data_processed_at': datetime.now().isoformat()
        }
        
        logger.info("Data transformation completed")
        return ratings_df, metadata
    
    def create_temporal_splits(self, ratings_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create temporal train/validation/test splits
        Simulates real-world ETL where test data is "future" data
        """
        logger.info("Creating temporal data splits...")
        
        if 'timestamp' not in ratings_df.columns:
            logger.warning("No timestamp column found. Using random splits instead.")
            return self._create_random_splits(ratings_df)
        
        # Sort by timestamp
        ratings_df = ratings_df.sort_values('timestamp')
        
        # Calculate split indices
        n_ratings = len(ratings_df)
        train_idx = int(n_ratings * self.config['train_ratio'])
        val_idx = int(n_ratings * (self.config['train_ratio'] + self.config['val_ratio']))
        
        # Create splits
        train_df = ratings_df.iloc[:train_idx].copy()
        val_df = ratings_df.iloc[train_idx:val_idx].copy()
        test_df = ratings_df.iloc[val_idx:].copy()
        
        logger.info(f"Temporal splits: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")
        
        return train_df, val_df, test_df
    
    def _create_random_splits(self, ratings_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Fallback random splits when no timestamp available"""
        ratings_df = ratings_df.sample(frac=1, random_state=42)  # Shuffle
        
        n_ratings = len(ratings_df)
        train_idx = int(n_ratings * self.config['train_ratio'])
        val_idx = int(n_ratings * (self.config['train_ratio'] + self.config['val_ratio']))
        
        train_df = ratings_df.iloc[:train_idx].copy()
        val_df = ratings_df.iloc[train_idx:val_idx].copy()
        test_df = ratings_df.iloc[val_idx:].copy()
        
        return train_df, val_df, test_df
    
    def save_processed_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                          test_df: pd.DataFrame, movies_df: pd.DataFrame, 
                          metadata: Dict, quality_metrics: DataQualityMetrics):
        """Save all processed data and metadata"""
        logger.info("Saving processed data...")
        
        # Save data splits
        train_df.to_csv(self.output_dir / 'train_data.csv', index=False)
        val_df.to_csv(self.output_dir / 'val_data.csv', index=False)
        test_df.to_csv(self.output_dir / 'test_data.csv', index=False)
        
        # Save movies with features
        movies_df.to_csv(self.output_dir / 'movies_processed.csv', index=False)
        
        # Save mappings
        mappings = {
            'user_mappings': self.user_mappings,
            'movie_mappings': self.movie_mappings,
            'reverse_user_mappings': self.reverse_user_mappings,
            'reverse_movie_mappings': self.reverse_movie_mappings
        }
        
        with open(self.output_dir / 'data_mappings.pkl', 'wb') as f:
            pickle.dump(mappings, f)
        
        # Save metadata
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save quality metrics
        with open(self.output_dir / 'quality_metrics.json', 'w') as f:
            json.dump(asdict(quality_metrics), f, indent=2, default=str)
        
        logger.info(f"✅ All data saved to {self.output_dir}")
    
    def run_full_pipeline(self, data_source: str = None) -> Dict:
        """
        Run the complete ETL pipeline
        
        Args:
            data_source: Path to source data
            
        Returns:
            Pipeline execution summary
        """
        start_time = datetime.now()
        logger.info("🚀 Starting ETL pipeline...")
        
        try:
            # Extract
            ratings_df, movies_df, tags_df = self.extract_data(data_source)
            
            # Validate
            quality_metrics = self.validate_data_quality(ratings_df, movies_df)
            
            if not quality_metrics.validation_passed:
                raise ValueError("Data quality validation failed")
            
            # Transform
            ratings_df, metadata = self.transform_data(ratings_df, movies_df)
            
            # Create splits
            train_df, val_df, test_df = self.create_temporal_splits(ratings_df)
            
            # Load (save)
            self.save_processed_data(train_df, val_df, test_df, movies_df, metadata, quality_metrics)
            
            # Summary
            execution_time = (datetime.now() - start_time).total_seconds()
            summary = {
                'status': 'SUCCESS',
                'execution_time_seconds': execution_time,
                'data_processed': {
                    'train_samples': len(train_df),
                    'val_samples': len(val_df),
                    'test_samples': len(test_df),
                    'total_users': metadata['n_users'],
                    'total_movies': metadata['n_movies']
                },
                'quality_metrics': asdict(quality_metrics),
                'output_directory': str(self.output_dir),
                'completed_at': datetime.now().isoformat()
            }
            
            logger.info(f"✅ ETL pipeline completed successfully in {execution_time:.1f}s")
            return summary
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_summary = {
                'status': 'FAILED',
                'execution_time_seconds': execution_time,
                'error': str(e),
                'failed_at': datetime.now().isoformat()
            }
            logger.error(f"❌ ETL pipeline failed: {e}")
            return error_summary

def main():
    """Run ETL pipeline from command line"""
    import argparse
    
    parser = argparse.ArgumentParser(description='MovieLens ETL Pipeline')
    parser.add_argument('--data_source', type=str, help='Path to source data directory')
    parser.add_argument('--config', type=str, default='etl_config.json', help='Config file path')
    parser.add_argument('--output', type=str, help='Output directory override')
    
    args = parser.parse_args()
    
    # Initialize ETL
    etl = MovieLensETL(config_path=args.config)
    
    if args.output:
        etl.output_dir = Path(args.output)
        etl.output_dir.mkdir(exist_ok=True)
    
    # Run pipeline
    summary = etl.run_full_pipeline(data_source=args.data_source)
    
    # Print summary
    print("\n" + "="*50)
    print("ETL PIPELINE SUMMARY")
    print("="*50)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()