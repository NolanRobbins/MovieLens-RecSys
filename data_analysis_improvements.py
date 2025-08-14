# Comprehensive Data Analysis and Improvement Strategies
import pandas as pd
import numpy as np
# Optional plotting imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import torch

class MovieLensDataAnalyzer:
    """
    Comprehensive data analysis for MovieLens dataset
    """
    def __init__(self, train_df, val_df, movies_df=None):
        self.train_df = train_df.copy()
        self.val_df = val_df.copy()
        self.movies_df = movies_df
        self.analysis_results = {}
        
    def comprehensive_analysis(self):
        """Run complete data analysis pipeline"""
        print("🔍 Starting Comprehensive Data Analysis")
        print("=" * 50)
        
        # Basic statistics
        self._basic_statistics()
        
        # Rating distribution analysis
        self._analyze_rating_distribution()
        
        # User behavior analysis
        self._analyze_user_behavior()
        
        # Movie popularity analysis
        self._analyze_movie_popularity()
        
        # Sparsity analysis
        self._analyze_sparsity()
        
        # Temporal patterns (if timestamp available)
        self._analyze_temporal_patterns()
        
        # Data quality issues
        self._detect_data_quality_issues()
        
        # Generate recommendations
        self._generate_improvement_recommendations()
        
        return self.analysis_results
    
    def _basic_statistics(self):
        """Basic dataset statistics"""
        print("\n📊 Basic Dataset Statistics")
        print("-" * 30)
        
        stats = {
            'train_samples': len(self.train_df),
            'val_samples': len(self.val_df),
            'total_users': len(set(self.train_df['user_id'].unique()) | set(self.val_df['user_id'].unique())),
            'total_movies': len(set(self.train_df['movie_id'].unique()) | set(self.val_df['movie_id'].unique())),
            'train_users': self.train_df['user_id'].nunique(),
            'train_movies': self.train_df['movie_id'].nunique(),
            'val_users': self.val_df['user_id'].nunique(),
            'val_movies': self.val_df['movie_id'].nunique(),
        }
        
        for key, value in stats.items():
            print(f"   {key.replace('_', ' ').title()}: {value:,}")
        
        # Data density
        total_possible = stats['total_users'] * stats['total_movies']
        density = (stats['train_samples'] + stats['val_samples']) / total_possible * 100
        print(f"   Data Density: {density:.6f}%")
        
        self.analysis_results['basic_stats'] = stats
        self.analysis_results['density'] = density
    
    def _analyze_rating_distribution(self):
        """Analyze rating distribution patterns"""
        print("\n📈 Rating Distribution Analysis")
        print("-" * 30)
        
        train_ratings = self.train_df['rating'].values
        val_ratings = self.val_df['rating'].values
        
        # Basic statistics
        train_stats = {
            'mean': np.mean(train_ratings),
            'std': np.std(train_ratings),
            'min': np.min(train_ratings),
            'max': np.max(train_ratings),
            'median': np.median(train_ratings)
        }
        
        val_stats = {
            'mean': np.mean(val_ratings),
            'std': np.std(val_ratings),
            'min': np.min(val_ratings),
            'max': np.max(val_ratings),
            'median': np.median(val_ratings)
        }
        
        print(f"   Train - Mean: {train_stats['mean']:.3f}, Std: {train_stats['std']:.3f}")
        print(f"   Val   - Mean: {val_stats['mean']:.3f}, Std: {val_stats['std']:.3f}")
        
        # Rating distribution
        train_dist = np.bincount(train_ratings.astype(int))
        val_dist = np.bincount(val_ratings.astype(int))
        
        print(f"\n   Rating Distribution (Train/Val):")
        for rating in range(1, 6):
            if rating < len(train_dist) and rating < len(val_dist):
                train_pct = train_dist[rating] / len(train_ratings) * 100
                val_pct = val_dist[rating] / len(val_ratings) * 100
                print(f"   {rating}⭐: {train_pct:.1f}% / {val_pct:.1f}%")
        
        # Check for rating bias
        most_common_rating = np.argmax(train_dist[1:]) + 1  # Skip 0-rating
        bias_percentage = train_dist[most_common_rating] / len(train_ratings) * 100
        
        if bias_percentage > 40:
            print(f"   ⚠️ Rating bias detected: {most_common_rating}⭐ = {bias_percentage:.1f}%")
        
        self.analysis_results['rating_stats'] = {
            'train': train_stats,
            'val': val_stats,
            'bias_rating': most_common_rating,
            'bias_percentage': bias_percentage
        }
    
    def _analyze_user_behavior(self):
        """Analyze user behavior patterns"""
        print("\n👥 User Behavior Analysis")
        print("-" * 30)
        
        # User activity distribution
        user_counts = self.train_df['user_id'].value_counts()
        
        behavior_stats = {
            'mean_ratings_per_user': user_counts.mean(),
            'median_ratings_per_user': user_counts.median(),
            'std_ratings_per_user': user_counts.std(),
            'min_ratings_per_user': user_counts.min(),
            'max_ratings_per_user': user_counts.max()
        }
        
        print(f"   Avg ratings per user: {behavior_stats['mean_ratings_per_user']:.1f}")
        print(f"   Median ratings per user: {behavior_stats['median_ratings_per_user']:.1f}")
        print(f"   Most active user: {behavior_stats['max_ratings_per_user']} ratings")
        
        # User activity categories
        very_active = (user_counts >= 100).sum()
        active = ((user_counts >= 20) & (user_counts < 100)).sum()
        casual = ((user_counts >= 5) & (user_counts < 20)).sum()
        sparse = (user_counts < 5).sum()
        
        total_users = len(user_counts)
        print(f"\n   User Activity Categories:")
        print(f"   Very Active (100+ ratings): {very_active} ({very_active/total_users*100:.1f}%)")
        print(f"   Active (20-99 ratings): {active} ({active/total_users*100:.1f}%)")
        print(f"   Casual (5-19 ratings): {casual} ({casual/total_users*100:.1f}%)")
        print(f"   Sparse (<5 ratings): {sparse} ({sparse/total_users*100:.1f}%)")
        
        # User rating patterns
        user_avg_ratings = self.train_df.groupby('user_id')['rating'].agg(['mean', 'std', 'count'])
        
        # Harsh vs generous raters
        harsh_raters = (user_avg_ratings['mean'] < 3.0).sum()
        generous_raters = (user_avg_ratings['mean'] > 4.0).sum()
        
        print(f"\n   Rating Tendencies:")
        print(f"   Harsh raters (avg < 3.0): {harsh_raters} ({harsh_raters/total_users*100:.1f}%)")
        print(f"   Generous raters (avg > 4.0): {generous_raters} ({generous_raters/total_users*100:.1f}%)")
        
        self.analysis_results['user_behavior'] = {
            **behavior_stats,
            'activity_distribution': {
                'very_active': very_active,
                'active': active,
                'casual': casual,
                'sparse': sparse
            },
            'rating_tendencies': {
                'harsh_raters': harsh_raters,
                'generous_raters': generous_raters
            }
        }
    
    def _analyze_movie_popularity(self):
        """Analyze movie popularity patterns"""
        print("\n🎬 Movie Popularity Analysis")
        print("-" * 30)
        
        # Movie rating counts
        movie_counts = self.train_df['movie_id'].value_counts()
        
        popularity_stats = {
            'mean_ratings_per_movie': movie_counts.mean(),
            'median_ratings_per_movie': movie_counts.median(),
            'std_ratings_per_movie': movie_counts.std(),
            'min_ratings_per_movie': movie_counts.min(),
            'max_ratings_per_movie': movie_counts.max()
        }
        
        print(f"   Avg ratings per movie: {popularity_stats['mean_ratings_per_movie']:.1f}")
        print(f"   Median ratings per movie: {popularity_stats['median_ratings_per_movie']:.1f}")
        print(f"   Most popular movie: {popularity_stats['max_ratings_per_movie']} ratings")
        
        # Movie popularity categories
        blockbuster = (movie_counts >= 1000).sum()
        popular = ((movie_counts >= 100) & (movie_counts < 1000)).sum()
        niche = ((movie_counts >= 10) & (movie_counts < 100)).sum()
        obscure = (movie_counts < 10).sum()
        
        total_movies = len(movie_counts)
        print(f"\n   Movie Popularity Categories:")
        print(f"   Blockbuster (1000+ ratings): {blockbuster} ({blockbuster/total_movies*100:.1f}%)")
        print(f"   Popular (100-999 ratings): {popular} ({popular/total_movies*100:.1f}%)")
        print(f"   Niche (10-99 ratings): {niche} ({niche/total_movies*100:.1f}%)")
        print(f"   Obscure (<10 ratings): {obscure} ({obscure/total_movies*100:.1f}%)")
        
        # Movie quality distribution
        movie_avg_ratings = self.train_df.groupby('movie_id')['rating'].agg(['mean', 'std', 'count'])
        
        # High vs low rated movies
        high_rated = (movie_avg_ratings['mean'] >= 4.0).sum()
        low_rated = (movie_avg_ratings['mean'] <= 2.5).sum()
        
        print(f"\n   Movie Quality:")
        print(f"   High rated (avg >= 4.0): {high_rated} ({high_rated/total_movies*100:.1f}%)")
        print(f"   Low rated (avg <= 2.5): {low_rated} ({low_rated/total_movies*100:.1f}%)")
        
        self.analysis_results['movie_popularity'] = {
            **popularity_stats,
            'popularity_distribution': {
                'blockbuster': blockbuster,
                'popular': popular,
                'niche': niche,
                'obscure': obscure
            },
            'quality_distribution': {
                'high_rated': high_rated,
                'low_rated': low_rated
            }
        }
    
    def _analyze_sparsity(self):
        """Analyze data sparsity issues"""
        print("\n🕳️ Sparsity Analysis")
        print("-" * 30)
        
        # Create user-movie matrix
        n_users = self.train_df['user_id'].nunique()
        n_movies = self.train_df['movie_id'].nunique()
        n_ratings = len(self.train_df)
        
        # Sparsity metrics
        total_possible = n_users * n_movies
        sparsity = (1 - n_ratings / total_possible) * 100
        
        print(f"   Matrix size: {n_users:,} users × {n_movies:,} movies")
        print(f"   Total possible ratings: {total_possible:,}")
        print(f"   Actual ratings: {n_ratings:,}")
        print(f"   Sparsity: {sparsity:.4f}%")
        
        # Cold start analysis
        user_counts = self.train_df['user_id'].value_counts()
        movie_counts = self.train_df['movie_id'].value_counts()
        
        cold_start_thresholds = [1, 5, 10, 20]
        for threshold in cold_start_thresholds:
            cold_users = (user_counts < threshold).sum()
            cold_movies = (movie_counts < threshold).sum()
            print(f"   Users with <{threshold} ratings: {cold_users} ({cold_users/n_users*100:.1f}%)")
            print(f"   Movies with <{threshold} ratings: {cold_movies} ({cold_movies/n_movies*100:.1f}%)")
        
        self.analysis_results['sparsity'] = {
            'sparsity_percentage': sparsity,
            'matrix_size': (n_users, n_movies),
            'cold_start_users': {f'<{t}': (user_counts < t).sum() for t in cold_start_thresholds},
            'cold_start_movies': {f'<{t}': (movie_counts < t).sum() for t in cold_start_thresholds}
        }
    
    def _analyze_temporal_patterns(self):
        """Analyze temporal patterns if timestamp data is available"""
        print("\n⏰ Temporal Analysis")
        print("-" * 30)
        
        if 'timestamp' in self.train_df.columns:
            # Convert timestamp to datetime
            self.train_df['datetime'] = pd.to_datetime(self.train_df['timestamp'], unit='s')
            
            # Time range
            start_date = self.train_df['datetime'].min()
            end_date = self.train_df['datetime'].max()
            duration = end_date - start_date
            
            print(f"   Time range: {start_date.date()} to {end_date.date()}")
            print(f"   Duration: {duration.days} days")
            
            # Rating patterns over time
            monthly_counts = self.train_df.groupby(self.train_df['datetime'].dt.to_period('M')).size()
            print(f"   Avg ratings per month: {monthly_counts.mean():.0f}")
            print(f"   Peak month: {monthly_counts.max():,} ratings")
            
            self.analysis_results['temporal'] = {
                'start_date': start_date,
                'end_date': end_date,
                'duration_days': duration.days,
                'monthly_avg': monthly_counts.mean()
            }
        else:
            print("   No timestamp data available")
            self.analysis_results['temporal'] = None
    
    def _detect_data_quality_issues(self):
        """Detect various data quality issues"""
        print("\n🔍 Data Quality Issues")
        print("-" * 30)
        
        issues = []
        
        # Check for duplicate ratings
        train_duplicates = self.train_df.duplicated(['user_id', 'movie_id']).sum()
        val_duplicates = self.val_df.duplicated(['user_id', 'movie_id']).sum()
        
        if train_duplicates > 0:
            issues.append(f"Duplicate ratings in training: {train_duplicates}")
            print(f"   ⚠️ Duplicate ratings in training: {train_duplicates}")
        
        if val_duplicates > 0:
            issues.append(f"Duplicate ratings in validation: {val_duplicates}")
            print(f"   ⚠️ Duplicate ratings in validation: {val_duplicates}")
        
        # Check for rating range issues
        min_rating = min(self.train_df['rating'].min(), self.val_df['rating'].min())
        max_rating = max(self.train_df['rating'].max(), self.val_df['rating'].max())
        
        if min_rating < 0.5 or max_rating > 5.0:
            issues.append(f"Rating range issue: {min_rating} to {max_rating}")
            print(f"   ⚠️ Unusual rating range: {min_rating} to {max_rating}")
        
        # Check for data leakage (same user-movie pairs in train and val)
        train_pairs = set(zip(self.train_df['user_id'], self.train_df['movie_id']))
        val_pairs = set(zip(self.val_df['user_id'], self.val_df['movie_id']))
        overlap = len(train_pairs & val_pairs)
        
        if overlap > 0:
            issues.append(f"Data leakage: {overlap} overlapping user-movie pairs")
            print(f"   ❌ Data leakage: {overlap} overlapping user-movie pairs")
        
        # Check for extreme users/movies
        user_counts = self.train_df['user_id'].value_counts()
        movie_counts = self.train_df['movie_id'].value_counts()
        
        extreme_users = (user_counts > 1000).sum()
        extreme_movies = (movie_counts > 10000).sum()
        
        if extreme_users > 0:
            print(f"   ⚠️ Extreme users (>1000 ratings): {extreme_users}")
        
        if extreme_movies > 0:
            print(f"   ⚠️ Extreme movies (>10000 ratings): {extreme_movies}")
        
        if not issues:
            print("   ✅ No major data quality issues detected")
        
        self.analysis_results['data_quality'] = {
            'issues': issues,
            'duplicates': {'train': train_duplicates, 'val': val_duplicates},
            'data_leakage': overlap,
            'extreme_users': extreme_users,
            'extreme_movies': extreme_movies
        }
    
    def _generate_improvement_recommendations(self):
        """Generate data improvement recommendations"""
        print("\n💡 Improvement Recommendations")
        print("-" * 30)
        
        recommendations = []
        
        # Based on sparsity
        sparsity = self.analysis_results['sparsity']['sparsity_percentage']
        if sparsity > 99.9:
            recommendations.append("Extremely sparse data - consider data augmentation or implicit feedback")
        
        # Based on user behavior
        sparse_users = self.analysis_results['user_behavior']['activity_distribution']['sparse']
        total_users = sum(self.analysis_results['user_behavior']['activity_distribution'].values())
        
        if sparse_users / total_users > 0.5:
            recommendations.append("High proportion of sparse users - implement cold-start handling")
        
        # Based on movie popularity
        obscure_movies = self.analysis_results['movie_popularity']['popularity_distribution']['obscure']
        total_movies = sum(self.analysis_results['movie_popularity']['popularity_distribution'].values())
        
        if obscure_movies / total_movies > 0.7:
            recommendations.append("Many obscure movies - consider popularity-based sampling")
        
        # Based on rating bias
        bias_pct = self.analysis_results['rating_stats']['bias_percentage']
        if bias_pct > 40:
            recommendations.append("Strong rating bias - consider rating normalization")
        
        # Based on data quality
        if self.analysis_results['data_quality']['data_leakage'] > 0:
            recommendations.append("CRITICAL: Data leakage detected - review train/val split")
        
        # Based on user sparsity
        sparse_users = self.analysis_results['user_behavior']['activity_distribution']['sparse']
        total_users = sum(self.analysis_results['user_behavior']['activity_distribution'].values())
        if sparse_users / total_users > 0.6:
            recommendations.append("High user sparsity - implement advanced cold-start handling")
        
        # Final model architecture recommendations
        recommendations.append("Use embedding dropout of 0.1-0.2 to prevent overfitting")
        recommendations.append("Implement gradient clipping (max_norm=1.0) for training stability")
        recommendations.append("Use learning rate scheduling with warmup")
        
        self.analysis_results['recommendations'] = recommendations
        
        print(f"\n💡  Data Improvement Recommendations ({len(recommendations)} items):")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        return self.analysis_results

def run_comprehensive_analysis(train_df, val_df, movies_df=None):
    """Run complete analysis and return results"""
    analyzer = MovieLensDataAnalyzer(train_df, val_df, movies_df)
    return analyzer.comprehensive_analysis()

if __name__ == "__main__":
    print("🔍 MovieLens Data Analysis Tool")
    print("This script provides comprehensive analysis of MovieLens recommendation data")
    print("Use: from data_analysis_improvements import run_comprehensive_analysis")