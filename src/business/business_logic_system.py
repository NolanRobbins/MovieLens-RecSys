# 🏢 Business Logic Layer for Production Recommendations
# Based on Chapter 14: Business Logic

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

@dataclass
class UserProfile:
    """User profile with preferences and constraints"""
    user_id: int
    hard_avoids: Set[int] = None  # Movie IDs to never recommend
    genre_preferences: Dict[str, float] = None  # Genre weights
    age_rating_limit: str = 'R'  # Max content rating
    language_preferences: Set[str] = None  # Preferred languages
    recency_bias: float = 0.5  # Preference for newer content (0-1)
    diversity_preference: float = 0.3  # How much diversity vs accuracy (0-1)
    
    def __post_init__(self):
        if self.hard_avoids is None:
            self.hard_avoids = set()
        if self.genre_preferences is None:
            self.genre_preferences = {}
        if self.language_preferences is None:
            self.language_preferences = {'English'}

class BusinessRulesEngine:
    """
    Production business rules for recommendation filtering and ranking
    Implements hard avoids, soft preferences, and business constraints
    """
    
    def __init__(self, movie_metadata: pd.DataFrame):
        """
        Args:
            movie_metadata: DataFrame with columns [movieId, title, genres, year, rating, language, popularity]
        """
        self.movie_metadata = movie_metadata
        self.popularity_scores = self._compute_popularity_scores()
        
    def _compute_popularity_scores(self) -> Dict[int, float]:
        """Compute popularity scores for inventory health"""
        if 'popularity' in self.movie_metadata.columns:
            return dict(zip(self.movie_metadata['movieId'], 
                          self.movie_metadata['popularity']))
        else:
            # Fallback: use inverse of movie ID as popularity proxy
            return dict(zip(self.movie_metadata['movieId'], 
                          1.0 / (self.movie_metadata['movieId'] + 1)))
    
    def apply_hard_avoids(self, recommendations: List[int], 
                         user_profile: UserProfile) -> List[int]:
        """
        Remove items from hard avoid list
        
        Examples:
        - Movies user explicitly disliked
        - Content above age rating
        - Wrong language
        """
        filtered = []
        
        for movie_id in recommendations:
            # Skip if in hard avoids
            if movie_id in user_profile.hard_avoids:
                continue
                
            movie_info = self.movie_metadata[
                self.movie_metadata['movieId'] == movie_id
            ]
            
            if movie_info.empty:
                continue
                
            movie_row = movie_info.iloc[0]
            
            # Age rating filter
            if 'rating' in movie_row and movie_row['rating']:
                if not self._check_age_rating(movie_row['rating'], 
                                            user_profile.age_rating_limit):
                    continue
            
            # Language filter  
            if 'language' in movie_row and movie_row['language']:
                if movie_row['language'] not in user_profile.language_preferences:
                    continue
            
            filtered.append(movie_id)
        
        return filtered
    
    def _check_age_rating(self, movie_rating: str, user_limit: str) -> bool:
        """Check if movie rating is within user's limit"""
        rating_hierarchy = ['G', 'PG', 'PG-13', 'R', 'NC-17']
        
        try:
            movie_level = rating_hierarchy.index(movie_rating)
            user_level = rating_hierarchy.index(user_limit)
            return movie_level <= user_level
        except ValueError:
            return True  # Default to allow if rating unknown
    
    def apply_soft_preferences(self, recommendations: List[int], 
                             scores: List[float], 
                             user_profile: UserProfile) -> Tuple[List[int], List[float]]:
        """
        Adjust scores based on user preferences
        
        Business rules:
        - Genre preferences
        - Recency bias
        - Diversity vs accuracy trade-off
        """
        adjusted_scores = []
        
        for movie_id, score in zip(recommendations, scores):
            movie_info = self.movie_metadata[
                self.movie_metadata['movieId'] == movie_id
            ]
            
            if movie_info.empty:
                adjusted_scores.append(score)
                continue
                
            movie_row = movie_info.iloc[0]
            adjusted_score = score
            
            # Genre preference adjustment
            if 'genres' in movie_row and movie_row['genres']:
                genre_boost = self._calculate_genre_boost(
                    movie_row['genres'], user_profile.genre_preferences
                )
                adjusted_score *= (1 + genre_boost)
            
            # Recency bias
            if 'year' in movie_row and movie_row['year']:
                recency_boost = self._calculate_recency_boost(
                    movie_row['year'], user_profile.recency_bias
                )
                adjusted_score *= (1 + recency_boost)
            
            adjusted_scores.append(adjusted_score)
        
        # Re-sort by adjusted scores
        sorted_pairs = sorted(zip(recommendations, adjusted_scores), 
                            key=lambda x: x[1], reverse=True)
        
        recommendations, adjusted_scores = zip(*sorted_pairs)
        return list(recommendations), list(adjusted_scores)
    
    def _calculate_genre_boost(self, movie_genres: str, 
                             genre_preferences: Dict[str, float]) -> float:
        """Calculate genre preference boost"""
        if not movie_genres or not genre_preferences:
            return 0.0
        
        genres = movie_genres.split('|')
        total_boost = 0.0
        
        for genre in genres:
            if genre in genre_preferences:
                total_boost += genre_preferences[genre]
        
        return total_boost / len(genres) if genres else 0.0
    
    def _calculate_recency_boost(self, movie_year: int, recency_bias: float) -> float:
        """Calculate recency bias boost"""
        if not movie_year or recency_bias == 0:
            return 0.0
        
        current_year = 2024
        years_old = current_year - movie_year
        
        # Newer movies get higher boost
        max_boost = 0.5 * recency_bias
        if years_old <= 0:
            return max_boost
        elif years_old >= 30:
            return -max_boost
        else:
            # Linear decay over 30 years
            return max_boost * (1 - years_old / 30)
    
    def apply_inventory_health(self, recommendations: List[int], 
                             scores: List[float]) -> Tuple[List[int], List[float]]:
        """
        Apply inventory health rules to promote certain content
        
        Business rules:
        - Boost less popular items for discovery
        - Promote seasonal/featured content
        - Balance catalog coverage
        """
        adjusted_scores = []
        
        for movie_id, score in zip(recommendations, scores):
            adjusted_score = score
            
            # Less popular items get small boost for diversity
            popularity = self.popularity_scores.get(movie_id, 0.5)
            if popularity < 0.3:  # Less popular items
                adjusted_score *= 1.1  # 10% boost
            
            adjusted_scores.append(adjusted_score)
        
        return recommendations, adjusted_scores