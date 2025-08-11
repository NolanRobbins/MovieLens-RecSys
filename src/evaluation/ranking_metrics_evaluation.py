# 📊 Advanced Ranking Metrics for Business Impact
# Based on Chapter 11: Personalized Recommendation Metrics

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ndcg_score
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

class RecommenderMetrics:
    """
    Comprehensive ranking metrics for recommendation systems
    Implements mAP, MRR, NDCG, and business-specific metrics
    """
    
    def __init__(self, k_values: List[int] = [5, 10, 20, 50]):
        self.k_values = k_values
        
    def mean_average_precision_at_k(self, y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
        """
        Calculate Mean Average Precision @ k
        Higher values = better ranking of relevant items
        """
        # Get top-k predictions
        top_k_indices = np.argsort(y_scores)[::-1][:k]
        
        # Check which are relevant (rating >= 4.0 for MovieLens)
        relevant_mask = y_true >= 4.0
        
        precision_scores = []
        num_relevant = 0
        
        for i, idx in enumerate(top_k_indices):
            if relevant_mask[idx]:
                num_relevant += 1
                precision_at_i = num_relevant / (i + 1)
                precision_scores.append(precision_at_i)
        
        return np.mean(precision_scores) if precision_scores else 0.0
    
    def mean_reciprocal_rank_at_k(self, y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
        """
        Calculate Mean Reciprocal Rank @ k
        Focuses on position of first relevant item
        """
        top_k_indices = np.argsort(y_scores)[::-1][:k]
        relevant_mask = y_true >= 4.0
        
        for i, idx in enumerate(top_k_indices):
            if relevant_mask[idx]:
                return 1.0 / (i + 1)
        return 0.0
    
    def ndcg_at_k(self, y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain @ k
        Accounts for graded relevance and position bias
        """
        # Convert to relevance scores (0-4 scale)
        relevance_scores = np.maximum(0, y_true - 1)  # 0.5->0, 5.0->4
        
        return ndcg_score(
            relevance_scores.reshape(1, -1), 
            y_scores.reshape(1, -1), 
            k=k
        )
    
    def precision_at_k(self, y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
        """Calculate Precision @ k"""
        top_k_indices = np.argsort(y_scores)[::-1][:k]
        relevant_mask = y_true >= 4.0
        return np.sum(relevant_mask[top_k_indices]) / k
    
    def recall_at_k(self, y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
        """Calculate Recall @ k"""
        top_k_indices = np.argsort(y_scores)[::-1][:k]
        relevant_mask = y_true >= 4.0
        total_relevant = np.sum(relevant_mask)
        
        if total_relevant == 0:
            return 0.0
        
        return np.sum(relevant_mask[top_k_indices]) / total_relevant
    
    def coverage_at_k(self, recommendations: List[List[int]], total_items: int, k: int) -> float:
        """
        Calculate catalog coverage - what % of items get recommended
        Important for business health and long-tail discovery
        """
        recommended_items = set()
        for user_recs in recommendations:
            recommended_items.update(user_recs[:k])
        
        return len(recommended_items) / total_items
    
    def business_metrics_at_k(self, y_true: np.ndarray, y_scores: np.ndarray, 
                            movie_popularity: np.ndarray, k: int) -> Dict[str, float]:
        """
        Calculate business-specific metrics
        """
        top_k_indices = np.argsort(y_scores)[::-1][:k]
        relevant_mask = y_true >= 4.0
        
        # Hit Rate (did we recommend anything good?)
        hit_rate = 1.0 if np.any(relevant_mask[top_k_indices]) else 0.0
        
        # Average popularity bias (lower = more diverse/long-tail)
        avg_popularity = np.mean(movie_popularity[top_k_indices])
        
        # Novelty (lower popularity = higher novelty)
        novelty = -np.log2(avg_popularity + 1e-8)
        
        return {
            'hit_rate': hit_rate,
            'avg_popularity': avg_popularity,
            'novelty': novelty
        }
    
    def evaluate_user_recommendations(self, user_id: int, y_true: np.ndarray, 
                                    y_scores: np.ndarray, movie_popularity: np.ndarray) -> Dict:
        """Comprehensive evaluation for a single user"""
        results = {'user_id': user_id}
        
        for k in self.k_values:
            results.update({
                f'map@{k}': self.mean_average_precision_at_k(y_true, y_scores, k),
                f'mrr@{k}': self.mean_reciprocal_rank_at_k(y_true, y_scores, k),
                f'ndcg@{k}': self.ndcg_at_k(y_true, y_scores, k),
                f'precision@{k}': self.precision_at_k(y_true, y_scores, k),
                f'recall@{k}': self.recall_at_k(y_true, y_scores, k),
            })
            
            # Business metrics
            biz_metrics = self.business_metrics_at_k(y_true, y_scores, movie_popularity, k)
            for metric, value in biz_metrics.items():
                results[f'{metric}@{k}'] = value
        
        return results

def evaluate_recommendation_system(model, test_loader, n_users, n_movies, 
                                 movie_popularity, device='cpu'):
    """
    Comprehensive evaluation of recommendation system
    Returns business-relevant metrics beyond RMSE
    """
    model.eval()
    metrics_calculator = RecommenderMetrics()
    all_results = []
    
    print("🔍 Evaluating recommendation system with ranking metrics...")
    
    # Movie popularity for diversity metrics
    if movie_popularity is None:
        movie_popularity = np.random.exponential(0.1, n_movies)  # Placeholder
    
    minmax = (0.5, 5.0)
    
    # Evaluate per user (sample to avoid memory issues)
    sample_users = np.random.choice(n_users, min(1000, n_users), replace=False)
    
    with torch.no_grad():
        for user_id in sample_users:
            # Get all predictions for this user
            user_tensor = torch.full((n_movies,), user_id, dtype=torch.long, device=device)
            movie_tensor = torch.arange(n_movies, dtype=torch.long, device=device)
            
            predictions, _, _ = model(user_tensor, movie_tensor, minmax)
            predictions = predictions.cpu().squeeze().numpy()
            
            # Create dummy ground truth (in practice, use actual test ratings)
            y_true = np.random.uniform(0.5, 5.0, n_movies)  # Replace with real data
            
            # Evaluate this user
            user_results = metrics_calculator.evaluate_user_recommendations(
                user_id, y_true, predictions, movie_popularity
            )
            all_results.append(user_results)
    
    # Aggregate results
    results_df = pd.DataFrame(all_results)
    aggregated = results_df.select_dtypes(include=[np.number]).mean()
    
    print("\n📊 RANKING METRICS RESULTS:")
    print("=" * 50)
    
    for k in [5, 10, 20]:
        print(f"\n📈 @ {k} recommendations:")
        print(f"  mAP:       {aggregated[f'map@{k}']:.4f}")
        print(f"  MRR:       {aggregated[f'mrr@{k}']:.4f}")
        print(f"  NDCG:      {aggregated[f'ndcg@{k}']:.4f}")
        print(f"  Precision: {aggregated[f'precision@{k}']:.4f}")
        print(f"  Recall:    {aggregated[f'recall@{k}']:.4f}")
        print(f"  Hit Rate:  {aggregated[f'hit_rate@{k}']:.4f}")
        print(f"  Novelty:   {aggregated[f'novelty@{k}']:.4f}")
    
    return aggregated, results_df

# Business Impact Calculator
class BusinessImpactCalculator:
    """Calculate business KPIs from recommendation metrics"""
    
    def __init__(self, baseline_ctr: float = 0.02, avg_order_value: float = 15.0):
        self.baseline_ctr = baseline_ctr
        self.avg_order_value = avg_order_value
    
    def calculate_revenue_impact(self, precision_at_10: float, users_per_day: int = 10000):
        """
        Estimate revenue impact from improved precision
        """
        # Improved CTR from better recommendations
        improved_ctr = self.baseline_ctr * (1 + precision_at_10)
        ctr_lift = improved_ctr - self.baseline_ctr
        
        # Daily revenue impact
        additional_conversions = users_per_day * ctr_lift
        daily_revenue_impact = additional_conversions * self.avg_order_value
        annual_revenue_impact = daily_revenue_impact * 365
        
        return {
            'ctr_lift': ctr_lift,
            'daily_revenue_impact': daily_revenue_impact,
            'annual_revenue_impact': annual_revenue_impact,
            'roi_percentage': (ctr_lift / self.baseline_ctr) * 100
        }
    
    def user_satisfaction_score(self, hit_rate: float, novelty: float, ndcg: float):
        """
        Composite satisfaction score for business dashboard
        """
        # Weighted combination of metrics
        satisfaction = (
            0.4 * hit_rate +      # Did they find something good?
            0.3 * ndcg +          # Was it well-ranked?
            0.3 * min(novelty/5, 1.0)  # Was it novel/diverse?
        )
        return min(satisfaction, 1.0)

print("📊 Advanced ranking metrics system ready!")
print("🎯 Use evaluate_recommendation_system() for comprehensive evaluation")
print("💰 Use BusinessImpactCalculator for revenue impact estimation")