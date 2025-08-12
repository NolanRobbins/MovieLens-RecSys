# 🎯 Learning-to-Rank Loss Functions for Hybrid VAE
# Based on Chapter 12: Training for Ranking

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

class RankingLosses:
    """
    Implementation of ranking loss functions for recommendation systems
    Includes WARP, BPR, and ListNet losses for better ranking performance
    """
    
    @staticmethod
    def warp_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor, 
                  margin: float = 1.0, num_negatives: int = 10) -> torch.Tensor:
        """
        Weighted Approximate-Rank Pairwise (WARP) Loss
        Optimizes for precision@k by focusing on hard negatives
        
        Args:
            pos_scores: Scores for positive items [batch_size]
            neg_scores: Scores for negative items [batch_size, num_negatives]
            margin: Minimum score difference between pos and neg
            num_negatives: Number of negative samples per positive
        """
        batch_size = pos_scores.size(0)
        pos_scores_expanded = pos_scores.unsqueeze(1)  # [batch_size, 1]
        
        # Calculate violations (when negative score is too high)
        violations = torch.clamp(margin - pos_scores_expanded + neg_scores, min=0)
        
        # Count number of violating negatives for WARP weighting
        violating_mask = violations > 0
        num_violations = violating_mask.sum(dim=1).float()
        
        # WARP weights (higher weight for harder to separate examples)
        warp_weights = torch.zeros_like(num_violations)
        for i in range(batch_size):
            if num_violations[i] > 0:
                rank_approx = num_negatives / num_violations[i]
                warp_weights[i] = torch.log(rank_approx + 1)
        
        # Weighted loss
        loss_per_sample = (violations * violating_mask.float()).sum(dim=1)
        weighted_loss = loss_per_sample * warp_weights
        
        return weighted_loss.mean()
    
    @staticmethod
    def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        """
        Bayesian Personalized Ranking (BPR) Loss
        Optimizes pairwise preferences: P(user prefers item_i > item_j)
        
        Args:
            pos_scores: Scores for positive items [batch_size]  
            neg_scores: Scores for negative items [batch_size]
        """
        score_diff = pos_scores - neg_scores
        return -F.logsigmoid(score_diff).mean()
    
    @staticmethod
    def listnet_loss(y_true: torch.Tensor, y_pred: torch.Tensor, 
                     temperature: float = 1.0) -> torch.Tensor:
        """
        ListNet Loss - optimizes entire list ordering
        
        Args:
            y_true: True relevance scores [batch_size, num_items]
            y_pred: Predicted scores [batch_size, num_items] 
            temperature: Temperature for softmax (lower = more focused)
        """
        # Convert to probability distributions
        y_true_probs = F.softmax(y_true / temperature, dim=-1)
        y_pred_log_probs = F.log_softmax(y_pred / temperature, dim=-1)
        
        # KL divergence between distributions
        return F.kl_div(y_pred_log_probs, y_true_probs, reduction='batchmean')
    
    @staticmethod
    def lambda_rank_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Simplified LambdaRank-style loss
        Focuses on swapping incorrectly ordered pairs
        """
        batch_size, num_items = y_true.shape
        
        # Create all pairs
        y_true_expanded = y_true.unsqueeze(2)  # [batch, items, 1]
        y_pred_expanded = y_pred.unsqueeze(2)
        
        y_true_pairs = y_true_expanded - y_true.unsqueeze(1)  # [batch, items, items]
        y_pred_pairs = y_pred_expanded - y_pred.unsqueeze(1)
        
        # Only consider pairs where true preference exists
        preference_mask = (y_true_pairs > 0).float()
        
        # Loss when predicted order disagrees with true order
        ranking_loss = torch.clamp(1 - y_pred_pairs, min=0) * preference_mask
        
        return ranking_loss.mean()

class HybridVAERanking(torch.nn.Module):
    """
    Enhanced Hybrid VAE with ranking-optimized training
    Combines reconstruction loss with ranking losses for better recommendations
    """
    
    def __init__(self, base_vae_model, ranking_weight: float = 0.1, 
                 ranking_loss_type: str = 'bpr'):
        super().__init__()
        self.base_model = base_vae_model
        self.ranking_weight = ranking_weight
        self.ranking_loss_type = ranking_loss_type
        self.ranking_losses = RankingLosses()
        
    def forward(self, users: torch.Tensor, movies: torch.Tensor, 
                minmax: Optional[Tuple[float, float]] = None):
        """Forward pass through base VAE model"""
        return self.base_model(users, movies, minmax)
    
    def compute_ranking_loss(self, user_batch: torch.Tensor, pos_items: torch.Tensor,
                           ratings: torch.Tensor, num_negatives: int = 5):
        """
        Compute ranking loss for a batch of users
        
        Args:
            user_batch: User IDs [batch_size]
            pos_items: Positive item IDs [batch_size]  
            ratings: True ratings [batch_size]
            num_negatives: Number of negative samples per positive
        """
        batch_size = user_batch.size(0)
        device = user_batch.device
        
        # Get positive scores
        pos_predictions, pos_mu, pos_logvar = self.forward(user_batch, pos_items)
        pos_scores = pos_predictions.squeeze()
        
        # Sample negative items
        neg_items = torch.randint(0, self.base_model.movie_embedding.num_embeddings,
                                (batch_size, num_negatives), device=device)
        
        # Expand users for negative sampling
        users_expanded = user_batch.unsqueeze(1).expand(-1, num_negatives).reshape(-1)
        neg_items_flat = neg_items.reshape(-1)
        
        # Get negative scores
        neg_predictions, _, _ = self.forward(users_expanded, neg_items_flat)
        neg_scores = neg_predictions.squeeze().reshape(batch_size, num_negatives)
        
        # Compute ranking loss based on type
        if self.ranking_loss_type == 'bpr':
            # Use mean of negative scores for BPR
            neg_scores_mean = neg_scores.mean(dim=1)
            ranking_loss = self.ranking_losses.bpr_loss(pos_scores, neg_scores_mean)
        elif self.ranking_loss_type == 'warp':
            ranking_loss = self.ranking_losses.warp_loss(pos_scores, neg_scores)
        else:
            raise ValueError(f"Unknown ranking loss type: {self.ranking_loss_type}")
        
        return ranking_loss, pos_mu, pos_logvar
    
    def hybrid_loss(self, user_batch: torch.Tensor, pos_items: torch.Tensor, 
                   ratings: torch.Tensor, kl_weight: float = 1.0):
        """
        Combined VAE reconstruction + ranking loss
        
        Returns:
            total_loss: Combined loss for backpropagation
            loss_components: Dictionary of individual loss components
        """
        # Compute ranking loss and VAE components
        ranking_loss, mu, logvar = self.compute_ranking_loss(user_batch, pos_items, ratings)
        
        # Standard VAE reconstruction loss
        pos_predictions, _, _ = self.forward(user_batch, pos_items)
        recon_loss = F.mse_loss(pos_predictions.squeeze(), ratings, reduction='mean')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Combined loss
        total_loss = recon_loss + kl_weight * kl_loss + self.ranking_weight * ranking_loss
        
        loss_components = {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'ranking_loss': ranking_loss
        }
        
        return total_loss, loss_components

def train_with_ranking_loss(model, train_loader, optimizer, device, 
                          ranking_weight=0.1, kl_weight=1.0):
    """
    Training loop with ranking loss integration
    """
    model.train()
    total_losses = []
    
    # Wrap base model for ranking
    ranking_model = HybridVAERanking(model, ranking_weight=ranking_weight)
    ranking_model.to(device)
    
    for batch_idx, (batch_data, batch_ratings) in enumerate(train_loader):
        batch_data = batch_data.to(device)
        batch_ratings = batch_ratings.to(device)
        
        users = batch_data[:, 0]
        items = batch_data[:, 1]
        
        optimizer.zero_grad()
        
        # Compute hybrid loss
        total_loss, loss_components = ranking_model.hybrid_loss(
            users, items, batch_ratings, kl_weight=kl_weight
        )
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_losses.append(total_loss.item())
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}: '
                  f'Total: {loss_components["total_loss"]:.4f}, '
                  f'Recon: {loss_components["recon_loss"]:.4f}, '
                  f'KL: {loss_components["kl_loss"]:.4f}, '
                  f'Ranking: {loss_components["ranking_loss"]:.4f}')
    
    return np.mean(total_losses)

# Business-Optimized Retrieval Pipeline
class BusinessOptimizedRetrieval:
    """
    Two-stage retrieval + ranking system optimized for business metrics
    """
    
    def __init__(self, vae_model, candidate_size: int = 100):
        self.vae_model = vae_model
        self.candidate_size = candidate_size
        
    def retrieve_candidates(self, user_id: int, n_movies: int, 
                          exclude_seen: bool = True) -> torch.Tensor:
        """
        Fast candidate retrieval using VAE embeddings
        """
        # Use user embedding similarity for fast retrieval
        user_emb = self.vae_model.user_embedding(torch.tensor([user_id]))
        movie_embs = self.vae_model.movie_embedding.weight
        
        # Cosine similarity for candidate selection
        similarities = F.cosine_similarity(user_emb, movie_embs, dim=1)
        
        # Get top candidates
        _, candidate_indices = torch.topk(similarities, self.candidate_size)
        
        return candidate_indices
    
    def rank_candidates(self, user_id: int, candidate_items: torch.Tensor) -> torch.Tensor:
        """
        Detailed ranking of candidates using full VAE model
        """
        device = candidate_items.device
        num_candidates = candidate_items.size(0)
        
        user_tensor = torch.full((num_candidates,), user_id, device=device)
        
        with torch.no_grad():
            predictions, _, _ = self.vae_model(user_tensor, candidate_items, (0.5, 5.0))
            scores = predictions.squeeze()
        
        # Sort by predicted scores
        sorted_indices = torch.argsort(scores, descending=True)
        
        return candidate_items[sorted_indices], scores[sorted_indices]

print("🎯 Ranking-optimized VAE system ready!")
print("📈 Use HybridVAERanking for better recommendation ranking")
print("🚀 Use BusinessOptimizedRetrieval for scalable candidate selection")