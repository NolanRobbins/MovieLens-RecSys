"""
Evaluation metrics for recommender systems
Supports both rating prediction and ranking metrics
"""

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import List, Dict, Tuple
import math


def compute_rmse(predictions: List[float], targets: List[float]) -> float:
    """Compute Root Mean Squared Error"""
    return math.sqrt(mean_squared_error(targets, predictions))


def compute_mae(predictions: List[float], targets: List[float]) -> float:
    """Compute Mean Absolute Error"""
    return mean_absolute_error(targets, predictions)


def compute_hit_ratio(predictions: np.ndarray, 
                     targets: np.ndarray, 
                     k: int = 10) -> float:
    """
    Compute Hit Ratio @ K
    
    Args:
        predictions: Predicted scores [batch_size, n_items]
        targets: Ground truth items [batch_size]
        k: Top-k for evaluation
        
    Returns:
        Hit ratio @ k
    """
    hits = 0
    total = 0
    
    for i in range(len(predictions)):
        # Get top-k predictions
        top_k_items = np.argsort(predictions[i])[-k:]
        
        # Check if target item is in top-k
        if targets[i] in top_k_items:
            hits += 1
        total += 1
    
    return hits / total if total > 0 else 0.0


def compute_ndcg(predictions: np.ndarray,
                targets: np.ndarray,
                k: int = 10) -> float:
    """
    Compute Normalized Discounted Cumulative Gain @ K
    
    Args:
        predictions: Predicted scores [batch_size, n_items]
        targets: Ground truth items [batch_size] 
        k: Top-k for evaluation
        
    Returns:
        NDCG @ k
    """
    ndcg_scores = []
    
    for i in range(len(predictions)):
        # Get top-k predictions with scores
        top_k_indices = np.argsort(predictions[i])[-k:][::-1]
        
        # Calculate DCG
        dcg = 0.0
        for j, item_idx in enumerate(top_k_indices):
            if item_idx == targets[i]:
                dcg = 1.0 / math.log2(j + 2)  # j+2 because log2(1) = 0
                break
        
        # IDCG (Ideal DCG) for single relevant item
        idcg = 1.0 / math.log2(2)  # Best possible position is rank 1
        
        # NDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)
    
    return np.mean(ndcg_scores)


def compute_mrr(predictions: np.ndarray,
               targets: np.ndarray,
               k: int = 10) -> float:
    """
    Compute Mean Reciprocal Rank @ K
    
    Args:
        predictions: Predicted scores [batch_size, n_items]
        targets: Ground truth items [batch_size]
        k: Top-k for evaluation
        
    Returns:
        MRR @ k
    """
    reciprocal_ranks = []
    
    for i in range(len(predictions)):
        # Get top-k predictions
        top_k_indices = np.argsort(predictions[i])[-k:][::-1]
        
        # Find rank of target item
        rank = None
        for j, item_idx in enumerate(top_k_indices):
            if item_idx == targets[i]:
                rank = j + 1  # 1-indexed rank
                break
        
        # Reciprocal rank
        rr = 1.0 / rank if rank is not None else 0.0
        reciprocal_ranks.append(rr)
    
    return np.mean(reciprocal_ranks)


def compute_metrics(predictions: List[float], 
                   targets: List[float],
                   ranking_preds: np.ndarray = None,
                   ranking_targets: np.ndarray = None) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics
    
    Args:
        predictions: Rating predictions for RMSE/MAE
        targets: Rating targets for RMSE/MAE
        ranking_preds: Score matrix for ranking metrics [batch_size, n_items]
        ranking_targets: Target items for ranking metrics [batch_size]
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Rating prediction metrics
    if predictions and targets:
        metrics['rmse'] = compute_rmse(predictions, targets)
        metrics['mae'] = compute_mae(predictions, targets)
    
    # Ranking metrics
    if ranking_preds is not None and ranking_targets is not None:
        metrics['hr@10'] = compute_hit_ratio(ranking_preds, ranking_targets, k=10)
        metrics['ndcg@10'] = compute_ndcg(ranking_preds, ranking_targets, k=10)
        metrics['mrr@10'] = compute_mrr(ranking_preds, ranking_targets, k=10)
    
    return metrics


def log_metrics(metrics: Dict[str, float], 
               epoch: int = None,
               phase: str = "val") -> None:
    """
    Log metrics in a formatted way
    
    Args:
        metrics: Dictionary of metrics
        epoch: Current epoch number
        phase: Training phase (train/val/test)
    """
    if epoch is not None:
        print(f"\nEpoch {epoch} - {phase.upper()} Metrics:")
    else:
        print(f"\n{phase.upper()} Metrics:")
    
    print("-" * 40)
    
    # Rating metrics
    if 'rmse' in metrics:
        print(f"RMSE: {metrics['rmse']:.4f}")
    if 'mae' in metrics:
        print(f"MAE:  {metrics['mae']:.4f}")
    
    # Ranking metrics
    if 'hr@10' in metrics:
        print(f"HR@10:   {metrics['hr@10']:.4f}")
    if 'ndcg@10' in metrics:
        print(f"NDCG@10: {metrics['ndcg@10']:.4f}")
    if 'mrr@10' in metrics:
        print(f"MRR@10:  {metrics['mrr@10']:.4f}")
    
    print("-" * 40)


class EarlyStopping:
    """Early stopping utility for training"""
    
    def __init__(self, patience: int = 15, min_delta: float = 0.001):
        """
        Initialize early stopping
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum improvement to reset patience
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = float('inf')
        self.counter = 0
        
    def __call__(self, val_score: float) -> bool:
        """
        Check if training should stop
        
        Args:
            val_score: Current validation score (lower is better)
            
        Returns:
            True if training should stop
        """
        if val_score < self.best_score - self.min_delta:
            self.best_score = val_score
            self.counter = 0
        else:
            self.counter += 1
        
        return self.counter >= self.patience


def evaluate_leave_one_out(model: torch.nn.Module,
                          data_loader: torch.utils.data.DataLoader,
                          device: torch.device,
                          n_items: int) -> Dict[str, float]:
    """
    Evaluate model using leave-one-out protocol
    Standard evaluation for recommender systems
    
    Args:
        model: Trained model
        data_loader: Test data loader
        device: Device for computation
        n_items: Total number of items
        
    Returns:
        Dictionary of ranking metrics
    """
    model.eval()
    
    all_hr = []
    all_ndcg = []
    all_mrr = []
    
    with torch.no_grad():
        for batch in data_loader:
            # For Neural CF
            if isinstance(batch, tuple):
                users, items, ratings = batch
                users, items = users.to(device), items.to(device)
                
                # Get predictions for all items for each user
                batch_size = users.size(0)
                all_predictions = []
                
                for i in range(batch_size):
                    user_id = users[i:i+1]
                    all_item_ids = torch.arange(n_items, device=device)
                    user_ids_expanded = user_id.expand(n_items)
                    
                    predictions = model(user_ids_expanded, all_item_ids)
                    all_predictions.append(predictions.cpu().numpy())
                
                predictions_matrix = np.array(all_predictions)
                target_items = items.cpu().numpy()
            
            # For SS4Rec (sequential data)
            else:
                # Handle sequential batch format
                # Implementation depends on SS4Rec model interface
                pass
            
            # Compute ranking metrics for this batch
            hr = compute_hit_ratio(predictions_matrix, target_items, k=10)
            ndcg = compute_ndcg(predictions_matrix, target_items, k=10)
            mrr = compute_mrr(predictions_matrix, target_items, k=10)
            
            all_hr.append(hr)
            all_ndcg.append(ndcg)
            all_mrr.append(mrr)
    
    return {
        'hr@10': np.mean(all_hr),
        'ndcg@10': np.mean(all_ndcg),
        'mrr@10': np.mean(all_mrr)
    }