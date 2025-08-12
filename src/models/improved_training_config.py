"""
Fixed Training Configuration for MovieLens Hybrid VAE
Addresses critical issues causing RMSE 1.21 performance problem
"""

# CRITICAL FIX #1: Use MEAN reduction, not SUM
def fixed_vae_loss_function(predictions, targets, mu, logvar, kl_weight=0.1):
    """Fixed VAE loss function with proper scaling"""
    # FIXED: Use 'mean' instead of 'sum' - this was the major issue!
    recon_loss = F.mse_loss(predictions.squeeze(), targets, reduction='mean')
    
    # FIXED: Proper KL divergence scaling
    batch_size = predictions.size(0)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    total_loss = recon_loss + kl_weight * kl_loss
    return total_loss, recon_loss, kl_loss

# FIXED TRAINING CONFIGURATION
improved_config = {
    'data_path': '/workspace/data',
    'save_path': '/workspace/hybrid_vae_improved.pt',
    'batch_size': 1024,          # Larger batches for stability
    'n_epochs': 150,             # More epochs since we fixed the issues
    'lr': 5e-4,                  # FIXED: Higher learning rate
    'weight_decay': 1e-5,
    'n_factors': 150,
    'hidden_dims': [512, 256, 128], # Added third layer
    'latent_dim': 64,
    'dropout_rate': 0.15,        # FIXED: Reduced dropout
    'kl_weight': 0.1,           # FIXED: Much lower KL weight  
    'patience': 20,
    'seed': 42,
    'use_wandb': True
}

# Expected Results with Fixed Configuration:
# - RMSE should drop to 0.85-0.95 (significant improvement)
# - R² should become positive (>0.1)
# - Model should actually learn rating patterns
# - Prediction std should match target std better