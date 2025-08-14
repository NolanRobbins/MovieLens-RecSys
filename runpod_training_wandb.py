"""
Runpod A100 Training Script with Full W&B Integration and Monitoring
Enhanced version with comprehensive analytics and A100 performance tracking
"""

import os
import sys
import time
import pickle
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
import logging
import wandb
import psutil
from datetime import datetime

# GPU monitoring imports
try:
    import GPUtil
    import nvidia_ml_py3 as nvml
    nvml.nvmlInit()
    HAS_GPU_MONITORING = True
except ImportError:
    print("⚠️ GPU monitoring disabled - install GPUtil and nvidia-ml-py for full metrics")
    HAS_GPU_MONITORING = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class A100OptimizedHybridVAE(nn.Module):
    """
    A100-Optimized Hybrid VAE with comprehensive W&B logging
    """
    
    def __init__(self, n_users, n_movies, n_factors=160, hidden_dims=[640, 384, 192], 
                 latent_dim=80, dropout_rate=0.38):
        super(A100OptimizedHybridVAE, self).__init__()
        
        self.n_users = n_users
        self.n_movies = n_movies
        self.n_factors = n_factors
        self.latent_dim = latent_dim
        
        # Embedding layers - optimized for A100
        self.user_embedding = nn.Embedding(n_users, n_factors)
        self.movie_embedding = nn.Embedding(n_movies, n_factors)
        self.embedding_dropout = nn.Dropout(dropout_rate * 0.5)
        
        # Encoder with batch normalization
        encoder_layers = []
        input_dim = n_factors * 2
        
        for i, hidden_dim in enumerate(hidden_dims):
            encoder_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),  # Better than BatchNorm for VAEs
                nn.GELU(),  # A100-optimized activation
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # VAE bottleneck with numerical stability
        self.mu_layer = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar_layer = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder with skip connections concept
        decoder_layers = []
        input_dim = latent_dim
        
        for i, hidden_dim in enumerate(reversed(hidden_dims)):
            decoder_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate * 0.7)  # Lower dropout in decoder
            ])
            input_dim = hidden_dim
        
        # Final prediction layer
        decoder_layers.extend([
            nn.Linear(input_dim, 64),  # Additional layer for capacity
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 1)
        ])
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Initialize weights properly
        self._init_weights()
        
    def _init_weights(self):
        """Xavier initialization with proper gains for A100"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Smaller gain for stability
                nn.init.xavier_normal_(m.weight, gain=0.8)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.01)  # Smaller std
                
    def encode(self, users, movies):
        """Encode user-movie pairs to latent space"""
        user_emb = self.user_embedding(users)
        movie_emb = self.movie_embedding(movies)
        
        # Dropout on embeddings
        user_emb = self.embedding_dropout(user_emb)
        movie_emb = self.embedding_dropout(movie_emb)
        
        # Concatenate features
        features = torch.cat([user_emb, movie_emb], dim=1)
        
        # Encode
        encoded = self.encoder(features)
        mu = self.mu_layer(encoded)
        logvar = self.logvar_layer(encoded)
        
        # Clamp logvar for numerical stability
        logvar = torch.clamp(logvar, -15, 10)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick with numerical stability"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            # Prevent NaN with finite check
            std = torch.clamp(std, 1e-8, 50)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def forward(self, users, movies):
        """Forward pass with stability checks"""
        mu, logvar = self.encode(users, movies)
        z = self.reparameterize(mu, logvar)
        
        # Decode
        rating_pred = self.decoder(z)
        
        # Scale to rating range [0.5, 5.0] with sigmoid
        rating_pred = torch.sigmoid(rating_pred) * 4.5 + 0.5
        
        return rating_pred, mu, logvar

class GPUMonitor:
    """Monitor A100 GPU performance and log to W&B"""
    
    def __init__(self):
        self.has_monitoring = HAS_GPU_MONITORING
        if self.has_monitoring:
            self.device_count = nvml.nvmlDeviceGetCount()
        else:
            self.device_count = 0
    
    def get_gpu_metrics(self):
        """Get comprehensive GPU metrics"""
        if not self.has_monitoring or self.device_count == 0:
            return {}
        
        metrics = {}
        for i in range(self.device_count):
            try:
                handle = nvml.nvmlDeviceGetHandleByIndex(i)
                
                # Memory info
                mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                metrics[f'gpu_{i}_memory_used_gb'] = mem_info.used / 1024**3
                metrics[f'gpu_{i}_memory_total_gb'] = mem_info.total / 1024**3
                metrics[f'gpu_{i}_memory_util_percent'] = (mem_info.used / mem_info.total) * 100
                
                # GPU utilization
                util = nvml.nvmlDeviceGetUtilizationRates(handle)
                metrics[f'gpu_{i}_util_percent'] = util.gpu
                
                # Temperature
                temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                metrics[f'gpu_{i}_temp_c'] = temp
                
                # Power
                try:
                    power = nvml.nvmlDeviceGetPowerUsage(handle)
                    metrics[f'gpu_{i}_power_watts'] = power / 1000.0
                except:
                    pass
                
                # Clock speeds
                try:
                    graphics_clock = nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_GRAPHICS)
                    memory_clock = nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_MEM)
                    metrics[f'gpu_{i}_graphics_clock_mhz'] = graphics_clock
                    metrics[f'gpu_{i}_memory_clock_mhz'] = memory_clock
                except:
                    pass
                    
            except Exception as e:
                logger.warning(f"Failed to get GPU {i} metrics: {e}")
                
        return metrics

def setup_wandb(config):
    """Initialize W&B with comprehensive config"""
    
    # Enhanced run name with timestamp and key config
    run_name = f"a100-experiment-{datetime.now().strftime('%m%d-%H%M')}-rmse{config.get('target_rmse', '0.55')}"
    
    wandb.init(
        project="movielens-hybrid-vae-a100",
        name=run_name,
        config=config,
        tags=["a100", "runpod", "hybrid-vae", "production", "optimized"],
        notes=f"A100 training targeting RMSE {config.get('target_rmse', '0.55')} with enhanced monitoring"
    )
    
    # Log system info
    wandb.log({
        "system/cpu_count": psutil.cpu_count(),
        "system/memory_gb": psutil.virtual_memory().total / 1024**3,
        "system/python_version": sys.version,
        "system/pytorch_version": torch.__version__,
        "system/cuda_available": torch.cuda.is_available(),
        "system/cuda_device_count": torch.cuda.device_count(),
    })
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        wandb.log({
            "system/gpu_name": gpu_name,
            "system/is_a100": "A100" in gpu_name,
            "system/cuda_version": torch.version.cuda,
        })
    
    return wandb

def vae_loss(predictions, targets, mu, logvar, kl_weight=0.008, beta_annealing=1.0):
    """
    VAE loss with numerical stability and comprehensive logging
    """
    # Reconstruction loss
    recon_loss = F.mse_loss(predictions.squeeze(), targets, reduction='mean')
    
    # KL divergence with numerical stability
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = torch.clamp(kl_loss, 0, 100)  # Prevent explosion
    
    # Apply annealing and weighting
    weighted_kl = kl_weight * beta_annealing * kl_loss
    total_loss = recon_loss + weighted_kl
    
    # Check for NaN
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        logger.error("NaN/Inf detected in loss computation!")
        return None, recon_loss, kl_loss
    
    return total_loss, recon_loss, kl_loss

def train_epoch(model, train_loader, optimizer, scaler, epoch, config, gpu_monitor):
    """Train for one epoch with comprehensive logging"""
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    num_batches = 0
    
    # Beta annealing for KL weight
    kl_warmup_epochs = config.get('kl_warmup_epochs', 20)
    if epoch < kl_warmup_epochs:
        beta_annealing = epoch / kl_warmup_epochs
    else:
        beta_annealing = 1.0
    
    start_time = time.time()
    
    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        X_batch = X_batch.cuda(non_blocking=True)
        y_batch = y_batch.cuda(non_blocking=True)
        
        users = X_batch[:, 0]
        movies = X_batch[:, 1]
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with autocast():
            predictions, mu, logvar = model(users, movies)
            loss_result = vae_loss(
                predictions, y_batch, mu, logvar, 
                kl_weight=config['kl_weight'], 
                beta_annealing=beta_annealing
            )
            
            if loss_result[0] is None:  # NaN detected
                logger.error(f"NaN loss at epoch {epoch}, batch {batch_idx}")
                continue
                
            loss, recon_loss, kl_loss = loss_result
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Check for gradient issues
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            logger.warning(f"Gradient issue detected at epoch {epoch}, batch {batch_idx}")
            continue
            
        scaler.step(optimizer)
        scaler.update()
        
        # Accumulate losses
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        num_batches += 1
        
        # Log every 100 batches
        if batch_idx % 100 == 0:
            # Get GPU metrics
            gpu_metrics = gpu_monitor.get_gpu_metrics()
            
            # Calculate RMSE
            with torch.no_grad():
                rmse = torch.sqrt(F.mse_loss(predictions.squeeze(), y_batch))
            
            log_dict = {
                "train/batch_loss": loss.item(),
                "train/batch_recon_loss": recon_loss.item(),
                "train/batch_kl_loss": kl_loss.item(),
                "train/batch_rmse": rmse.item(),
                "train/grad_norm": grad_norm.item(),
                "train/beta_annealing": beta_annealing,
                "train/learning_rate": optimizer.param_groups[0]['lr'],
                "train/predictions_mean": predictions.mean().item(),
                "train/predictions_std": predictions.std().item(),
                "train/mu_mean": mu.mean().item(),
                "train/mu_std": mu.std().item(),
                "train/logvar_mean": logvar.mean().item(),
                "train/epoch": epoch,
                "train/batch": batch_idx,
                **gpu_metrics
            }
            
            wandb.log(log_dict)
    
    epoch_time = time.time() - start_time
    
    if num_batches == 0:
        logger.error("No valid batches processed!")
        return float('inf'), float('inf'), float('inf'), epoch_time
    
    avg_loss = total_loss / num_batches
    avg_recon = total_recon / num_batches
    avg_kl = total_kl / num_batches
    
    return avg_loss, avg_recon, avg_kl, epoch_time

def validate_epoch(model, val_loader, epoch, config, gpu_monitor):
    """Validate model with comprehensive metrics"""
    model.eval()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    num_batches = 0
    all_predictions = []
    all_targets = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (X_batch, y_batch) in enumerate(val_loader):
            X_batch = X_batch.cuda(non_blocking=True)
            y_batch = y_batch.cuda(non_blocking=True)
            
            users = X_batch[:, 0]
            movies = X_batch[:, 1]
            
            with autocast():
                predictions, mu, logvar = model(users, movies)
                loss_result = vae_loss(
                    predictions, y_batch, mu, logvar, 
                    kl_weight=config['kl_weight'], 
                    beta_annealing=1.0
                )
                
                if loss_result[0] is None:
                    continue
                    
                loss, recon_loss, kl_loss = loss_result
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            num_batches += 1
            
            # Store for RMSE calculation
            all_predictions.append(predictions.squeeze().cpu())
            all_targets.append(y_batch.cpu())
    
    val_time = time.time() - start_time
    
    if num_batches == 0:
        return float('inf'), float('inf'), val_time
    
    # Calculate comprehensive metrics
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    
    avg_loss = total_loss / num_batches
    avg_recon = total_recon / num_batches
    avg_kl = total_kl / num_batches
    rmse = torch.sqrt(F.mse_loss(all_predictions, all_targets)).item()
    mae = F.l1_loss(all_predictions, all_targets).item()
    
    # Get GPU metrics
    gpu_metrics = gpu_monitor.get_gpu_metrics()
    
    # Log validation metrics
    log_dict = {
        "val/loss": avg_loss,
        "val/recon_loss": avg_recon,
        "val/kl_loss": avg_kl,
        "val/rmse": rmse,
        "val/mae": mae,
        "val/predictions_mean": all_predictions.mean().item(),
        "val/predictions_std": all_predictions.std().item(),
        "val/targets_mean": all_targets.mean().item(),
        "val/targets_std": all_targets.std().item(),
        "val/epoch": epoch,
        "val/time_minutes": val_time / 60,
        **gpu_metrics
    }
    
    wandb.log(log_dict)
    
    return rmse, avg_loss, val_time

def main():
    """Main training function with full W&B integration"""
    
    # Enhanced configuration for A100 optimization
    config = {
        # Model architecture (based on experiment_2 success)
        'n_factors': 160,
        'hidden_dims': [640, 384, 192],
        'latent_dim': 80,
        'dropout_rate': 0.38,
        
        # Training parameters (A100 optimized)
        'batch_size': 6144,  # Large batch for A100
        'n_epochs': 100,
        'lr': 2e-4,  # Conservative start
        'weight_decay': 1e-5,
        'kl_weight': 0.008,  # Very low (critical for stability)
        'kl_warmup_epochs': 20,
        'patience': 15,
        'target_rmse': 0.55,
        
        # A100 optimizations
        'mixed_precision': True,
        'tf32_enabled': True,
        'gradient_clipping': True,
        'num_workers': 8,
        'pin_memory': True,
        
        # Monitoring
        'log_interval': 100,
        'save_interval': 10,
        'gpu_monitoring': True,
    }
    
    print("🚀 Starting A100-Optimized Hybrid VAE Training with W&B")
    print("=" * 60)
    
    # Enable TF32 for A100
    if config['tf32_enabled']:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✅ TF32 enabled for A100 optimization")
    
    # Setup W&B
    wandb_run = setup_wandb(config)
    
    # Initialize GPU monitoring
    gpu_monitor = GPUMonitor()
    
    # Load data
    print("📊 Loading data...")
    data_path = Path('/workspace/data')  # Runpod path
    if not data_path.exists():
        data_path = Path('data/processed')  # Local fallback
        
    train_df = pd.read_csv(data_path / 'train_data.csv')
    val_df = pd.read_csv(data_path / 'val_data.csv')
    
    with open(data_path / 'data_mappings.pkl', 'rb') as f:
        mappings = pickle.load(f)
    
    print(f"   Training samples: {len(train_df):,}")
    print(f"   Validation samples: {len(val_df):,}")
    print(f"   Users: {len(mappings['user_to_index']):,}")
    print(f"   Movies: {len(mappings['movie_to_index']):,}")
    
    # Log data statistics to W&B
    wandb.log({
        "data/train_samples": len(train_df),
        "data/val_samples": len(val_df),
        "data/n_users": len(mappings['user_to_index']),
        "data/n_movies": len(mappings['movie_to_index']),
        "data/sparsity": 1 - (len(train_df) + len(val_df)) / (len(mappings['user_to_index']) * len(mappings['movie_to_index'])),
    })
    
    # Prepare datasets
    print("🔧 Preparing datasets...")
    train_df['user_idx'] = train_df['user_id'].map(mappings['user_to_index'])
    train_df['movie_idx'] = train_df['movie_id'].map(mappings['movie_to_index'])
    val_df['user_idx'] = val_df['user_id'].map(mappings['user_to_index'])
    val_df['movie_idx'] = val_df['movie_id'].map(mappings['movie_to_index'])
    
    # Create tensors
    X_train = torch.LongTensor(train_df[['user_idx', 'movie_idx']].values)
    y_train = torch.FloatTensor(train_df['rating'].values)
    X_val = torch.LongTensor(val_df[['user_idx', 'movie_idx']].values)
    y_val = torch.FloatTensor(val_df['rating'].values)
    
    # Data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )
    
    print(f"   Batch size: {config['batch_size']}")
    print(f"   Train batches: {len(train_loader):,}")
    print(f"   Val batches: {len(val_loader):,}")
    
    # Initialize model
    print("🧠 Initializing model...")
    model = A100OptimizedHybridVAE(
        n_users=len(mappings['user_to_index']),
        n_movies=len(mappings['movie_to_index']),
        n_factors=config['n_factors'],
        hidden_dims=config['hidden_dims'],
        latent_dim=config['latent_dim'],
        dropout_rate=config['dropout_rate']
    ).cuda()
    
    # Count parameters
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Model parameters: {param_count:,}")
    
    # Log model to W&B
    wandb.log({"model/parameters": param_count})
    wandb.watch(model, log="all", log_freq=1000)
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['lr'], 
        weight_decay=config['weight_decay'],
        eps=1e-8
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6
    )
    
    # Mixed precision scaler
    scaler = GradScaler() if config['mixed_precision'] else None
    
    # Training loop
    print("\\n🏋️ Starting training...")
    best_rmse = float('inf')
    patience_counter = 0
    training_start = time.time()
    
    for epoch in range(1, config['n_epochs'] + 1):
        epoch_start = time.time()
        
        # Train
        train_loss, train_recon, train_kl, train_time = train_epoch(
            model, train_loader, optimizer, scaler, epoch, config, gpu_monitor
        )
        
        # Validate
        val_rmse, val_loss, val_time = validate_epoch(
            model, val_loader, epoch, config, gpu_monitor
        )
        
        # Scheduler step
        scheduler.step(val_rmse)
        
        epoch_time = time.time() - epoch_start
        total_time = time.time() - training_start
        
        # Log epoch summary
        log_dict = {
            "epoch/number": epoch,
            "epoch/train_loss": train_loss,
            "epoch/train_recon": train_recon,
            "epoch/train_kl": train_kl,
            "epoch/val_rmse": val_rmse,
            "epoch/val_loss": val_loss,
            "epoch/train_time_min": train_time / 60,
            "epoch/val_time_min": val_time / 60,
            "epoch/total_time_min": epoch_time / 60,
            "epoch/cumulative_time_min": total_time / 60,
            "epoch/lr": optimizer.param_groups[0]['lr'],
            "epoch/is_best": val_rmse < best_rmse,
        }
        
        # Add GPU metrics
        gpu_metrics = gpu_monitor.get_gpu_metrics()
        log_dict.update(gpu_metrics)
        
        wandb.log(log_dict)
        
        # Progress update
        progress = f"Epoch {epoch:3d}/{config['n_epochs']} | "
        progress += f"Train Loss: {train_loss:.4f} | "
        progress += f"Val RMSE: {val_rmse:.4f} | "
        progress += f"Time: {epoch_time/60:.1f}min"
        
        if val_rmse < best_rmse:
            progress += " 🏆 NEW BEST!"
            best_rmse = val_rmse
            patience_counter = 0
            
            # Save best model
            save_path = Path('/workspace/models/hybrid_vae_a100_best.pt')
            save_path.parent.mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_rmse': best_rmse,
                'config': config,
            }, save_path)
            
            # Log model artifact to W&B
            artifact = wandb.Artifact(f'hybrid-vae-a100-best', type='model')
            artifact.add_file(str(save_path))
            wandb.log_artifact(artifact)
            
        else:
            patience_counter += 1
            
        print(progress)
        
        # Early stopping
        if patience_counter >= config['patience']:
            print(f"\\n⏹️ Early stopping triggered after {epoch} epochs")
            print(f"   Best RMSE: {best_rmse:.4f}")
            break
    
    total_training_time = time.time() - training_start
    
    # Final summary
    print("\\n🎉 Training completed!")
    print(f"   Best RMSE: {best_rmse:.4f}")
    print(f"   Total time: {total_training_time/3600:.1f} hours")
    print(f"   Target achieved: {'✅' if best_rmse <= config['target_rmse'] else '❌'}")
    
    # Log final results
    wandb.log({
        "final/best_rmse": best_rmse,
        "final/total_time_hours": total_training_time / 3600,
        "final/target_achieved": best_rmse <= config['target_rmse'],
        "final/epochs_trained": epoch,
    })
    
    # Save final summary
    summary = {
        "best_rmse": best_rmse,
        "total_time_hours": total_training_time / 3600,
        "epochs_trained": epoch,
        "config": config,
        "final_lr": optimizer.param_groups[0]['lr'],
        "target_achieved": best_rmse <= config['target_rmse']
    }
    
    with open('/workspace/models/training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    wandb.finish()
    return best_rmse

if __name__ == "__main__":
    try:
        best_rmse = main()
        print(f"\\n🚀 Training completed successfully! Best RMSE: {best_rmse:.4f}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        wandb.log({"error": str(e)})
        raise