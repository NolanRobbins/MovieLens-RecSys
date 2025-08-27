#!/usr/bin/env python3
"""
Comprehensive Debug Training Script for SS4Rec NaN Detection

This script enables detailed logging throughout the SS4Rec pipeline to help
identify the source of NaN values during training. It provides extensive
debugging information at every computation step.

Usage:
    python debug_ss4rec_training.py [--config path/to/config.yaml]
"""

import sys
import os
import logging
import warnings
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import numpy as np
import yaml

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def setup_comprehensive_logging() -> None:
    """Set up comprehensive logging for debugging"""
    
    # Create logs directory if it doesn't exist
    log_dir = project_root / "logs" / "debug"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "ss4rec_debug.log", mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set specific logger levels
    logging.getLogger("models.sota_2025.ss4rec").setLevel(logging.DEBUG)
    logging.getLogger("models.sota_2025.components.state_space_models").setLevel(logging.DEBUG)
    logging.getLogger(__name__).setLevel(logging.DEBUG)
    
    # Enable PyTorch anomaly detection
    torch.autograd.set_detect_anomaly(True)
    
    # Enable comprehensive warnings
    warnings.filterwarnings("always")
    
    logger = logging.getLogger(__name__)
    logger.info("🔧 Comprehensive debug logging enabled")
    logger.info(f"📁 Debug logs will be saved to: {log_dir / 'ss4rec_debug.log'}")
    
    return logger

def validate_data_integrity(data_loader, logger: logging.Logger) -> bool:
    """Validate data integrity before training"""
    logger.info("🔍 Validating data integrity...")
    
    issues_found = []
    
    for batch_idx, batch in enumerate(data_loader):
        if batch_idx >= 5:  # Check first 5 batches
            break
            
        users, item_seq, target_items, timestamps, target_ratings = batch
        
        # Check for NaN/Inf in inputs
        for name, tensor in [("users", users), ("item_seq", item_seq), 
                           ("target_items", target_items), ("timestamps", timestamps), 
                           ("target_ratings", target_ratings)]:
            if tensor is not None:
                if torch.isnan(tensor).any():
                    issues_found.append(f"NaN found in {name} at batch {batch_idx}")
                if torch.isinf(tensor).any():
                    issues_found.append(f"Inf found in {name} at batch {batch_idx}")
                if tensor.dtype in [torch.float32, torch.float64]:
                    if tensor.abs().max() > 1e6:
                        issues_found.append(f"Very large values in {name} at batch {batch_idx}: max={tensor.abs().max()}")
        
        # Check timestamp ordering
        if timestamps is not None:
            for seq_idx in range(timestamps.shape[0]):
                seq_timestamps = timestamps[seq_idx]
                # Remove padding (assume 0 is padding)
                valid_timestamps = seq_timestamps[seq_timestamps > 0]
                if len(valid_timestamps) > 1:
                    diffs = valid_timestamps[1:] - valid_timestamps[:-1]
                    if (diffs < 0).any():
                        issues_found.append(f"Non-monotonic timestamps in batch {batch_idx}, sequence {seq_idx}")
    
    if issues_found:
        logger.error("🚨 Data integrity issues found:")
        for issue in issues_found:
            logger.error(f"  - {issue}")
        return False
    else:
        logger.info("✅ Data integrity validation passed")
        return True

def debug_model_initialization(model: nn.Module, logger: logging.Logger) -> None:
    """Debug model initialization"""
    logger.info("🔍 Debugging model initialization...")
    
    param_stats = {}
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_stats[name] = {
                'shape': param.shape,
                'mean': param.data.mean().item(),
                'std': param.data.std().item(),
                'min': param.data.min().item(),
                'max': param.data.max().item(),
                'has_nan': torch.isnan(param.data).any().item(),
                'has_inf': torch.isinf(param.data).any().item()
            }
            
            if param_stats[name]['has_nan']:
                logger.error(f"🚨 NaN in parameter {name} at initialization!")
            if param_stats[name]['has_inf']:
                logger.error(f"🚨 Inf in parameter {name} at initialization!")
            if abs(param_stats[name]['mean']) > 1.0:
                logger.warning(f"⚠️ Large mean in parameter {name}: {param_stats[name]['mean']:.6f}")
            if param_stats[name]['std'] > 2.0:
                logger.warning(f"⚠️ Large std in parameter {name}: {param_stats[name]['std']:.6f}")
    
    logger.info(f"📊 Parameter statistics saved for {len(param_stats)} parameters")

def debug_forward_pass(model: nn.Module, batch, device: torch.device, logger: logging.Logger) -> Dict[str, Any]:
    """Debug a single forward pass"""
    logger.info("🔍 Debugging forward pass...")
    
    users, item_seq, target_items, timestamps, target_ratings = batch
    
    # Move to device
    users = users.to(device)
    item_seq = item_seq.to(device)
    target_items = target_items.to(device) if target_items is not None else None
    timestamps = timestamps.to(device) if timestamps is not None else None
    target_ratings = target_ratings.to(device) if target_ratings is not None else None
    
    model.eval()  # Set to eval mode for debugging
    
    debug_info = {
        'batch_size': users.shape[0],
        'seq_len': item_seq.shape[1],
        'forward_successful': False,
        'output_stats': None,
        'error': None
    }
    
    try:
        # Forward pass with debugging
        logger.info("🚀 Starting forward pass...")
        
        with torch.no_grad():
            output = model(users, item_seq, target_items, timestamps)
            
        # Check output
        if torch.isnan(output).any():
            logger.error("🚨 NaN detected in model output!")
            debug_info['error'] = "NaN in output"
        elif torch.isinf(output).any():
            logger.error("🚨 Inf detected in model output!")
            debug_info['error'] = "Inf in output"
        else:
            logger.info("✅ Forward pass completed without NaN/Inf")
            debug_info['forward_successful'] = True
            
        debug_info['output_stats'] = {
            'shape': output.shape,
            'mean': output.mean().item(),
            'std': output.std().item(),
            'min': output.min().item(),
            'max': output.max().item(),
            'nan_count': torch.isnan(output).sum().item(),
            'inf_count': torch.isinf(output).sum().item()
        }
        
    except Exception as e:
        logger.error(f"🚨 Forward pass failed with error: {e}")
        debug_info['error'] = str(e)
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    return debug_info

def debug_gradient_computation(model: nn.Module, batch, device: torch.device, logger: logging.Logger) -> Dict[str, Any]:
    """Debug gradient computation"""
    logger.info("🔍 Debugging gradient computation...")
    
    users, item_seq, target_items, timestamps, target_ratings = batch
    
    # Move to device
    users = users.to(device)
    item_seq = item_seq.to(device)
    target_items = target_items.to(device) if target_items is not None else None
    timestamps = timestamps.to(device) if timestamps is not None else None
    target_ratings = target_ratings.to(device) if target_ratings is not None else None
    
    model.train()  # Set to train mode
    
    gradient_info = {
        'backward_successful': False,
        'gradient_stats': {},
        'error': None
    }
    
    try:
        # Forward pass
        output = model(users, item_seq, target_items, timestamps)
        
        # Compute loss
        if target_ratings is not None:
            loss = nn.MSELoss()(output, target_ratings)
        else:
            # Dummy loss for testing
            loss = output.mean()
        
        logger.info(f"💰 Loss value: {loss.item():.6f}")
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_stats = {
                    'shape': param.grad.shape,
                    'mean': param.grad.mean().item(),
                    'std': param.grad.std().item(),
                    'min': param.grad.min().item(),
                    'max': param.grad.max().item(),
                    'norm': param.grad.norm().item(),
                    'nan_count': torch.isnan(param.grad).sum().item(),
                    'inf_count': torch.isinf(param.grad).sum().item()
                }
                gradient_info['gradient_stats'][name] = grad_stats
                
                if grad_stats['nan_count'] > 0:
                    logger.error(f"🚨 NaN gradient in {name}!")
                if grad_stats['inf_count'] > 0:
                    logger.error(f"🚨 Inf gradient in {name}!")
                if grad_stats['norm'] > 10.0:
                    logger.warning(f"⚠️ Large gradient norm in {name}: {grad_stats['norm']:.6f}")
        
        gradient_info['backward_successful'] = True
        logger.info("✅ Gradient computation completed")
        
    except Exception as e:
        logger.error(f"🚨 Gradient computation failed: {e}")
        gradient_info['error'] = str(e)
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    return gradient_info

def main():
    """Main debug training function"""
    logger = setup_comprehensive_logging()
    
    logger.info("🚀 Starting SS4Rec Comprehensive Debug Training")
    logger.info(f"🔧 PyTorch version: {torch.__version__}")
    logger.info(f"🔧 CUDA available: {torch.cuda.is_available()}")
    
    try:
        # Import necessary modules
        from models.sota_2025.ss4rec import create_ss4rec_model
        from training.utils.data_loaders import create_ss4rec_dataloaders
        
        # Load configuration
        config_path = project_root / "configs" / "ss4rec.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"📋 Configuration loaded from {config_path}")
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🔧 Using device: {device}")
        
        # Create data loaders
        logger.info("📚 Creating data loaders...")
        train_loader, val_loader, n_users, n_items = create_ss4rec_dataloaders(
            data_path=project_root / "data" / "processed",
            batch_size=4,  # Small batch size for debugging
            max_seq_len=config['model']['max_seq_len']
        )
        
        logger.info(f"📊 Dataset stats: {n_users} users, {n_items} items")
        
        # Validate data integrity
        if not validate_data_integrity(train_loader, logger):
            logger.error("🚨 Data integrity validation failed - stopping debug")
            return
        
        # Create model
        logger.info("🏗️ Creating SS4Rec model...")
        model = create_ss4rec_model(n_users, n_items, config['model'])
        model.to(device)
        
        logger.info(f"📊 Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Debug model initialization
        debug_model_initialization(model, logger)
        
        # Debug forward passes on first few batches
        logger.info("🔍 Debugging forward passes...")
        
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 3:  # Debug first 3 batches
                break
                
            logger.info(f"📦 Processing batch {batch_idx + 1}/3")
            
            # Debug forward pass
            forward_info = debug_forward_pass(model, batch, device, logger)
            
            if forward_info['forward_successful']:
                # Debug gradient computation
                gradient_info = debug_gradient_computation(model, batch, device, logger)
                
                if not gradient_info['backward_successful']:
                    logger.error(f"🚨 Gradient computation failed on batch {batch_idx}")
                    break
            else:
                logger.error(f"🚨 Forward pass failed on batch {batch_idx}")
                break
        
        logger.info("✅ Debug training completed successfully")
        logger.info("🔍 Check the debug log file for detailed tensor statistics")
        
    except Exception as e:
        logger.error(f"🚨 Debug training failed with error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main()