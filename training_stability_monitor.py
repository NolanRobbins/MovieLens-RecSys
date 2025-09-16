#!/usr/bin/env python3
"""
Training Stability Monitor for SS4Rec
====================================

This module provides comprehensive monitoring and safeguards for SS4Rec training
to prevent and debug numerical instability, memory issues, and CUDA problems.

Usage:
    from training_stability_monitor import TrainingMonitor

    monitor = TrainingMonitor()
    monitor.setup_training_hooks(model, optimizer)
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import psutil
import os
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
from datetime import datetime

class TrainingMonitor:
    """Comprehensive training stability monitor"""

    def __init__(self, log_dir: str = "logs/stability"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = self._setup_logging()
        self.stability_log = []
        self.gradient_history = []
        self.memory_history = []

        # Stability thresholds
        self.max_gradient_norm = 10.0
        self.min_gradient_norm = 1e-8
        self.max_loss_value = 1e6
        self.memory_warning_threshold = 0.9  # 90% of available memory

        # Monitoring flags
        self.nan_detected = False
        self.explosion_detected = False
        self.memory_issues = False

    def _setup_logging(self) -> logging.Logger:
        """Setup dedicated stability logging"""
        logger = logging.getLogger('stability_monitor')
        logger.setLevel(logging.DEBUG)

        # Create file handler
        log_file = self.log_dir / f"stability_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def validate_cuda_setup(self) -> Dict[str, Any]:
        """Validate CUDA setup and detect potential issues"""
        self.logger.info("🔥 Validating CUDA setup...")

        cuda_info = {}

        if not torch.cuda.is_available():
            self.logger.warning("⚠️ CUDA not available - training will use CPU")
            cuda_info['available'] = False
            return cuda_info

        cuda_info['available'] = True
        cuda_info['device_count'] = torch.cuda.device_count()
        cuda_info['current_device'] = torch.cuda.current_device()

        # Get detailed GPU info
        props = torch.cuda.get_device_properties(0)
        cuda_info['name'] = props.name
        cuda_info['total_memory'] = props.total_memory
        cuda_info['major'] = props.major
        cuda_info['minor'] = props.minor

        # Check memory
        memory_allocated = torch.cuda.memory_allocated(0)
        memory_reserved = torch.cuda.memory_reserved(0)
        memory_free = cuda_info['total_memory'] - memory_reserved

        cuda_info['memory_allocated'] = memory_allocated
        cuda_info['memory_reserved'] = memory_reserved
        cuda_info['memory_free'] = memory_free
        cuda_info['memory_utilization'] = memory_reserved / cuda_info['total_memory']

        self.logger.info(f"✅ GPU: {cuda_info['name']}")
        self.logger.info(f"✅ Memory: {memory_free / (1024**3):.1f}GB free / {cuda_info['total_memory'] / (1024**3):.1f}GB total")

        # Check for potential kernel issues
        try:
            # Test basic CUDA operations
            test_tensor = torch.randn(100, 100, device='cuda')
            test_result = torch.matmul(test_tensor, test_tensor.T)
            del test_tensor, test_result
            torch.cuda.empty_cache()

            self.logger.info("✅ CUDA operations test passed")
            cuda_info['operations_test'] = True

        except Exception as e:
            self.logger.error(f"❌ CUDA operations test failed: {e}")
            cuda_info['operations_test'] = False

        return cuda_info

    def monitor_memory_usage(self) -> Dict[str, float]:
        """Monitor system and GPU memory usage"""
        memory_info = {}

        # System memory
        sys_memory = psutil.virtual_memory()
        memory_info['system_used'] = sys_memory.percent
        memory_info['system_available'] = sys_memory.available / (1024**3)  # GB

        # GPU memory
        if torch.cuda.is_available():
            memory_info['gpu_allocated'] = torch.cuda.memory_allocated(0) / (1024**3)  # GB
            memory_info['gpu_reserved'] = torch.cuda.memory_reserved(0) / (1024**3)  # GB
            memory_info['gpu_free'] = (torch.cuda.get_device_properties(0).total_memory -
                                     torch.cuda.memory_reserved(0)) / (1024**3)  # GB

        # Check for memory warnings
        if memory_info['system_used'] > 90:
            self.logger.warning(f"⚠️ High system memory usage: {memory_info['system_used']:.1f}%")
            self.memory_issues = True

        if torch.cuda.is_available() and memory_info['gpu_free'] < 1.0:
            self.logger.warning(f"⚠️ Low GPU memory: {memory_info['gpu_free']:.1f}GB free")
            self.memory_issues = True

        self.memory_history.append(memory_info)
        return memory_info

    def check_tensor_health(self, tensor: torch.Tensor, name: str = "tensor") -> bool:
        """Check tensor for NaN, Inf, and extreme values"""
        if torch.isnan(tensor).any():
            self.logger.error(f"❌ NaN detected in {name}")
            self.nan_detected = True
            return False

        if torch.isinf(tensor).any():
            self.logger.error(f"❌ Inf detected in {name}")
            self.explosion_detected = True
            return False

        # Check for extreme values
        max_val = torch.max(torch.abs(tensor)).item()
        if max_val > 1e6:
            self.logger.warning(f"⚠️ Large values in {name}: max={max_val:.2e}")

        if max_val < 1e-10:
            self.logger.warning(f"⚠️ Very small values in {name}: max={max_val:.2e}")

        return True

    def monitor_gradients(self, model: nn.Module) -> Dict[str, float]:
        """Monitor gradient norms and detect gradient explosion/vanishing"""
        gradient_info = {}
        total_norm = 0.0
        param_count = 0

        gradient_norms = []

        for name, param in model.named_parameters():
            if param.grad is not None:
                # Check gradient health
                if not self.check_tensor_health(param.grad, f"gradient_{name}"):
                    gradient_info[f'{name}_healthy'] = False
                else:
                    gradient_info[f'{name}_healthy'] = True

                # Calculate gradient norm
                param_norm = param.grad.data.norm(2).item()
                gradient_norms.append(param_norm)
                total_norm += param_norm ** 2
                param_count += 1

        if param_count > 0:
            total_norm = total_norm ** 0.5
            gradient_info['total_norm'] = total_norm
            gradient_info['mean_norm'] = np.mean(gradient_norms)
            gradient_info['max_norm'] = np.max(gradient_norms)
            gradient_info['min_norm'] = np.min(gradient_norms)

            # Check for gradient explosion
            if total_norm > self.max_gradient_norm:
                self.logger.error(f"❌ Gradient explosion detected: norm={total_norm:.2e}")
                self.explosion_detected = True
                gradient_info['explosion'] = True
            else:
                gradient_info['explosion'] = False

            # Check for gradient vanishing
            if total_norm < self.min_gradient_norm:
                self.logger.warning(f"⚠️ Gradient vanishing detected: norm={total_norm:.2e}")
                gradient_info['vanishing'] = True
            else:
                gradient_info['vanishing'] = False

            # Log gradient info
            self.logger.debug(f"Gradient norm: {total_norm:.4f} (mean: {gradient_info['mean_norm']:.4f})")

        self.gradient_history.append(gradient_info)
        return gradient_info

    def monitor_loss(self, loss: torch.Tensor, epoch: int, batch: int) -> bool:
        """Monitor loss for stability issues"""
        if not self.check_tensor_health(loss, "loss"):
            return False

        loss_val = loss.item()

        # Check for extreme loss values
        if loss_val > self.max_loss_value:
            self.logger.error(f"❌ Extreme loss value detected: {loss_val:.2e} (epoch {epoch}, batch {batch})")
            return False

        # Log loss info
        self.logger.debug(f"Loss: {loss_val:.6f} (epoch {epoch}, batch {batch})")

        return True

    def setup_gradient_clipping(self, model: nn.Module, max_norm: float = 1.0):
        """Setup gradient clipping to prevent explosion"""
        def clip_gradients():
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            self.logger.debug(f"Applied gradient clipping with max_norm={max_norm}")

        return clip_gradients

    def create_stability_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                                  epoch: int, loss: float) -> str:
        """Create checkpoint when stability issues detected"""
        checkpoint_dir = self.log_dir / "emergency_checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_path = checkpoint_dir / f"emergency_epoch{epoch}_{timestamp}.pt"

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'stability_log': self.stability_log[-10:],  # Last 10 entries
            'gradient_history': self.gradient_history[-10:],
            'memory_history': self.memory_history[-10:],
            'timestamp': timestamp
        }

        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"💾 Emergency checkpoint saved: {checkpoint_path}")

        return str(checkpoint_path)

    def generate_stability_report(self) -> Dict[str, Any]:
        """Generate comprehensive stability report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'nan_detected': self.nan_detected,
            'explosion_detected': self.explosion_detected,
            'memory_issues': self.memory_issues,
            'gradient_history_length': len(self.gradient_history),
            'memory_history_length': len(self.memory_history),
            'stability_log_length': len(self.stability_log)
        }

        # Add gradient statistics
        if self.gradient_history:
            recent_gradients = self.gradient_history[-10:]
            total_norms = [g.get('total_norm', 0) for g in recent_gradients]
            report['recent_gradient_stats'] = {
                'mean_norm': np.mean(total_norms),
                'max_norm': np.max(total_norms),
                'min_norm': np.min(total_norms),
                'std_norm': np.std(total_norms)
            }

        # Add memory statistics
        if self.memory_history:
            recent_memory = self.memory_history[-10:]
            if torch.cuda.is_available():
                gpu_allocated = [m.get('gpu_allocated', 0) for m in recent_memory]
                report['recent_memory_stats'] = {
                    'mean_gpu_allocated': np.mean(gpu_allocated),
                    'max_gpu_allocated': np.max(gpu_allocated),
                    'peak_system_usage': max(m.get('system_used', 0) for m in recent_memory)
                }

        # Save report
        report_path = self.log_dir / f"stability_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"📊 Stability report saved: {report_path}")
        return report

    def should_stop_training(self) -> Tuple[bool, str]:
        """Determine if training should be stopped due to stability issues"""
        if self.nan_detected:
            return True, "NaN values detected in model outputs or gradients"

        if self.explosion_detected:
            return True, "Gradient explosion detected"

        if self.memory_issues and torch.cuda.is_available():
            recent_memory = self.memory_history[-5:] if len(self.memory_history) >= 5 else self.memory_history
            if all(m.get('gpu_free', float('inf')) < 0.5 for m in recent_memory):
                return True, "Persistent GPU memory shortage"

        return False, "Training stability OK"

def apply_numerical_stability_fixes(model: nn.Module) -> nn.Module:
    """Apply numerical stability fixes to SS4Rec model"""
    logger = logging.getLogger('stability_monitor')
    logger.info("🔧 Applying numerical stability fixes...")

    # Find TimeAwareSSM modules and apply fixes
    for name, module in model.named_modules():
        if hasattr(module, 'forward') and 'TimeAware' in str(type(module)):
            logger.info(f"Applying stability fixes to {name}")

            # Add gradient clipping hook
            def grad_clip_hook(grad):
                return torch.clamp(grad, -10.0, 10.0)

            for param in module.parameters():
                if param.requires_grad:
                    param.register_hook(grad_clip_hook)

    return model

# Example usage in training script
def create_monitored_training_loop():
    """Example of how to integrate monitoring into training loop"""

    def training_step(model, data_loader, optimizer, monitor, epoch):
        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(data_loader):
            # Monitor memory before forward pass
            monitor.monitor_memory_usage()

            # Forward pass
            optimizer.zero_grad()
            output = model(batch)
            loss = model.calculate_loss(batch)

            # Monitor loss stability
            if not monitor.monitor_loss(loss, epoch, batch_idx):
                checkpoint_path = monitor.create_stability_checkpoint(model, optimizer, epoch, loss.item())
                raise RuntimeError(f"Training instability detected. Emergency checkpoint: {checkpoint_path}")

            # Backward pass
            loss.backward()

            # Monitor gradients
            grad_info = monitor.monitor_gradients(model)

            # Apply gradient clipping if needed
            if grad_info.get('explosion', False):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                monitor.logger.warning("Applied emergency gradient clipping")

            # Optimizer step
            optimizer.step()

            total_loss += loss.item()

            # Check if training should stop
            should_stop, reason = monitor.should_stop_training()
            if should_stop:
                monitor.create_stability_checkpoint(model, optimizer, epoch, loss.item())
                raise RuntimeError(f"Training stopped: {reason}")

        return total_loss / len(data_loader)

    return training_step