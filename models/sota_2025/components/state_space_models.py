"""
State Space Model Components for SS4Rec
Based on official implementation and paper specifications
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import numpy as np
import logging

# Set up debug logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def log_tensor_stats(tensor: Optional[torch.Tensor], name: str, step: str = "") -> None:
    """Log tensor statistics - smart mode for performance"""
    if tensor is None:
        logger.debug(f"🔍 [{step}] {name}: None")
        return
    
    # Handle different tensor types
    if tensor.dtype in [torch.float32, torch.float64, torch.float16]:
        # Float tensors - check for issues first
        has_nan = torch.isnan(tensor).any()
        has_inf = torch.isinf(tensor).any()
        has_large_values = tensor.abs().max() > 100.0
        
        # Only log detailed stats if there are issues OR it's a key tensor
        should_log_detailed = (has_nan or has_inf or has_large_values or 
                              "initial" in name or "final" in name or 
                              "output" in name or "loss" in name)
        
        if should_log_detailed:
            logger.debug(f"🔍 [{step}] {name}: shape={tensor.shape}, dtype={tensor.dtype}, "
                        f"range=[{tensor.min():.6f}, {tensor.max():.6f}], "
                        f"mean={tensor.mean():.6f}, std={tensor.std():.6f}, "
                        f"nan_count={torch.isnan(tensor).sum()}, inf_count={torch.isinf(tensor).sum()}")
        
        if has_nan:
            logger.error(f"🚨 NaN detected in {name} at step {step}!")
        if has_inf:
            logger.error(f"🚨 Inf detected in {name} at step {step}!")
    else:
        # Integer tensors - basic statistics only for key tensors
        if "initial" in name or "final" in name:
            logger.debug(f"🔍 [{step}] {name}: shape={tensor.shape}, dtype={tensor.dtype}, "
                        f"range=[{tensor.min()}, {tensor.max()}], "
                        f"unique_values={tensor.unique().numel()}")

def apply_gradient_clipping(module: nn.Module, max_norm: float = 1.0) -> float:
    """
    Apply gradient clipping to SS4Rec state space models
    Returns the gradient norm for monitoring
    """
    if hasattr(module, 'parameters'):
        total_norm = torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm)
        return total_norm.item()
    return 0.0

def check_numerical_stability(tensor: torch.Tensor, name: str, max_val: float = 1e3) -> torch.Tensor:
    """
    Check and fix numerical stability issues while preserving SS4Rec architecture
    Uses conservative bounds to maintain mathematical correctness
    """
    # Only check float tensors for NaN/Inf
    if tensor.dtype in [torch.float32, torch.float64, torch.float16]:
        if torch.isnan(tensor).any():
            logger.error(f"🚨 NaN in {name} - replacing with zeros (preserving tensor structure)")
            tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
        
        if torch.isinf(tensor).any():
            logger.error(f"🚨 Inf in {name} - clamping to stable range [{-max_val}, {max_val}]")
            tensor = torch.clamp(tensor, min=-max_val, max=max_val)
        
        # Conservative stability check for SS4Rec matrices
        if tensor.abs().max() > max_val:
            logger.warning(f"⚠️ Large values in {name} - applying conservative clamping")
            tensor = torch.clamp(tensor, min=-max_val, max=max_val)
    
    return tensor


class S5Layer(nn.Module):
    """
    S5 State Space Model Layer (Time-Aware SSM)
    
    Handles irregular time intervals between user interactions
    with variable discretization step sizes.
    """
    
    def __init__(self, 
                 d_model: int,
                 d_state: int = 16,
                 dt_min: float = 0.001,
                 dt_max: float = 0.1,
                 dropout: float = 0.0):
        """
        Initialize S5 layer
        
        Args:
            d_model: Model dimension
            d_state: State dimension  
            dt_min: Minimum discretization step
            dt_max: Maximum discretization step
            dropout: Dropout probability
        """
        super(S5Layer, self).__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.dt_min = dt_min
        self.dt_max = dt_max
        
        # State space parameters
        self.A_log = nn.Parameter(torch.randn(d_state, dtype=torch.float32))
        self.B = nn.Parameter(torch.randn(d_state, d_model, dtype=torch.float32))
        self.C = nn.Parameter(torch.randn(d_model, d_state, dtype=torch.float32))
        self.D = nn.Parameter(torch.randn(d_model, dtype=torch.float32))
        
        # Time-dependent parameters
        self.dt_proj = nn.Linear(d_model, 1, bias=True)
        
        # Normalization and dropout
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters following S5/SS4Rec stability requirements"""
        # Initialize A matrix (diagonal, stable) - following SS4Rec methodology
        with torch.no_grad():
            # SS4Rec uses exponentially spaced negative eigenvalues for stability
            # Range from -1 to -0.001 ensures stability while maintaining expressiveness
            A_diag = torch.logspace(
                start=math.log10(0.001), 
                end=math.log10(1.0), 
                steps=self.d_state
            )
            # Make negative for stability (characteristic of continuous-time systems)
            A_diag = -A_diag
            # Store log for parameterization (SS4Rec uses log-space for numerical stability)
            self.A_log.copy_(torch.log(-A_diag))
        
        # Initialize B and C following SS4Rec paper - smaller scale for stability
        # SS4Rec uses careful initialization to prevent gradient explosion
        nn.init.xavier_uniform_(self.B, gain=0.1)
        nn.init.xavier_uniform_(self.C, gain=0.1)
        
        # Initialize D (skip connection) - small non-zero for SS4Rec architecture
        nn.init.constant_(self.D, 0.01)
        
        # Initialize dt projection following SS4Rec temporal modeling requirements
        # Small range ensures reasonable discretization steps
        nn.init.uniform_(self.dt_proj.weight, -0.01, 0.01)
        nn.init.uniform_(self.dt_proj.bias, -0.01, 0.01)
    
    def forward(self, 
                x: torch.Tensor,
                time_intervals: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with time-aware processing
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            time_intervals: Time intervals [batch_size, seq_len-1]
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        logger.debug(f"🔄 S5Layer.forward: batch_size={batch_size}, seq_len={seq_len}, d_model={d_model}")
        
        # Input validation and stabilization - preserving SS4Rec architecture
        x = check_numerical_stability(x, "input_x_raw")
        x = torch.clamp(x, min=-10.0, max=10.0)  # Prevent extreme inputs
        
        # Log input tensor stats
        log_tensor_stats(x, "input_x", "S5_start")
        log_tensor_stats(time_intervals, "time_intervals", "S5_start")
        
        # Compute time-dependent discretization
        dt = self.dt_proj(x)  # [batch_size, seq_len, 1]
        log_tensor_stats(dt, "dt_raw", "S5_dt_computation")
        
        dt = torch.sigmoid(dt) * (self.dt_max - self.dt_min) + self.dt_min
        log_tensor_stats(dt, "dt_after_sigmoid", "S5_dt_computation")
        
        dt = torch.clamp(dt, min=1e-6, max=1.0)  # Prevent extreme dt values
        dt = check_numerical_stability(dt, "dt_after_clamp")
        log_tensor_stats(dt, "dt_final", "S5_dt_computation")
        
        # Adjust dt based on actual time intervals if provided
        if time_intervals is not None:
            # Ensure time_intervals matches sequence length
            batch_size, current_len = time_intervals.shape
            expected_len = dt.shape[1]  # Get sequence length from dt tensor
            logger.debug(f"🔍 Time intervals adjustment: current_len={current_len}, expected_len={expected_len}")
            
            if current_len < expected_len:
                # Pad with the last time interval value
                padding = time_intervals[:, -1:].expand(batch_size, expected_len - current_len)
                time_intervals_padded = torch.cat([time_intervals, padding], dim=1)
                logger.debug(f"🔍 Padded time_intervals from {current_len} to {expected_len}")
            elif current_len > expected_len:
                # Truncate to match expected length
                time_intervals_padded = time_intervals[:, :expected_len]
                logger.debug(f"🔍 Truncated time_intervals from {current_len} to {expected_len}")
            else:
                time_intervals_padded = time_intervals
                logger.debug(f"🔍 time_intervals length matches: {current_len}")
            
            log_tensor_stats(time_intervals_padded, "time_intervals_padded", "S5_time_adjustment")
            
            # Use proper broadcasting: dt is [batch_size, seq_len, 1], time_intervals is [batch_size, seq_len]
            dt = dt * time_intervals_padded.unsqueeze(-1)
            log_tensor_stats(dt, "dt_after_time_mult", "S5_time_adjustment")
            
            # Check for problematic multiplications
            mult_factor = time_intervals_padded.unsqueeze(-1)
            if (mult_factor > 100).any():
                logger.warning(f"🚨 Large time multiplication factors detected: max={mult_factor.max():.6f}")
            if (mult_factor < 1e-6).any():
                logger.warning(f"🚨 Very small time multiplication factors detected: min={mult_factor.min():.6f}")
        
        # Discretize state space matrices
        log_tensor_stats(self.A_log, "A_log", "S5_discretization")
        A = -torch.exp(self.A_log)  # [d_state]
        A = check_numerical_stability(A, "A_matrix")
        log_tensor_stats(A, "A_matrix", "S5_discretization")
        
        # Check A matrix stability (should be negative for stability)
        if (A > 0).any():
            logger.error(f"🚨 Positive values in A matrix detected! This causes instability. Max A: {A.max():.6f}")
        
        # Clamp dt*A to prevent numerical explosion in exp() - critical for SS4Rec stability
        dt_A = dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        log_tensor_stats(dt_A, "dt_A_before_clamp", "S5_discretization")
        
        # Conservative clamping following SS4Rec numerical requirements
        dt_A = torch.clamp(dt_A, min=-5.0, max=5.0)  # More conservative to prevent overflow
        log_tensor_stats(dt_A, "dt_A_after_clamp", "S5_discretization")
        
        dA = torch.exp(dt_A)  # [batch_size, seq_len, d_state]
        dA = check_numerical_stability(dA, "dA_matrix")
        log_tensor_stats(dA, "dA_matrix", "S5_discretization")
        
        # Log B matrix stats
        log_tensor_stats(self.B, "B_matrix", "S5_discretization")
        dB = (dt.unsqueeze(-1) * self.B.unsqueeze(0).unsqueeze(0))  # [batch_size, seq_len, d_state, d_model]
        dB = check_numerical_stability(dB, "dB_matrix")
        log_tensor_stats(dB, "dB_matrix", "S5_discretization")
        
        # State space computation
        states = torch.zeros(batch_size, self.d_state, device=x.device, dtype=x.dtype)
        log_tensor_stats(states, "initial_states", "S5_computation")
        log_tensor_stats(self.C, "C_matrix", "S5_computation")
        log_tensor_stats(self.D, "D_matrix", "S5_computation")
        
        outputs = []
        
        for t in range(seq_len):
            # Update state: x_{t+1} = A * x_t + B * u_t
            u_t = x[:, t, :]  # [batch_size, d_model]
            # Only log first, last, and problematic timesteps
            if t == 0 or t == seq_len-1 or torch.isnan(u_t).any():
                log_tensor_stats(u_t, f"u_t_{t}", "S5_timestep")
            
            # Fix: squeeze dA to get proper 2D tensor [batch_size, d_state]
            dA_t = dA[:, t, :]
            if dA_t.dim() > 2:
                dA_t = dA_t.squeeze(1)  # Remove middle dimension
            
            # Log einsum inputs for key timesteps only
            dB_t = dB[:, t, :, :]
            
            if t == 0 or t == seq_len-1:
                logger.debug(f"🔍 Einsum 'bsd,bd->bs': dB_t.shape={dB_t.shape}, u_t.shape={u_t.shape}")
            
            state_update = torch.einsum('bsd,bd->bs', dB_t, u_t)
            state_update = check_numerical_stability(state_update, f"state_update_{t}")
            
            # Check state update computation
            state_mult = dA_t * states
            
            states = state_mult + state_update
            states = check_numerical_stability(states, f"states_{t}")
            
            # Compute output: y_t = C * x_t + D * u_t
            if t == 0 or t == seq_len-1:
                logger.debug(f"🔍 Einsum 'bs,ds->bd': states.shape={states.shape}, C.shape={self.C.shape}")
            
            states_contrib = torch.einsum('bs,ds->bd', states, self.C)
            states_contrib = check_numerical_stability(states_contrib, f"states_contrib_{t}")
            
            D_contrib = self.D * u_t
            
            y_t = states_contrib + D_contrib
            y_t = check_numerical_stability(y_t, f"y_t_{t}")
            
            outputs.append(y_t)
        
        output = torch.stack(outputs, dim=1)  # [batch_size, seq_len, d_model]
        output = check_numerical_stability(output, "output_before_norm")
        log_tensor_stats(output, "output_before_norm", "S5_finalization")
        
        # Apply normalization and dropout
        output = self.norm(output)
        output = check_numerical_stability(output, "output_after_norm")
        log_tensor_stats(output, "output_after_norm", "S5_finalization")
        
        output = self.dropout(output)
        log_tensor_stats(output, "output_final", "S5_finalization")
        
        logger.debug(f"✅ S5Layer.forward completed successfully")
        return output


class MambaLayer(nn.Module):
    """
    Mamba State Space Model Layer (Relation-Aware SSM)
    
    Models contextual dependencies between items with
    selective state space mechanisms.
    """
    
    def __init__(self,
                 d_model: int,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2,
                 dropout: float = 0.0):
        """
        Initialize Mamba layer
        
        Args:
            d_model: Model dimension
            d_state: State dimension
            d_conv: Convolution dimension  
            expand: Expansion factor
            dropout: Dropout probability
        """
        super(MambaLayer, self).__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)
        
        # Input projections
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Convolution for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            padding=d_conv - 1,
            groups=self.d_inner,
        )
        
        # SSM parameters (selective)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        
        # State space matrices - following SS4Rec selective SSM methodology
        # A matrix: Use HiPPO initialization as per SS4Rec paper
        # Initialize with stable negative diagonal values
        A_diagonal = torch.logspace(
            start=math.log10(0.001),
            end=math.log10(1.0), 
            steps=d_state
        )
        self.A_log = nn.Parameter(-A_diagonal)  # Negative for stability
        
        # D matrix: Skip connection following SS4Rec architecture
        self.D = nn.Parameter(torch.ones(self.d_inner) * 0.1)  # Smaller scale for stability
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # Normalization and activation
        self.norm = nn.LayerNorm(d_model)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with selective state space processing
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        logger.debug(f"🔄 MambaLayer.forward: batch_size={batch_size}, seq_len={seq_len}, d_model={d_model}")
        
        # Input validation and stabilization - preserving SS4Rec selective SSM architecture
        x = check_numerical_stability(x, "mamba_input_x_raw")
        x = torch.clamp(x, min=-10.0, max=10.0)  # Prevent extreme inputs
        
        # Log input tensor stats
        log_tensor_stats(x, "mamba_input_x", "Mamba_start")
        
        # Input projection and gating
        x_and_res = self.in_proj(x)  # [batch_size, seq_len, 2 * d_inner]
        x_and_res = check_numerical_stability(x_and_res, "mamba_in_proj")
        log_tensor_stats(x_and_res, "mamba_in_proj", "Mamba_projection")
        
        x_proj, res = x_and_res.split([self.d_inner, self.d_inner], dim=-1)
        log_tensor_stats(x_proj, "mamba_x_proj", "Mamba_projection")
        log_tensor_stats(res, "mamba_res", "Mamba_projection")
        
        # Apply convolution for local context
        x_proj_transposed = x_proj.transpose(1, 2)
        log_tensor_stats(x_proj_transposed, "mamba_x_proj_transposed", "Mamba_conv")
        
        x_conv_raw = self.conv1d(x_proj_transposed)[:, :, :seq_len]
        log_tensor_stats(x_conv_raw, "mamba_conv_raw", "Mamba_conv")
        
        x_conv = x_conv_raw.transpose(1, 2)
        x_conv = check_numerical_stability(x_conv, "mamba_conv_after_transpose")
        log_tensor_stats(x_conv, "mamba_conv_before_activation", "Mamba_conv")
        
        x_conv = self.activation(x_conv)
        x_conv = check_numerical_stability(x_conv, "mamba_conv_after_activation")
        log_tensor_stats(x_conv, "mamba_conv_after_activation", "Mamba_conv")
        
        # Selective SSM parameters
        x_ssm = self.x_proj(x_conv)  # [batch_size, seq_len, 2 * d_state]
        x_ssm = check_numerical_stability(x_ssm, "mamba_x_ssm")
        log_tensor_stats(x_ssm, "mamba_x_ssm", "Mamba_ssm_params")
        
        dt = self.dt_proj(x_conv)    # [batch_size, seq_len, d_inner]
        dt = check_numerical_stability(dt, "mamba_dt")
        log_tensor_stats(dt, "mamba_dt", "Mamba_ssm_params")
        
        # Split B and C matrices
        B, C = x_ssm.split([self.d_state, self.d_state], dim=-1)
        log_tensor_stats(B, "mamba_B", "Mamba_ssm_params")
        log_tensor_stats(C, "mamba_C", "Mamba_ssm_params")
        
        # Apply selective state space
        log_tensor_stats(self.A_log, "mamba_A_log", "Mamba_ssm_computation")
        log_tensor_stats(self.D, "mamba_D", "Mamba_ssm_computation")
        
        y = self.selective_scan(x_conv, dt, self.A_log, B, C, self.D)
        y = check_numerical_stability(y, "mamba_selective_scan_output")
        log_tensor_stats(y, "mamba_selective_scan_output", "Mamba_ssm_computation")
        
        # Gating and output projection
        res_activated = self.activation(res)
        log_tensor_stats(res_activated, "mamba_res_activated", "Mamba_output")
        
        y = y * res_activated
        y = check_numerical_stability(y, "mamba_y_after_gating")
        log_tensor_stats(y, "mamba_y_after_gating", "Mamba_output")
        
        output = self.out_proj(y)
        output = check_numerical_stability(output, "mamba_out_proj")
        log_tensor_stats(output, "mamba_out_proj", "Mamba_output")
        
        # Residual connection, normalization, and dropout
        residual_sum = output + x
        log_tensor_stats(residual_sum, "mamba_residual_sum", "Mamba_output")
        
        output = self.norm(residual_sum)
        output = check_numerical_stability(output, "mamba_after_norm")
        log_tensor_stats(output, "mamba_after_norm", "Mamba_output")
        
        output = self.dropout(output)
        log_tensor_stats(output, "mamba_final_output", "Mamba_output")
        
        logger.debug(f"✅ MambaLayer.forward completed successfully")
        return output
    
    def selective_scan(self, 
                      u: torch.Tensor,
                      delta: torch.Tensor, 
                      A: torch.Tensor,
                      B: torch.Tensor,
                      C: torch.Tensor,
                      D: torch.Tensor) -> torch.Tensor:
        """
        Selective state space scan operation
        
        Args:
            u: Input sequence [batch_size, seq_len, d_inner]
            delta: Time steps [batch_size, seq_len, d_inner]  
            A: State matrix [d_state]
            B: Input matrix [batch_size, seq_len, d_state]
            C: Output matrix [batch_size, seq_len, d_state]
            D: Skip connection [d_inner]
            
        Returns:
            Output sequence [batch_size, seq_len, d_inner]
        """
        batch_size, seq_len, d_inner = u.shape
        d_state = A.shape[0]
        logger.debug(f"🔄 selective_scan: batch_size={batch_size}, seq_len={seq_len}, d_inner={d_inner}, d_state={d_state}")
        
        # Log all input tensors
        log_tensor_stats(u, "scan_u", "Scan_inputs")
        log_tensor_stats(delta, "scan_delta", "Scan_inputs")
        log_tensor_stats(A, "scan_A", "Scan_inputs")
        log_tensor_stats(B, "scan_B", "Scan_inputs")
        log_tensor_stats(C, "scan_C", "Scan_inputs")
        log_tensor_stats(D, "scan_D", "Scan_inputs")
        
        # Discretize A matrix
        delta_A_product = delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        log_tensor_stats(delta_A_product, "delta_A_product", "Scan_discretization")
        
        # Conservative clamping to prevent exp overflow - following SS4Rec stability requirements
        delta_A_product = torch.clamp(delta_A_product, min=-5.0, max=5.0)
        
        deltaA = torch.exp(delta_A_product)  # [batch_size, seq_len, d_inner, d_state]
        deltaA = check_numerical_stability(deltaA, "scan_deltaA")
        log_tensor_stats(deltaA, "scan_deltaA", "Scan_discretization")
        
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # [batch_size, seq_len, d_inner, d_state]
        deltaB = check_numerical_stability(deltaB, "scan_deltaB")
        log_tensor_stats(deltaB, "scan_deltaB", "Scan_discretization")
        
        # Initialize state
        x = torch.zeros(batch_size, d_inner, d_state, device=u.device, dtype=u.dtype)
        log_tensor_stats(x, "scan_initial_state", "Scan_computation")
        outputs = []
        
        for t in range(seq_len):
            # Update state
            deltaA_t = deltaA[:, t]
            deltaB_t = deltaB[:, t]
            u_t = u[:, t].unsqueeze(-1)
            
            # State multiplication and update
            state_mult = deltaA_t * x
            state_input = deltaB_t * u_t
            
            x = state_mult + state_input
            x = check_numerical_stability(x, f"scan_state_{t}")
            
            # Compute output
            C_t = C[:, t]
            if t == 0 or t == seq_len-1:
                logger.debug(f"🔍 Einsum 'bid,bd->bi': x.shape={x.shape}, C_t.shape={C_t.shape}")
            
            einsum_result = torch.einsum('bid,bd->bi', x, C_t)
            D_contrib = D * u[:, t]
            
            y = einsum_result + D_contrib
            y = check_numerical_stability(y, f"scan_output_{t}")
            
            outputs.append(y)
        
        final_output = torch.stack(outputs, dim=1)
        log_tensor_stats(final_output, "scan_final_output", "Scan_completion")
        logger.debug(f"✅ selective_scan completed successfully")
        
        return final_output


class SSBlock(nn.Module):
    """
    Combined SS Block with S5 and Mamba layers
    
    Implements the hybrid time-aware + relation-aware
    state space processing from SS4Rec paper.
    """
    
    def __init__(self,
                 d_model: int,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2,
                 dt_min: float = 0.001,
                 dt_max: float = 0.1,
                 dropout: float = 0.0):
        """
        Initialize SS Block
        
        Args:
            d_model: Model dimension
            d_state: State dimension
            d_conv: Convolution dimension
            expand: Expansion factor
            dt_min: Minimum time step
            dt_max: Maximum time step
            dropout: Dropout probability
        """
        super(SSBlock, self).__init__()
        
        self.s5_layer = S5Layer(
            d_model=d_model,
            d_state=d_state,
            dt_min=dt_min,
            dt_max=dt_max,
            dropout=dropout
        )
        
        self.mamba_layer = MambaLayer(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout
        )
        
    def forward(self, 
                x: torch.Tensor,
                time_intervals: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through combined SS block
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            time_intervals: Time intervals [batch_size, seq_len-1]
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        logger.debug(f"🔄 SSBlock.forward: input shape={x.shape}")
        log_tensor_stats(x, "ssblock_input", "SSBlock_start")
        
        # Time-aware processing with S5
        logger.debug(f"🔍 Starting S5Layer processing...")
        x = self.s5_layer(x, time_intervals)
        x = check_numerical_stability(x, "ssblock_after_s5")
        log_tensor_stats(x, "ssblock_after_s5", "SSBlock_s5_complete")
        
        # Relation-aware processing with Mamba  
        logger.debug(f"🔍 Starting MambaLayer processing...")
        x = self.mamba_layer(x)
        x = check_numerical_stability(x, "ssblock_after_mamba")
        log_tensor_stats(x, "ssblock_after_mamba", "SSBlock_mamba_complete")
        
        logger.debug(f"✅ SSBlock.forward completed successfully")
        return x