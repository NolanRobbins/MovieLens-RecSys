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
        """Initialize parameters following S5 paper"""
        # Initialize A matrix (diagonal, stable)
        with torch.no_grad():
            A = torch.diag(torch.linspace(-1, -0.1, self.d_state))
            self.A_log.copy_(torch.log(-A.diag()))
        
        # Initialize B and C
        nn.init.xavier_uniform_(self.B)
        nn.init.xavier_uniform_(self.C)
        
        # Initialize dt projection
        self.dt_proj.weight.data.uniform_(-0.1, 0.1)
        self.dt_proj.bias.data.uniform_(-0.1, 0.1)
    
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
        
        # Compute time-dependent discretization
        dt = self.dt_proj(x)  # [batch_size, seq_len, 1]
        dt = torch.sigmoid(dt) * (self.dt_max - self.dt_min) + self.dt_min
        
        # Adjust dt based on actual time intervals if provided
        if time_intervals is not None:
            # Ensure time_intervals matches sequence length
            batch_size, current_len = time_intervals.shape
            expected_len = dt.shape[1]  # Get sequence length from dt tensor
            
            if current_len < expected_len:
                # Pad with the last time interval value
                padding = time_intervals[:, -1:].expand(batch_size, expected_len - current_len)
                time_intervals_padded = torch.cat([time_intervals, padding], dim=1)
            elif current_len > expected_len:
                # Truncate to match expected length
                time_intervals_padded = time_intervals[:, :expected_len]
            else:
                time_intervals_padded = time_intervals
            
            dt = dt.squeeze(-1) * time_intervals_padded.unsqueeze(-1)
        
        # Discretize state space matrices
        A = -torch.exp(self.A_log)  # [d_state]
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # [batch_size, seq_len, d_state]
        dB = (dt.unsqueeze(-1) * self.B.unsqueeze(0).unsqueeze(0))  # [batch_size, seq_len, d_state, d_model]
        
        # State space computation
        states = torch.zeros(batch_size, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        
        for t in range(seq_len):
            # Update state: x_{t+1} = A * x_t + B * u_t
            u_t = x[:, t, :]  # [batch_size, d_model]
            states = dA[:, t, :] * states + torch.einsum('bsd,bd->bs', dB[:, t, :, :], u_t)
            
            # Compute output: y_t = C * x_t + D * u_t
            y_t = torch.einsum('ds,bs->bd', self.C, states) + self.D * u_t
            outputs.append(y_t)
        
        output = torch.stack(outputs, dim=1)  # [batch_size, seq_len, d_model]
        
        # Apply normalization and dropout
        output = self.norm(output)
        output = self.dropout(output)
        
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
        
        # State space matrices
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
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
        
        # Input projection and gating
        x_and_res = self.in_proj(x)  # [batch_size, seq_len, 2 * d_inner]
        x_proj, res = x_and_res.split([self.d_inner, self.d_inner], dim=-1)
        
        # Apply convolution for local context
        x_conv = self.conv1d(x_proj.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        x_conv = self.activation(x_conv)
        
        # Selective SSM parameters
        x_ssm = self.x_proj(x_conv)  # [batch_size, seq_len, 2 * d_state]
        dt = self.dt_proj(x_conv)    # [batch_size, seq_len, d_inner]
        
        # Split B and C matrices
        B, C = x_ssm.split([self.d_state, self.d_state], dim=-1)
        
        # Apply selective state space
        y = self.selective_scan(x_conv, dt, self.A_log, B, C, self.D)
        
        # Gating and output projection
        y = y * self.activation(res)
        output = self.out_proj(y)
        
        # Residual connection, normalization, and dropout
        output = self.norm(output + x)
        output = self.dropout(output)
        
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
        
        # Discretize A matrix
        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # [batch_size, seq_len, d_inner, d_state]
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # [batch_size, seq_len, d_inner, d_state]
        
        # Initialize state
        x = torch.zeros(batch_size, d_inner, d_state, device=u.device, dtype=u.dtype)
        outputs = []
        
        for t in range(seq_len):
            # Update state
            x = deltaA[:, t] * x + deltaB[:, t] * u[:, t].unsqueeze(-1)
            
            # Compute output
            y = torch.einsum('bid,bd->bi', x, C[:, t]) + D * u[:, t]
            outputs.append(y)
        
        return torch.stack(outputs, dim=1)


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
        # Time-aware processing with S5
        x = self.s5_layer(x, time_intervals)
        
        # Relation-aware processing with Mamba  
        x = self.mamba_layer(x)
        
        return x