"""
AdaptiveResonanceLayer: Core layer implementation for the Resonance architecture.
Implements stable frequency mixing with complexity monitoring at the edge of chaos.
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import pdist

from vendor.fourier_head.fourier_head import Fourier_Head

class AdaptiveResonanceLayer(nn.Module):
    """
    A stable implementation of an adaptive resonance layer using multiple Fourier heads
    with learned attention weights over frequency bands. Includes comprehensive
    complexity monitoring at the edge of chaos.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        min_freq: int = 4,
        max_freq: int = 64,
        num_heads: int = 8,
        dropout: float = 0.1,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        
        # Validate and store parameters
        assert min_freq < max_freq, "min_freq must be less than max_freq"
        assert num_heads >= 2, "need at least 2 heads for frequency mixing"
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        
        # Create geometrically spaced frequency bands
        self.frequency_bands = torch.logspace(
            math.log2(min_freq),
            math.log2(max_freq),
            num_heads,
            base=2,
            device=device
        ).int()
        
        # Shared input projection to reduce parameter count
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU()
        )
        
        # Create Fourier heads for each frequency band
        self.heads = nn.ModuleList([
            Fourier_Head(
                input_dim=input_dim,
                output_dim=output_dim,
                num_frequencies=freq.item(),
                device=device
            )
            for freq in self.frequency_bands
        ])
        
        # Attention mechanism over frequency bands
        self.frequency_attention = nn.Parameter(
            torch.zeros(num_heads),  # Initialize to zeros for softmax
            requires_grad=True
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Tracking metrics
        self.register_buffer('head_usage', torch.zeros(num_heads))
        self.register_buffer('update_steps', torch.tensor(0))
        
        # Storage for complexity monitoring
        self.head_inputs = None
        self.head_outputs = None
        self.last_lyap = None
    
    def get_frequency_distribution(self) -> torch.Tensor:
        """Return current frequency attention distribution."""
        return torch.softmax(self.frequency_attention, dim=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with intermediate storage for complexity monitoring.
        """
        batch_size = x.shape[0]
        
        # Store input for complexity analysis
        self.head_inputs = self.input_projection(x)
        
        # Compute and store all head outputs
        self.head_outputs = torch.stack([
            head(self.head_inputs) for head in self.heads
        ])
        
        # Get attention weights and apply dropout
        weights = self.get_frequency_distribution()
        weights = self.dropout(weights)
        
        # Update usage statistics during training
        if self.training:
            with torch.no_grad():
                self.head_usage = 0.99 * self.head_usage + 0.01 * weights
                self.update_steps += 1
        
        # Final weighted sum
        output = (self.head_outputs * weights.view(-1, 1, 1)).sum(dim=0)
        return output

    def calculate_lz_complexity(self, tensor: torch.Tensor) -> float:
        """
        Calculate Lempel-Ziv complexity using numpy for efficiency.
        Converts tensor to binary sequence based on mean threshold.
        """
        # Convert to numpy and binarize
        arr = tensor.detach().cpu().numpy()
        binary = (arr > arr.mean()).astype(np.int8)
        
        # Flatten and convert to string
        sequence = ''.join(binary.flatten().astype(str))
        
        # Calculate complexity as number of different substrings
        substrings = set()
        n = len(sequence)
        
        for i in range(n):
            for j in range(i + 1, n + 1):
                substrings.add(sequence[i:j])
                
        return len(substrings) / n

    def estimate_lyapunov(self, outputs: torch.Tensor, inputs: torch.Tensor) -> float:
        """
        Estimate largest Lyapunov exponent using nearest neighbors method.
        Uses scipy for efficient distance calculations.
        """
        # Convert to numpy for efficient distance calculations
        out_np = outputs.detach().cpu().numpy()
        in_np = inputs.detach().cpu().numpy()
        
        # Calculate pairwise distances
        d0 = pdist(in_np)
        d1 = pdist(out_np)
        
        # Avoid log(0)
        d0[d0 < 1e-10] = 1e-10
        d1[d1 < 1e-10] = 1e-10
        
        # Estimate Lyapunov exponent
        lyap = np.mean(np.log(d1 / d0))
        return float(lyap)

    def get_complexity_metrics(self) -> Dict[str, float]:
        """
        Calculate comprehensive complexity metrics using stored intermediates.
        Includes Lempel-Ziv complexity, Lyapunov exponent, and edge of chaos score.
        """
        with torch.no_grad():
            # Sample subset for efficiency
            sample_size = min(32, self.head_outputs.size(1))
            sampled_outputs = self.head_outputs[:, :sample_size]
            sampled_inputs = self.head_inputs[:sample_size]
            
            # Calculate Lempel-Ziv complexity
            lz = self.calculate_lz_complexity(sampled_outputs)
            
            # Update Lyapunov estimate periodically
            if self.update_steps % 100 == 0 or self.last_lyap is None:
                lyap = self.estimate_lyapunov(sampled_outputs[0], sampled_inputs)
                self.last_lyap = lyap
            
            # Get weight distribution metrics
            weights = self.get_frequency_distribution()
            weight_entropy = entropy(weights.cpu().numpy())
            
            # Calculate edge of chaos score
            # High when: moderate LZ complexity, small positive Lyapunov, good weight distribution
            edge_score = (
                np.exp(-((lz - 0.5) / 0.2)**2) *          # Target LZ ≈ 0.5
                np.exp(-((self.last_lyap - 0.1) / 0.1)**2) *  # Target Lyapunov ≈ 0.1
                np.exp(-((weight_entropy / np.log(self.num_heads) - 0.5) / 0.2)**2)  # Target entropy
            )
            
            metrics = {
                'lempel_ziv_complexity': lz,
                'lyapunov_exponent': self.last_lyap,
                'weight_entropy': weight_entropy,
                'edge_of_chaos_score': edge_score,
                'head_usage': self.head_usage.tolist(),
                'frequencies': self.frequency_bands.tolist(),
                'steps': self.update_steps.item()
            }
            
            return metrics
    
    def prune_unused_heads(self, threshold: float = 0.01) -> int:
        """Prune heads that have usage below threshold."""
        if not self.training:
            mask = self.head_usage >= threshold
            if mask.all():
                return 0
                
            self.heads = nn.ModuleList([
                h for h, m in zip(self.heads, mask) if m
            ])
            self.frequency_bands = self.frequency_bands[mask]
            self.frequency_attention = nn.Parameter(
                self.frequency_attention[mask]
            )
            self.head_usage = self.head_usage[mask]
            
            return (~mask).sum().item()
        return 0