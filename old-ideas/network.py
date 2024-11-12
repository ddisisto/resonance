"""
Resonance Network: A neural network architecture using adaptive frequency distributions
to maintain optimal complexity at each layer.

Each layer operates at its own "edge of chaos" through stable frequency mixing,
potentially creating more efficient paths for information processing.
"""

from typing import List, Tuple, Optional, Dict, Union
import torch
import torch.nn as nn
from torch import Tensor

from layer import AdaptiveResonanceLayer

class ResonanceNetwork(nn.Module):
    """
    A neural network composed of AdaptiveResonanceLayers, optimized for
    maintaining each layer at its own optimal complexity level.
    """
    
    def __init__(
        self,
        layer_configs: List[Tuple[int, int]],  # List of (in_dim, out_dim)
        min_freq: int = 4,
        max_freq: int = 64,
        heads_per_layer: Optional[List[int]] = None,
        dropout: float = 0.1,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        
        if heads_per_layer is None:
            heads_per_layer = [8] * len(layer_configs)
        
        assert len(heads_per_layer) == len(layer_configs), \
            "Must specify heads for each layer"
            
        # Build layers
        self.layers = nn.ModuleList([
            AdaptiveResonanceLayer(
                input_dim=in_dim,
                output_dim=out_dim,
                min_freq=min_freq,
                max_freq=max_freq,
                num_heads=num_heads,
                dropout=dropout,
                device=device
            )
            for (in_dim, out_dim), num_heads 
            in zip(layer_configs, heads_per_layer)
        ])
        
        # Initialize metrics tracking
        self.register_buffer('step_counter', torch.tensor(0))
        
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through all layers."""
        for layer in self.layers:
            x = layer(x)
        
        if self.training:
            self.step_counter += 1
            
        return x
    
    def get_complexity_profile(self) -> List[Dict[str, float]]:
        """Return complexity metrics for all layers."""
        return [layer.get_complexity_metrics() for layer in self.layers]
    
    def prune_unused_heads(self, threshold: float = 0.01) -> List[int]:
        """Prune unused heads across all layers."""
        return [layer.prune_unused_heads(threshold) for layer in self.layers]
    
    def get_frequency_distributions(self) -> List[Tensor]:
        """Get current frequency attention distributions for all layers."""
        return [layer.get_frequency_distribution() for layer in self.layers]

class ResonanceClassifier(ResonanceNetwork):
    """
    Convenience class for classification tasks, adds softmax output.
    """
    
    def forward(self, x: Tensor) -> Tensor:
        x = super().forward(x)
        return torch.softmax(x, dim=-1)

def create_resonance_model(
    input_dim: int,
    output_dim: int,
    hidden_dims: Union[List[int], int],
    **kwargs
) -> ResonanceNetwork:
    """
    Convenience function to create a ResonanceNetwork with specified dimensions.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        hidden_dims: Either a list of hidden dimensions or a single int for
                    uniform hidden layers
    """
    if isinstance(hidden_dims, int):
        hidden_dims = [hidden_dims] * 3  # Default to 3 hidden layers
        
    dims = [input_dim] + hidden_dims + [output_dim]
    layer_configs = list(zip(dims[:-1], dims[1:]))
    
    return ResonanceNetwork(layer_configs, **kwargs)

def create_narrow_deep_network(
    input_dim: int,
    output_dim: int,
    min_width: int = 32,
    depth: int = 24,
    **kwargs
) -> ResonanceNetwork:
    """
    Creates a narrow, deep network optimized for reasoning tasks.
    Network narrows towards the middle then widens again.
    """
    mid = depth // 2
    
    # Create narrowing then widening dimension pattern
    dims = [input_dim]
    
    # Narrow down
    for i in range(mid):
        next_dim = max(min_width, 
                      int(input_dim * torch.exp(torch.tensor(-(i/mid)**2))))
        dims.append(next_dim)
    
    # Widen up
    for i in range(mid, depth-1):
        factor = (i - mid) / (depth - mid - 1)
        next_dim = int(min_width + (output_dim - min_width) * factor)
        dims.append(next_dim)
        
    dims.append(output_dim)
    
    layer_configs = list(zip(dims[:-1], dims[1:]))
    
    return ResonanceNetwork(layer_configs, **kwargs)

if __name__ == "__main__":
    # Simple test/demo
    model = create_resonance_model(
        input_dim=64,
        output_dim=10,
        hidden_dims=[128, 96, 64],
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    x = torch.randn(32, 64)  # batch_size=32, input_dim=64
    y = model(x)
    
    print(f"Model created successfully!")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Print complexity profile
    print("\nComplexity profile:")
    for i, metrics in enumerate(model.get_complexity_profile()):
        print(f"Layer {i}:", metrics)