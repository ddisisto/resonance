"""A minimal transformer implementation with instrumentation capabilities."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from .instrumentation.base import InstrumentationManager


class InstrumentedAttention(nn.Module):
    """Self-attention with instrumentation hooks."""
    
    def __init__(self, d_model: int, n_heads: int, manager: InstrumentationManager):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.manager = manager
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        layer_idx: Optional[int] = None
    ) -> torch.Tensor:
        batch_size = query.size(0)
        
        # Linear transformations
        q = self.q_linear(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.d_k, dtype=torch.float)
        )
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        
        attention = F.softmax(scores, dim=-1)
        
        # Record attention pattern
        self.manager.measure(
            name="attention",
            tensor=attention,
            layer_idx=layer_idx,
            metadata={"n_heads": self.n_heads}
        )
        
        # Apply attention to values
        output = torch.matmul(attention, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.out(output)


class InstrumentedTransformerBlock(nn.Module):
    """Transformer block with instrumentation capabilities."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        manager: InstrumentationManager,
        layer_idx: Optional[int] = None
    ):
        super().__init__()
        self.manager = manager
        self.layer_idx = layer_idx
        
        self.attention = InstrumentedAttention(d_model, n_heads, manager)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        attended = self.attention(x, x, x, mask, self.layer_idx)
        x = self.norm1(x + attended)
        
        # Record normalized attention output
        self.manager.measure(
            name="attention_output",
            tensor=x,
            layer_idx=self.layer_idx
        )
        
        # Feedforward
        ff_out = self.ff(x)
        
        # Record FFN activations
        self.manager.measure(
            name="ffn_activation",
            tensor=ff_out,
            layer_idx=self.layer_idx
        )
        
        output = self.norm2(x + ff_out)
        
        # Record layer output
        self.manager.measure(
            name="layer_output",
            tensor=output,
            layer_idx=self.layer_idx
        )
        
        return output


class InstrumentedTransformer(nn.Module):
    """Minimal transformer with comprehensive instrumentation capabilities."""
    
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 1024,
        vocab_size: int = 1000,
    ):
        super().__init__()
        self.manager = InstrumentationManager()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                InstrumentedTransformerBlock(
                    d_model, n_heads, d_ff, self.manager, layer_idx=i
                )
                for i in range(n_layers)
            ]
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        self.manager.clear()  # Clear previous measurements
        
        x = self.embedding(x)
        
        # Record embedding output
        self.manager.measure(
            name="embedding_output",
            tensor=x
        )
        
        for layer in self.layers:
            x = layer(x, mask)
        
        output = self.fc_out(x)
        
        # Record final output
        self.manager.measure(
            name="model_output",
            tensor=output
        )
        
        # Collect analysis results
        analysis = self.manager.analyze()
        
        return output, analysis
    
    def register_instrument(self, instrument: 'Instrument') -> None:
        """Register a new instrumentation plugin."""
        self.manager.register(instrument)
    
    def get_visualizations(self) -> Dict[str, Dict[str, Any]]:
        """Get visualizations from all active instruments."""
        return self.manager.visualize()
