import torch
import torch.nn as nn
import numpy as np
from collections import deque
import matplotlib.pyplot as plt


# Base class for pattern generators
class PatternGenerator:
    def __init__(self, seq_length: int):
        self.seq_length = seq_length
    
    def generate(self) -> torch.Tensor:
        raise NotImplementedError

# Simple examples:
class SineWaveGenerator(PatternGenerator):
    def __init__(self, seq_length: int, frequency: float, phase: float = 0.0):
        super().__init__(seq_length)
        self.frequency = frequency
        self.phase = phase
    
    def generate(self) -> torch.Tensor:
        t = torch.linspace(0, 2*np.pi, self.seq_length)
        return torch.sin(self.frequency * t + self.phase)

class ResonanceNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        n_oscillators: int,
        pattern_length: int,
        hidden_size: int = 32
    ):
        super().__init__()
        self.n_oscillators = n_oscillators
        self.pattern_length = pattern_length
        
        # Pattern histories
        self.pattern_buffers = [deque(maxlen=pattern_length) for _ in range(n_oscillators)]
        
        # Core network
        self.network = nn.Sequential(
            nn.Linear(input_size + n_oscillators + n_oscillators*pattern_length, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        
        # Oscillator outputs
        self.oscillator_nets = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(n_oscillators)
        ])

    def forward(self, x: torch.Tensor, prev_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Combine input with previous states and pattern histories
        pattern_history = torch.cat([
            torch.tensor(list(buffer)) for buffer in self.pattern_buffers
        ]).flatten()
        
        combined = torch.cat([x, prev_states, pattern_history])
        
        # Process through network
        features = self.network(combined)
        
        # Generate new oscillator states
        new_states = torch.cat([
            net(features) for net in self.oscillator_nets
        ])
        
        # Update pattern buffers
        for i, value in enumerate(new_states):
            self.pattern_buffers[i].append(value.item())
            
        return features, new_states

# Test pattern generator
def test_generators():
    sine_gen = SineWaveGenerator(100, 1.0)
    pattern = sine_gen.generate()
    plt.plot(pattern.numpy())
    plt.show()