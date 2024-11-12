# Resonance

*Neural networks operating at the edge of chaos*

## Overview
Resonance is an experimental neural network architecture that adaptively tunes each layer's frequency distribution to maintain optimal complexity. By operating each layer at its own "edge of chaos", the network aims to create efficient paths for information flow, potentially offering better performance on reasoning tasks with fewer parameters.

## Key Concepts
- Adaptive Fourier heads in each layer
- Dynamic frequency distribution
- Layer-wise complexity optimization
- Task-specific architecture generation

## Current Status
Early development / proof of concept. Core components:
- Basic ResonanceNetwork implementation
- Architecture generation for different task types
- Simple complexity tracking framework

## Installation
```bash
git clone https://github.com/yourusername/resonance.git
cd resonance
pip install -r requirements.txt
```

## Quick Start
```python
from resonance import ResonanceArchitectureGenerator, ResonanceNetwork

# Create a test network
generator = ResonanceArchitectureGenerator(input_dim=64, output_dim=10)
configs = generator.generate_architecture("reasoning")
model = ResonanceNetwork(configs)
```

## TODOs
- [ ] Implement proper complexity metrics (Lempel-Ziv, Lyapunov, compression)
- [ ] Add dynamic frequency adjustment during training
- [ ] Implement logging and visualization tools
- [ ] Add training loops and evaluation code
- [ ] Create proper test suite
- [ ] Document theoretical foundations
- [ ] Benchmark against baseline architectures

## Dependencies
- PyTorch >= 2.1.0
- NumPy >= 1.24.0
- einops >= 0.7.0

## License
MIT

## Citation
If you use this work, please cite:
```
@misc{resonance2024,
  title={Resonance: Adaptive Frequency Distribution in Fourier Neural Networks},
  author={[Your Names]},
  year={2024},
  note={Work in Progress}
}
```

## Contributing
This is an experimental research project. Issues and pull requests welcome!
