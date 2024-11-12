# Resonance

Neural network signal analysis framework for exploring internal dynamics and patterns.

## Overview

Resonance provides tools for analyzing and visualizing the internal dynamics of transformer-based neural networks through the lens of signal processing. The framework allows for:

- Real-time monitoring of neural activations
- Analysis of frequency components and patterns
- Measurement of computational complexity
- Visualization of internal dynamics

## Project Structure

```
resonance/
├── instrumentation/         # Instrumentation system
│   ├── base.py             # Base classes for instruments
│   ├── frequency.py        # Frequency analysis
│   └── complexity.py       # Complexity analysis
├── transformer.py          # Instrumented transformer implementation
└── signal_analysis.py      # Core signal processing utilities

examples/
└── basic_analysis.py       # Example usage
```

## Core Components

### Instrumentation System

The instrumentation system is designed to be modular and extensible. Each instrument provides specific types of analysis:

- **FrequencyAnalyzer**: Analyzes frequency components and patterns in neural activations
- **ComplexityAnalyzer**: Measures the complexity and structure of information processing

### Instrumented Transformer

A minimal transformer implementation with comprehensive instrumentation capabilities. Key features:

- Measurement points throughout the architecture
- Real-time signal collection
- Integrated analysis and visualization

## Usage

Basic example:

```python
from resonance.transformer import InstrumentedTransformer
from resonance.instrumentation.frequency import FrequencyAnalyzer
from resonance.instrumentation.complexity import ComplexityAnalyzer

# Create model and instruments
model = InstrumentedTransformer()
model.register_instrument(FrequencyAnalyzer())
model.register_instrument(ComplexityAnalyzer())

# Run model with analysis
output, analysis = model(input_sequence)

# Get visualizations
visualizations = model.get_visualizations()
```

See `examples/basic_analysis.py` for a complete working example.

## Development

To set up the development environment:

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the example:
   ```bash
   python examples/basic_analysis.py
   ```

## Next Steps

- Add more specialized instruments for different types of analysis
- Implement real-time visualization interface
- Add support for different model architectures
- Develop pattern detection capabilities
