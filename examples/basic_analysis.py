"""Example demonstrating basic usage of the instrumented transformer."""

import torch
import plotly.graph_objects as go
from typing import List

from resonance.transformer import InstrumentedTransformer
from resonance.instrumentation.frequency import FrequencyAnalyzer
from resonance.instrumentation.complexity import ComplexityAnalyzer


def generate_sample_sequence(length: int, vocab_size: int) -> torch.Tensor:
    """Generate a sample sequence with some pattern."""
    # Create a sequence with repeating patterns
    base_pattern = torch.arange(10) % (vocab_size // 10)
    sequence = base_pattern.repeat(length // 10 + 1)[:length]
    return sequence.unsqueeze(0)  # Add batch dimension


def run_analysis(
    model: InstrumentedTransformer,
    sequence: torch.Tensor,
    plot_prefix: str = "analysis"
) -> None:
    """Run analysis and display results."""
    # Forward pass through the model
    output, analysis = model(sequence)
    
    # Get visualizations from instruments
    visualizations = model.get_visualizations()
    
    # Display plots
    for instrument_name, plots in visualizations.items():
        print(f"\nResults from {instrument_name}:")
        
        # Save each plot
        for plot_name, fig in plots.items():
            filename = f"{plot_prefix}_{instrument_name}_{plot_name}.html"
            fig.write_html(filename)
            print(f"Saved {filename}")
        
        # Print analysis metrics
        if instrument_name in analysis:
            print("\nMetrics:")
            for metric_name, value in analysis[instrument_name].items():
                print(f"{metric_name}: {value}")


def main():
    # Create model and instruments
    model = InstrumentedTransformer(
        d_model=256,
        n_heads=8,
        n_layers=4,
        vocab_size=1000
    )
    
    # Register instruments
    model.register_instrument(FrequencyAnalyzer())
    model.register_instrument(ComplexityAnalyzer())
    
    # Generate sample data
    sequence = generate_sample_sequence(length=100, vocab_size=1000)
    
    print("Running analysis on sample sequence...")
    run_analysis(model, sequence)
    
    # Example with different pattern
    print("\nRunning analysis on modified sequence...")
    modified_sequence = torch.sin(torch.arange(100, dtype=torch.float) * 0.1) * 100
    modified_sequence = modified_sequence.long().unsqueeze(0)
    run_analysis(model, modified_sequence, plot_prefix="analysis_modified")


if __name__ == "__main__":
    main()
