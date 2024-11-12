"""
Basic test script to validate Fourier Head implementation using CPU only.
Properly handles the inverse softmax output.
"""

import torch
import torch.nn.functional as F
from .vendor.fourier_head.fourier_head import Fourier_Head

class TestNetwork(torch.nn.Module):
    def __init__(self, input_dim=16, output_dim=18, num_frequencies=42):
        super().__init__()
        
        self.classification_head = Fourier_Head(
            dim_input=input_dim,
            dim_output=output_dim,
            num_frequencies=num_frequencies,
            device="cpu"
        )
        
    def forward(self, x):
        # Apply softmax to convert log probabilities to actual probabilities
        logits = self.classification_head(x)
        return F.softmax(logits, dim=-1)

def main():
    print("Starting CPU-only test...")
    
    # Create test model
    model = TestNetwork()
    print("Model created successfully")
    
    # Test with random input
    batch_size = 32
    input_dim = 16
    test_input = torch.randn(batch_size, input_dim)
    print(f"\nInput shape: {test_input.shape}")
    
    try:
        # Get raw logits (inverse softmax)
        raw_output = model.classification_head(test_input)
        print("\nRaw logits (inverse softmax) statistics:")
        print(f"Mean: {raw_output.mean().item():.3f}")
        print(f"Std: {raw_output.std().item():.3f}")
        print(f"Min: {raw_output.min().item():.3f}")
        print(f"Max: {raw_output.max().item():.3f}")
        
        # Get actual probability distribution
        output = model(test_input)
        print("\nProbability distribution statistics:")
        print(f"Mean: {output.mean().item():.3f}")
        print(f"Std: {output.std().item():.3f}")
        print(f"Min: {output.min().item():.3f}")
        print(f"Max: {output.max().item():.3f}")
        
        # Verify probability properties
        sums = output.sum(dim=-1)
        print(f"\nBatch probability sums (should be exactly 1):")
        print(f"Mean sum: {sums.mean().item():.6f}")
        print(f"Min sum: {sums.min().item():.6f}")
        print(f"Max sum: {sums.max().item():.6f}")
        
        # Show distribution shape for first batch item
        print(f"\nFirst batch item distribution:")
        probs = output[0].detach().numpy()
        print("Bins:", ' '.join(f'{p:.3f}' for p in probs))
        
    except Exception as e:
        print(f"Error during forward pass: {str(e)}")
        raise

if __name__ == "__main__":
    main()