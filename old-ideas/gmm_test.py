"""
Simple GMM learning example with corrected visualization scaling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from vendor.fourier_head.fourier_head import Fourier_Head

class GMMNetwork(nn.Module):
    """Simple network with encoder and Fourier head."""
    def __init__(
        self,
        input_dim=1,
        hidden_dim=64,
        output_bins=50,
        num_frequencies=16,
        device="cpu"
    ):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.fourier_head = Fourier_Head(
            dim_input=hidden_dim,
            dim_output=output_bins,
            num_frequencies=num_frequencies,
            device=device
        )
        
    def forward(self, x):
        features = self.encoder(x)
        logits = self.fourier_head(features)
        return logits

def generate_gmm_data(n_samples, means=[-0.5, 0.5], stds=[0.15, 0.15], weights=[0.5, 0.5]):
    """Generate samples from a Gaussian mixture model."""
    samples = []
    for _ in range(n_samples):
        idx = np.random.choice(len(means), p=weights)
        sample = np.random.normal(means[idx], stds[idx])
        samples.append(np.clip(sample, -1, 1))
    return torch.tensor(samples, dtype=torch.float32)

def visualize_distribution(model, true_samples, bins=50, save_path=None):
    """Visualize distribution with corrected density scaling."""
    plt.figure(figsize=(12, 6))
    
    # Plot true distribution
    hist, bin_edges = np.histogram(true_samples.numpy(), bins=bins, density=True)
    plt.hist(true_samples.numpy(), bins=bins, density=True, alpha=0.5, 
            label='True Distribution', color='blue')
    
    # Generate points for learned distribution
    x = torch.linspace(-1, 1, bins).unsqueeze(1)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=-1)
    
    # Scale the predicted distribution to match histogram density
    bin_width = (bin_edges[1] - bin_edges[0])
    scaling_factor = 1.0 / bin_width
    
    bin_centers = np.linspace(-1, 1, bins)
    plt.plot(bin_centers, probs.squeeze() * scaling_factor, 'r-', 
            label='Learned Distribution', linewidth=2)
    
    plt.title('True vs Learned GMM Distribution')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def main():
    # Configuration
    n_samples = 10000
    batch_size = 128
    n_epochs = 1000  # You can adjust this
    device = "cpu"
    
    # Create model and optimizer
    model = GMMNetwork(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("Starting GMM training...")
    
    # Training loop
    for epoch in range(n_epochs):
        # Generate batch of data
        x = generate_gmm_data(batch_size)
        x = x.to(device)
        
        # Forward pass
        logits = model(x.unsqueeze(1))
        
        # Convert continuous values to bin indices
        bins = torch.linspace(-1, 1, model.fourier_head.dim_output + 1)
        target_bins = torch.bucketize(x, bins) - 1
        target_bins = torch.clamp(target_bins, 0, model.fourier_head.dim_output - 1)
        
        # Compute loss
        loss = F.cross_entropy(logits, target_bins)
        reg_loss = model.fourier_head.loss_regularization
        total_loss = loss + reg_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss.item():.4f}")
    
    print("\nTraining completed. Generating visualization...")
    
    # Generate validation data and visualize results
    test_samples = generate_gmm_data(1000)
    visualize_distribution(model, test_samples, save_path='gmm_results.png')
    
    print("Results saved to gmm_results.png")

if __name__ == "__main__":
    main()