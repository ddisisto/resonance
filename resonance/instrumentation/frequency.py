"""Frequency analysis instrumentation for neural networks."""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import plotly.graph_objects as go
import torch
from scipy import signal

from .base import Instrument, MeasurementPoint


class FrequencyAnalyzer(Instrument):
    """Analyzes frequency components and patterns in neural activations."""
    
    def __init__(self):
        super().__init__("frequency_analyzer")
        # Frequency bands of interest (normalized frequencies)
        self.frequency_bands = {
            'low': (0, 0.1),    # Slow-changing patterns
            'mid': (0.1, 0.5),  # Medium-frequency components
            'high': (0.5, 1.0)  # Fast-changing patterns
        }
        # Storage for analysis results
        self.power_spectra: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}
        self.band_energies: Dict[int, Dict[str, List[float]]] = {}
        self.dominant_freqs: Dict[int, List[float]] = {}
    
    def measure(self, point: MeasurementPoint) -> None:
        """Analyze frequency components of the measurement."""
        if point.layer_idx is None:
            return
            
        # Convert to signal by averaging across feature dimension
        signal_data = point.tensor.mean(dim=-1).cpu().numpy()
        
        # Compute power spectrum
        freqs, power = signal.welch(signal_data, nperseg=min(256, signal_data.shape[-1]))
        
        # Store spectrum
        if point.layer_idx not in self.power_spectra:
            self.power_spectra[point.layer_idx] = []
        self.power_spectra[point.layer_idx].append((freqs, power))
        
        # Compute and store band energies
        if point.layer_idx not in self.band_energies:
            self.band_energies[point.layer_idx] = {band: [] for band in self.frequency_bands}
        
        for band_name, (low, high) in self.frequency_bands.items():
            mask = (freqs >= low) & (freqs < high)
            energy = np.mean(power[mask])
            self.band_energies[point.layer_idx][band_name].append(energy)
        
        # Track dominant frequency
        if point.layer_idx not in self.dominant_freqs:
            self.dominant_freqs[point.layer_idx] = []
        self.dominant_freqs[point.layer_idx].append(freqs[np.argmax(power)])
    
    def analyze(self) -> Dict[str, Any]:
        """Analyze frequency patterns across layers."""
        results = {
            'band_energy_stats': {},
            'dominant_frequency_stats': {},
            'frequency_stability': {}
        }
        
        # Compute statistics for each layer
        for layer_idx in self.power_spectra.keys():
            # Band energy statistics
            results['band_energy_stats'][layer_idx] = {
                band: {
                    'mean': np.mean(energies),
                    'std': np.std(energies)
                }
                for band, energies in self.band_energies[layer_idx].items()
            }
            
            # Dominant frequency statistics
            dom_freqs = self.dominant_freqs[layer_idx]
            results['dominant_frequency_stats'][layer_idx] = {
                'mean': np.mean(dom_freqs),
                'std': np.std(dom_freqs)
            }
            
            # Frequency stability (how consistent are the patterns)
            results['frequency_stability'][layer_idx] = self._compute_stability(layer_idx)
        
        return results
    
    def visualize(self) -> Dict[str, go.Figure]:
        """Create visualizations of frequency analysis."""
        figs = {}
        
        # Power spectrum evolution
        fig = go.Figure()
        for layer_idx, spectra in self.power_spectra.items():
            freqs, power = spectra[-1]  # Most recent measurement
            fig.add_trace(go.Scatter(
                x=freqs,
                y=power,
                name=f'Layer {layer_idx}',
                mode='lines'
            ))
        
        # Add band boundaries as vertical lines
        for band_name, (low, high) in self.frequency_bands.items():
            fig.add_vline(x=low, line_dash="dash", line_color="gray",
                         annotation_text=f"{band_name} start")
            fig.add_vline(x=high, line_dash="dash", line_color="gray",
                         annotation_text=f"{band_name} end")
        
        fig.update_layout(
            title='Power Spectrum Across Layers',
            xaxis_title='Normalized Frequency',
            yaxis_title='Power',
            showlegend=True
        )
        figs['power_spectrum'] = fig
        
        # Band energy evolution
        fig = go.Figure()
        for layer_idx in self.band_energies:
            for band_name, energies in self.band_energies[layer_idx].items():
                fig.add_trace(go.Scatter(
                    y=energies,
                    name=f'Layer {layer_idx} - {band_name}',
                    mode='lines'
                ))
        
        fig.update_layout(
            title='Band Energy Evolution',
            xaxis_title='Time Step',
            yaxis_title='Energy',
            showlegend=True
        )
        figs['band_energy'] = fig
        
        return figs
    
    def _compute_stability(self, layer_idx: int) -> float:
        """Compute stability score for frequency patterns."""
        # Use standard deviation of dominant frequencies as a stability metric
        # Lower std = more stable
        dom_freqs = self.dominant_freqs[layer_idx]
        return float(1.0 / (np.std(dom_freqs) + 1e-6))  # Add epsilon to avoid division by zero
    
    def clear(self) -> None:
        """Clear all stored measurements."""
        super().clear()
        self.power_spectra.clear()
        self.band_energies.clear()
        self.dominant_freqs.clear()
