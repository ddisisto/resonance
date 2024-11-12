"""Basic signal analysis utilities for neural network activations."""

import numpy as np
from scipy import signal


def compute_power_spectrum(activation_sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the power spectrum of an activation sequence.
    
    Args:
        activation_sequence: Time series of activations to analyze
        
    Returns:
        frequencies: Array of frequency components
        power: Power at each frequency
    """
    frequencies, power = signal.welch(activation_sequence)
    return frequencies, power


def measure_coherence(signal1: np.ndarray, signal2: np.ndarray) -> float:
    """Measure coherence between two activation sequences.
    
    Args:
        signal1: First activation sequence
        signal2: Second activation sequence
        
    Returns:
        coherence: Coherence measure between the signals
    """
    _, coherence = signal.coherence(signal1, signal2)
    return np.mean(coherence)  # Return average coherence across frequencies
