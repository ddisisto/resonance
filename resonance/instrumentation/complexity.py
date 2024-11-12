"""Complexity analysis instrumentation for neural networks."""

from typing import Dict, Any, List, Optional
import numpy as np
import plotly.graph_objects as go
import torch
from scipy.stats import entropy

from .base import Instrument, MeasurementPoint


class ComplexityAnalyzer(Instrument):
    """Analyzes the complexity and structure of neural network patterns."""
    
    def __init__(self):
        super().__init__("complexity_analyzer")
        # Storage for measurements and analysis
        self.activation_history: Dict[int, List[torch.Tensor]] = {}
        self.complexity_scores: Dict[int, List[float]] = {}
        self.structure_metrics: Dict[int, List[Dict[str, float]]] = {}
        self.window_size = 50  # Number of patterns to keep in memory
    
    def measure(self, point: MeasurementPoint) -> None:
        """Analyze complexity of activation patterns."""
        if point.layer_idx is None:
            return
            
        # Store activation pattern
        if point.layer_idx not in self.activation_history:
            self.activation_history[point.layer_idx] = []
        
        # Average across batch dimension if present
        pattern = point.tensor.mean(dim=0) if point.tensor.dim() > 2 else point.tensor
        pattern = pattern.cpu()
        
        self.activation_history[point.layer_idx].append(pattern)
        
        # Keep only recent patterns
        if len(self.activation_history[point.layer_idx]) > self.window_size:
            self.activation_history[point.layer_idx].pop(0)
        
        # Compute complexity metrics
        metrics = self._compute_complexity_metrics(pattern)
        
        # Store complexity score
        if point.layer_idx not in self.complexity_scores:
            self.complexity_scores[point.layer_idx] = []
        self.complexity_scores[point.layer_idx].append(metrics['overall_complexity'])
        
        # Store structure metrics
        if point.layer_idx not in self.structure_metrics:
            self.structure_metrics[point.layer_idx] = []
        self.structure_metrics[point.layer_idx].append(metrics)
    
    def analyze(self) -> Dict[str, Any]:
        """Analyze complexity patterns and trends."""
        results = {
            'complexity_trends': {},
            'structure_evolution': {},
            'layer_interactions': {}
        }
        
        # Analyze complexity trends for each layer
        for layer_idx, scores in self.complexity_scores.items():
            results['complexity_trends'][layer_idx] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'trend': float(np.polyfit(range(len(scores)), scores, 1)[0])
            }
        
        # Analyze structure evolution
        for layer_idx, metrics in self.structure_metrics.items():
            results['structure_evolution'][layer_idx] = {
                metric: {
                    'mean': float(np.mean([m[metric] for m in metrics])),
                    'std': float(np.std([m[metric] for m in metrics]))
                }
                for metric in metrics[0].keys()
            }
        
        # Analyze layer interactions if we have multiple layers
        if len(self.activation_history) > 1:
            results['layer_interactions'] = self._analyze_layer_interactions()
        
        return results
    
    def visualize(self) -> Dict[str, go.Figure]:
        """Create visualizations of complexity analysis."""
        figs = {}
        
        # Complexity evolution over time
        fig = go.Figure()
        for layer_idx, scores in self.complexity_scores.items():
            fig.add_trace(go.Scatter(
                y=scores,
                name=f'Layer {layer_idx}',
                mode='lines'
            ))
        
        fig.update_layout(
            title='Complexity Evolution',
            xaxis_title='Time Step',
            yaxis_title='Complexity Score',
            showlegend=True
        )
        figs['complexity_evolution'] = fig
        
        # Structure metrics evolution
        fig = go.Figure()
        for layer_idx, metrics in self.structure_metrics.items():
            for metric_name in metrics[0].keys():
                if metric_name != 'overall_complexity':  # Already shown in previous plot
                    values = [m[metric_name] for m in metrics]
                    fig.add_trace(go.Scatter(
                        y=values,
                        name=f'Layer {layer_idx} - {metric_name}',
                        mode='lines'
                    ))
        
        fig.update_layout(
            title='Structure Metrics Evolution',
            xaxis_title='Time Step',
            yaxis_title='Metric Value',
            showlegend=True
        )
        figs['structure_evolution'] = fig
        
        return figs
    
    def _compute_complexity_metrics(self, pattern: torch.Tensor) -> Dict[str, float]:
        """Compute various complexity metrics for a pattern."""
        # Convert to numpy for calculations
        data = pattern.numpy()
        
        # Gradient-based complexity
        gradients = np.gradient(data)
        gradient_complexity = np.mean(np.abs(gradients))
        
        # Distribution-based complexity
        hist, _ = np.histogram(data, bins=20, density=True)
        distribution_complexity = entropy(hist + 1e-10)  # Add small constant to avoid log(0)
        
        # Structure metrics
        sparsity = np.mean(np.abs(data) < 0.1)  # Proportion of near-zero values
        peak_ratio = np.max(np.abs(data)) / (np.mean(np.abs(data)) + 1e-10)
        
        # Combine metrics
        overall_complexity = (gradient_complexity + distribution_complexity) / 2
        
        return {
            'overall_complexity': float(overall_complexity),
            'gradient_complexity': float(gradient_complexity),
            'distribution_complexity': float(distribution_complexity),
            'sparsity': float(sparsity),
            'peak_ratio': float(peak_ratio)
        }
    
    def _analyze_layer_interactions(self) -> Dict[str, float]:
        """Analyze interactions between layers based on complexity patterns."""
        n_layers = len(self.activation_history)
        interactions = {}
        
        # Compare complexity patterns between layers
        for i in range(n_layers):
            for j in range(i + 1, n_layers):
                scores_i = self.complexity_scores[i]
                scores_j = self.complexity_scores[j]
                
                # Compute correlation between complexity scores
                correlation = np.corrcoef(scores_i, scores_j)[0, 1]
                interactions[f'correlation_{i}_{j}'] = float(correlation)
                
                # Compute complexity ratio
                ratio = np.mean(scores_i) / (np.mean(scores_j) + 1e-10)
                interactions[f'complexity_ratio_{i}_{j}'] = float(ratio)
        
        return interactions
    
    def clear(self) -> None:
        """Clear all stored measurements."""
        super().clear()
        self.activation_history.clear()
        self.complexity_scores.clear()
        self.structure_metrics.clear()
