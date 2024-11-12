"""Base classes for the neural network instrumentation system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import plotly.graph_objects as go
import torch


@dataclass
class MeasurementPoint:
    """A single measurement from the model."""
    
    name: str  # Name/type of measurement (e.g., 'attention', 'ffn_activation')
    tensor: torch.Tensor  # The actual data being measured
    layer_idx: Optional[int] = None  # Layer index if applicable
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional context


class Instrument(ABC):
    """Base class for model instrumentation plugins."""
    
    def __init__(self, name: str):
        self.name = name
        self.measurements: List[MeasurementPoint] = []
        self.is_active = True
    
    @abstractmethod
    def measure(self, point: MeasurementPoint) -> None:
        """Process and store a measurement point."""
        pass
    
    @abstractmethod
    def analyze(self) -> Dict[str, Any]:
        """Analyze collected measurements and return metrics."""
        pass
    
    @abstractmethod
    def visualize(self) -> Dict[str, go.Figure]:
        """Create visualizations of the collected measurements."""
        pass
    
    def clear(self) -> None:
        """Clear all stored measurements."""
        self.measurements.clear()
    
    def activate(self) -> None:
        """Activate this instrument."""
        self.is_active = True
    
    def deactivate(self) -> None:
        """Deactivate this instrument."""
        self.is_active = False


class InstrumentationManager:
    """Manages a collection of instrumentation plugins."""
    
    def __init__(self):
        self.instruments: Dict[str, Instrument] = {}
    
    def register(self, instrument: Instrument) -> None:
        """Register a new instrumentation plugin."""
        self.instruments[instrument.name] = instrument
    
    def measure(self, name: str, tensor: torch.Tensor, 
                layer_idx: Optional[int] = None,
                metadata: Optional[Dict[str, Any]] = None) -> None:
        """Distribute a measurement to all active instruments."""
        point = MeasurementPoint(name, tensor, layer_idx, metadata or {})
        for instrument in self.instruments.values():
            if instrument.is_active:
                instrument.measure(point)
    
    def analyze(self) -> Dict[str, Dict[str, Any]]:
        """Collect analysis results from all active instruments."""
        return {
            name: inst.analyze()
            for name, inst in self.instruments.items()
            if inst.is_active
        }
    
    def visualize(self) -> Dict[str, Dict[str, go.Figure]]:
        """Collect visualizations from all active instruments."""
        return {
            name: inst.visualize()
            for name, inst in self.instruments.items()
            if inst.is_active
        }
    
    def clear(self) -> None:
        """Clear all instruments."""
        for instrument in self.instruments.values():
            instrument.clear()
    
    def get_instrument(self, name: str) -> Optional[Instrument]:
        """Get a specific instrument by name."""
        return self.instruments.get(name)
    
    def activate(self, name: str) -> None:
        """Activate a specific instrument."""
        if name in self.instruments:
            self.instruments[name].activate()
    
    def deactivate(self, name: str) -> None:
        """Deactivate a specific instrument."""
        if name in self.instruments:
            self.instruments[name].deactivate()
