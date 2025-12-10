"""Memory efficiency metrics for the Titans model."""

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Union
from collections import defaultdict

from src.models.transformer_arch.titans_transformer import TitansLM, TransformerState


class MemoryEfficiencyMetrics:
    """Track memory efficiency metrics during training."""
    
    def __init__(self):
        """Initialize memory efficiency metrics tracker."""
        self.metrics = {
            'steps': [],
            'memory_usage': [],
            'update_magnitude': [],
            'memory_utilization': [],
            'retrieval_quality': []
        }
    
    def update(
        self, 
        step: int,
        memory_usage: float,
        update_magnitude: float,
        memory_utilization: float,
        retrieval_quality: float
    ) -> None:
        """Update memory efficiency metrics.
        
        Args:
            step: Current training step
            memory_usage: Memory usage in MB
            update_magnitude: Magnitude of memory updates
            memory_utilization: Percentage of memory being utilized
            retrieval_quality: Quality of memory retrieval
        """
        self.metrics['steps'].append(step)
        self.metrics['memory_usage'].append((step, memory_usage))
        self.metrics['update_magnitude'].append((step, update_magnitude))
        self.metrics['memory_utilization'].append((step, memory_utilization))
        self.metrics['retrieval_quality'].append((step, retrieval_quality))
    
    def get_metrics(self) -> Dict[str, List[Any]]:
        """Get all memory efficiency metrics.
        
        Returns:
            Dictionary of memory efficiency metrics
        """
        return self.metrics
