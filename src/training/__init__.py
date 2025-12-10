"""
Training utilities for the Titans model, including trainers and evaluation metrics.

This package provides classes and functions for training and evaluating
Titans language models, tracking metrics, and managing training processes.
"""

from src.training.metrics import (
    LanguageModelMetrics,
    MemoryMetrics,
    GenerationMetrics,
    MetricsTracker,
    compute_metrics_from_batch,
)

from src.training.trainer import (
    Trainer,
    DistributedTrainer,
)

__all__ = [
    # Metrics
    "LanguageModelMetrics",
    "MemoryMetrics",
    "GenerationMetrics",
    "MetricsTracker",
    "compute_metrics_from_batch",
    
    # Trainers
    "Trainer",
    "DistributedTrainer",
]