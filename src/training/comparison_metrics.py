"""Metrics for comparing multiple models."""

import torch
import numpy as np
from typing import Dict, Any, List, Optional
from collections import defaultdict

from src.training.metrics import MetricsTracker, LanguageModelMetrics


class ComparativeMetrics:
    """Metrics tracker for comparing multiple models."""
    
    def __init__(self, model_names: List[str]):
        """Initialize comparative metrics tracker.
        
        Args:
            model_names: List of model names to track
        """
        self.model_names = model_names
        self.metrics = {name: MetricsTracker() for name in model_names}
        self.history = {name: defaultdict(list) for name in model_names}
        self.inference_metrics = {name: {"time_per_token": [], "memory_usage": []} for name in model_names}
        self.step_counter = {name: 0 for name in model_names}
    
    def update(self, model_name: str, loss: torch.Tensor, logits: torch.Tensor, targets: torch.Tensor, step: Optional[int] = None) -> None:
        """Update metrics for a model.
        
        Args:
            model_name: Name of the model
            loss: Loss tensor
            logits: Model output logits
            targets: Ground truth targets
            step: Current step number (optional)
        """
        if model_name not in self.model_names:
            raise ValueError(f"Unknown model name: {model_name}")
        
        # Update Metrics
        loss = LanguageModelMetrics.compute_loss(logits, targets)
        perplexity = LanguageModelMetrics.compute_perplexity(loss)
        accuracy, _ = LanguageModelMetrics.compute_accuracy(logits, targets)

        metric_dict = {
            "loss": loss,
            "perplexity": perplexity,
            "accuracy": accuracy  # Fixed typo from "accuray" to "accuracy"
        }

        self.metrics[model_name].update(metrics_dict=metric_dict)
        
        # Update history
        if step is None:
            step = self.step_counter[model_name]
            self.step_counter[model_name] += 1
        
        # Record metrics in history
        curr_metrics = self.metrics[model_name].get_all_last()
        for metric_name, value in curr_metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()  # Convert tensor to float
            # Ensure (step, value) pairs are stored
            if not isinstance(value, tuple):
                self.history[model_name][metric_name].append((step, value))
        
        # REMOVED: Don't add to steps here since we're already managing steps in log_metric
        # This prevents duplicate step entries 

    def update_inference_metrics(self, model_name: str, time_per_token: float, memory_usage: float) -> None:
        """Update inference metrics for a model.
        
        Args:
            model_name: Name of the model
            time_per_token: Average time per token
            memory_usage: Memory usage in MB
        """
        if model_name not in self.model_names:
            raise ValueError(f"Unknown model name: {model_name}")
        
        self.inference_metrics[model_name]["time_per_token"].append(time_per_token)
        self.inference_metrics[model_name]["memory_usage"].append(memory_usage)
    
    def compute(self) -> Dict[str, Dict[str, float]]:
        """Compute final metrics for all models.
        
        Returns:
            Dict mapping model names to their metrics
        """
        results = {}
        for name in self.model_names:
            # Get metrics from the metrics object
            model_metrics = self.metrics[name].get_all_averages()
            
            # Add inference metrics if available
            time_per_token = self.inference_metrics[name]["time_per_token"]
            memory_usage = self.inference_metrics[name]["memory_usage"]
            
            if time_per_token:
                model_metrics["time_per_token"] = np.mean(time_per_token)
            if memory_usage:
                model_metrics["memory_usage"] = np.mean(memory_usage)
            
            results[name] = model_metrics
        
        return results
    
    def get_history(self) -> Dict[str, Dict[str, List]]:
        """Get training history for all models.
        
        Returns:
            Dict mapping model names to their metric histories
        """
        return self.history
    
    def get_inference_metrics(self) -> Dict[str, Dict[str, List[float]]]:
        """Get inference metrics for all models.
        
        Returns:
            Dict mapping model names to their inference metrics
        """
        return self.inference_metrics

    def get_metric(self, model_name: str, metric_name: str):
        """Retrieve a specific metric for a given model.

        Args:
            model_name: Name of the model (e.g., 'vanilla', 'titans').
            metric_name: Name of the metric to retrieve (e.g., 'loss', 'perplexity').

        Returns:
            The value of the requested metric, or None if not found.
        """
        if model_name in self.metrics:
            # Retrieve the latest value of the metric from MetricsTracker
            return self.metrics[model_name].get_last(metric_name)
        else:
            print(f"Warning: Metric '{metric_name}' for model '{model_name}' not found.")
            return None

    def log_metric(self, model_name: str, metric_name: str, value: float):
        """Log a specific metric for a given model.

        Args:
            model_name: Name of the model (e.g., 'vanilla', 'titans').
            metric_name: Name of the metric to log (e.g., 'loss', 'perplexity').
            value: Value of the metric to log.
        """
        if model_name not in self.history:
            raise ValueError(f"Unknown model name: {model_name}")
        
        if metric_name == 'steps':
            # Only update steps list if this is a step entry - prevent duplicates
            if not self.history[model_name].get('steps') or value > self.history[model_name]['steps'][-1]:
                self.history[model_name]['steps'].append(value)
        else:
            # For non-step metrics, append normally
            self.history[model_name][metric_name].append(value)
