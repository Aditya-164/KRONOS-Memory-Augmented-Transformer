import torch
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import math
import logging
from collections import defaultdict
from src.utils import get_logger, Logger

"""Evaluation metrics for the Titans model."""

import torch.nn.functional as F

# Setup logger
logger = get_logger(__name__)

class LanguageModelMetrics:
    """Metrics for evaluating language models."""

    @staticmethod
    def compute_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int = -100,
        label_smoothing: float = 0.0  # Added label smoothing parameter
    ) -> torch.Tensor:
        """Compute cross-entropy loss."""
        # Shift logits/labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()
        # Flatten for cross-entropy
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        # Trim to same length if mismatch
        if flat_logits.size(0) != flat_labels.size(0):
            logger.warning(f"Trimming compute_loss: logits {flat_logits.size(0)} vs labels {flat_labels.size(0)}")
            min_len = min(flat_logits.size(0), flat_labels.size(0))
            flat_logits = flat_logits[:min_len]
            flat_labels = flat_labels[:min_len]
        loss = F.cross_entropy(
            flat_logits,
            flat_labels,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing
        )
        return loss
    
    @staticmethod
    def compute_perplexity(loss: torch.Tensor) -> torch.Tensor:
        """Compute perplexity from loss."""
        perplexity = torch.exp(loss)
        logger.debug(f"Computed perplexity: {perplexity.item():.4f}")
        return perplexity
    
    @staticmethod
    def compute_accuracy(
        logits: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int = -100
    ) -> Tuple[torch.Tensor, int]:
        """Compute prediction accuracy."""
        # Shift logits/labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()
        # Flatten for accuracy
        flat_preds = shift_logits.view(-1, shift_logits.size(-1)).argmax(dim=-1)
        flat_labels = shift_labels.view(-1)
        # Trim to same length if mismatch
        if flat_preds.size(0) != flat_labels.size(0):
            logger.warning(f"Trimming compute_accuracy: preds {flat_preds.size(0)} vs labels {flat_labels.size(0)}")
            min_len = min(flat_preds.size(0), flat_labels.size(0))
            flat_preds = flat_preds[:min_len]
            flat_labels = flat_labels[:min_len]
        mask = (flat_labels != ignore_index)
        correct = (flat_preds == flat_labels) & mask
        total_valid = mask.sum().item()
        accuracy = correct.sum().float() / total_valid if total_valid > 0 else torch.tensor(0.0, device=logits.device)
        return accuracy, total_valid
    def update(self,logits: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int = -100):
        """Update metrics with new logits and targets."""
        return self.compute_accuracy(logits, targets, ignore_index), self.compute_perplexity(logits, targets, ignore_index), self.compute_loss(logits, targets, ignore_index)

class MemoryMetrics:
    """Metrics for evaluating memory usage and efficiency."""
    
    @staticmethod
    def compute_memory_usage(
        memory_states: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Compute memory usage statistics."""
        metrics = {}
        
        if not memory_states:
            logger.debug("No memory states provided for metrics")
            return metrics
        
        logger.debug(f"Computing memory metrics for {len(memory_states)} memory states")
        
        # Calculate memory norms
        memory_norms = [state.memory.norm(dim=-1).mean() for state in memory_states if hasattr(state, 'memory')]
        
        if memory_norms:
            metrics['memory_norm_mean'] = torch.stack(memory_norms).mean()
            metrics['memory_norm_std'] = torch.stack(memory_norms).std()
            metrics['memory_norm_max'] = torch.stack(memory_norms).max()
            
            logger.debug(f"Memory norm stats: mean={metrics['memory_norm_mean'].item():.4f}, "
                         f"std={metrics['memory_norm_std'].item():.4f}, "
                         f"max={metrics['memory_norm_max'].item():.4f}")
        
        # Calculate update magnitudes
        update_norms = [state.updates.norm(dim=-1).mean() for state in memory_states if hasattr(state, 'updates')]
        
        if update_norms:
            metrics['update_norm_mean'] = torch.stack(update_norms).mean()
            metrics['update_norm_std'] = torch.stack(update_norms).std()
            
            logger.debug(f"Update norm stats: mean={metrics['update_norm_mean'].item():.4f}, "
                         f"std={metrics['update_norm_std'].item():.4f}")
        
        return metrics


class GenerationMetrics:
    """Metrics for evaluating text generation quality."""
    
    @staticmethod
    def compute_sequence_repetition(
        sequences: torch.Tensor,
        ngram_sizes: List[int] = [2, 3, 4]
    ) -> Dict[str, float]:
        """Compute sequence repetition metrics."""
        metrics = {}
        batch_size, seq_len = sequences.shape
        sequences_np = sequences.detach().cpu().numpy()
        
        logger.debug(f"Computing sequence repetition for {batch_size} sequences of length {seq_len}")
        
        for n in ngram_sizes:
            if seq_len <= n:
                continue
                
            # Count repeated n-grams
            total_repetitions = 0
            total_ngrams = 0
            
            for b in range(batch_size):
                ngrams = set()
                repeats = 0
                
                for i in range(seq_len - n + 1):
                    ngram = tuple(sequences_np[b, i:i+n])
                    if ngram in ngrams:
                        repeats += 1
                    else:
                        ngrams.add(ngram)
                
                total_repetitions += repeats
                total_ngrams += (seq_len - n + 1)
            
            if total_ngrams > 0:
                rep_rate = total_repetitions / total_ngrams
                metrics[f'repetition_{n}gram'] = rep_rate
                logger.debug(f"Repetition rate for {n}-grams: {rep_rate:.4f}")
        
        return metrics
    
    @staticmethod
    def compute_token_entropy(
        logits: torch.Tensor,
        dim: int = -1
    ) -> torch.Tensor:
        """Compute entropy of token distributions."""
        probs = F.softmax(logits, dim=dim)
        log_probs = F.log_softmax(logits, dim=dim)
        entropy = -(probs * log_probs).sum(dim=dim)
        
        # Log average entropy for debugging
        avg_entropy = entropy.mean().item()
        logger.debug(f"Average token entropy: {avg_entropy:.4f}")
        
        return entropy


class MetricsTracker:
    """Track and aggregate metrics during training/evaluation."""
    
    def __init__(self):
        """Initialize an empty metrics tracker."""
        self.logger = get_logger(f"{__name__}.MetricsTracker")
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.metrics = defaultdict(list)
        self.counts = defaultdict(int)
        self.sums = defaultdict(float)
        self.logger.debug("Metrics tracker reset")
    
    def update(self, metrics_dict: Dict[str, Union[float, torch.Tensor]], batch_size: int = 1):
        """Update metrics with new values."""
        # self.logger.debug(f"Updating metrics with batch size {batch_size}")
        
        for name, value in metrics_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().float()
                if value.numel() == 1:
                    value = value.item()
                else:
                    value = value.cpu().numpy()
            
            self.metrics[name].append(value)
            self.sums[name] += value * batch_size
            self.counts[name] += batch_size
            
            # self.logger.debug(f"Updated metric '{name}': value={value}, "
            #                  f"running_avg={self.sums[name]/self.counts[name]:.4f}")
    
    def get_average(self, name: str) -> float:
        """Get average value for a metric."""
        if name not in self.counts or self.counts[name] == 0:
            return 0.0
        return self.sums[name] / self.counts[name]
    
    def get_all_averages(self) -> Dict[str, float]:
        """Get averages for all metrics."""
        averages = {name: self.get_average(name) for name in self.metrics}
        self.logger.debug(f"Current metric averages: {averages}")
        return averages
    
    def get_last(self, name: str) -> float:
        """Get the most recent value for a metric."""
        if name not in self.metrics or not self.metrics[name]:
            return 0.0
        return self.metrics[name][-1]

    def get_last(self, metric_name: str):
        """Retrieve the latest value of a specific metric

        Args:
            metric_name: Name of the metric to retrieve.

        Returns:
            The latest value of the metric, or None if not found.
        """
        if metric_name in self.metrics:
            return self.metrics[metric_name][-1]  # Return the last value
        else:
            print(f"Warning: Metric '{metric_name}' not found in MetricsTracker.")
            return None

    def get_all_last(self) -> Dict[str, float]:
        """Get the most recent values for all metrics."""
        last_values = {name: self.get_last(name) for name in self.metrics}
        self.logger.debug(f"Current last metric values: {last_values}")
        return last_values

def compute_metrics_from_batch(
    logits: torch.Tensor,
    targets: torch.Tensor,
    memory_states: Optional[List] = None,
    prefix: str = "",
    label_smoothing: float = 0.0  # Add label smoothing parameter
) -> Dict[str, float]:
    """Compute multiple metrics from a batch."""
    logger.debug(f"Computing metrics from batch with logits shape={logits.shape}, targets shape={targets.shape}")
    metrics = {}
    
    # Compute loss with label smoothing
    loss = LanguageModelMetrics.compute_loss(
        logits, 
        targets, 
        label_smoothing=label_smoothing  # Pass through label smoothing
    )
    metrics[f'{prefix}loss'] = loss.item()
    
    # Compute perplexity
    perplexity = LanguageModelMetrics.compute_perplexity(loss)
    metrics[f'{prefix}perplexity'] = perplexity.item()
    
    # Compute accuracy
    accuracy, total_tokens = LanguageModelMetrics.compute_accuracy(logits, targets)
    metrics[f'{prefix}accuracy'] = accuracy.item()
    metrics[f'{prefix}total_tokens'] = total_tokens
    
    # Compute entropy
    token_entropy = GenerationMetrics.compute_token_entropy(logits).mean()
    metrics[f'{prefix}token_entropy'] = token_entropy.item()
    
    # Compute memory metrics if available
    if memory_states:
        logger.debug(f"Computing memory metrics from {len(memory_states)} states")
        memory_metrics = MemoryMetrics.compute_memory_usage(memory_states)
        for name, value in memory_metrics.items():
            metrics[f'{prefix}{name}'] = value.item() if isinstance(value, torch.Tensor) else value
    
    return metrics