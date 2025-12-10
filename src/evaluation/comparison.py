"""Evaluation utilities for model comparison."""

import time
import torch
import numpy as np
import gc
from typing import Dict, List, Tuple, Optional, Any, Union

from src.models.transformer_arch.titans_transformer import TitansLM
from src.models.transformer_arch.vanilla_transformer import VanillaLM
from src.training.comparison_metrics import ComparativeMetrics


def measure_inference_time(
    model: Union[TitansLM, VanillaLM],
    input_ids: torch.Tensor,
    max_length: int = 20,
    num_trials: int = 5,
    warmup_trials: int = 2
) -> Tuple[float, float]:
    """Measure inference time per token.
    
    Args:
        model: Language model
        input_ids: Input tensor
        max_length: Maximum generation length
        num_trials: Number of trials to average
        warmup_trials: Number of warmup trials
        
    Returns:
        avg_time_per_token: Average time per token in seconds
        memory_usage: Peak memory usage in MB
    """
    device = input_ids.device
    
    # Warmup runs
    for _ in range(warmup_trials):
        with torch.no_grad():
            model.generate(input_ids, max_length=5)
    
    # Clear memory
    torch.cuda.empty_cache() if device.type == 'cuda' else gc.collect()
    torch.cuda.reset_peak_memory_stats(device) if device.type == 'cuda' else None
    
    # Measure time
    times = []
    for _ in range(num_trials):
        start_time = time.time()
        with torch.no_grad():
            model.generate(input_ids, max_length=max_length)
        end_time = time.time()
        times.append(end_time - start_time)
    
    # Measure memory usage
    memory_usage = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == 'cuda' else 0
    
    # Calculate average time per token
    avg_time = np.mean(times)
    avg_time_per_token = avg_time / max_length
    
    return avg_time_per_token, memory_usage


def compare_models_inference(
    models: Dict[str, Union[TitansLM, VanillaLM]], 
    input_ids: torch.Tensor,
    max_length: int = 20,
    num_trials: int = 5,
    warmup_trials: int = 2
) -> Dict[str, Dict[str, float]]:
    """Compare inference metrics across models.
    
    Args:
        models: Dictionary of models
        input_ids: Input tensor
        max_length: Maximum generation length
        num_trials: Number of trials to average
        warmup_trials: Number of warmup trials
        
    Returns:
        results: Dictionary of inference results
    """
    results = {}
    
    for name, model in models.items():
        print(f"Measuring inference metrics for {name}...")
        time_per_token, memory_usage = measure_inference_time(
            model, input_ids, max_length, num_trials, warmup_trials
        )
        
        results[name] = {
            'time_per_token': time_per_token,
            'memory_usage': memory_usage
        }
        
        print(f"  Average time per token: {time_per_token:.6f} seconds")
        print(f"  Memory usage: {memory_usage:.2f} MB")
    
    return results


def evaluate_models(
    models: Dict[str, Union[TitansLM, VanillaLM]],
    dataloader: torch.utils.data.DataLoader,
    metrics: ComparativeMetrics,
    max_batches: Optional[int] = None,
    device: str = 'cuda',
    label_smoothing: float = 0.1  # Added label_smoothing parameter
) -> Dict[str, Dict[str, Any]]:
    """Evaluate all models on the same dataset.
    
    Args:
        models: Dictionary of models
        dataloader: DataLoader for evaluation
        metrics: ComparativeMetrics instance
        max_batches: Maximum number of batches to evaluate
        device: Device to use for evaluation
        label_smoothing: Label smoothing factor (default: 0.1)
        
    Returns:
        results: Dictionary of evaluation results
    """
    for name, model in models.items():
        print(f"Evaluating {name}...")
        model.eval()
        model.to(device)
        
        batch_count = 0
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                
                # Proper label shifting and masking
                if 'labels' in batch:
                    labels = batch['labels'].to(device)
                else:
                    # Create shifted targets for next token prediction
                    labels = input_ids.clone()[:, 1:]
                    input_ids = input_ids[:, :-1]
                    
                    # Mask padding tokens with -100
                    pad_token_id = getattr(model, 'pad_token_id', 0)
                    labels[labels == pad_token_id] = -100
                
                # Forward pass: handle models returning (logits, state)
                out = model(input_ids)
                logits = out[0] if isinstance(out, tuple) else out

                # Fix tuple handling: Ensure logits is a tensor, not a tuple
                if isinstance(logits, tuple):
                    logits = logits[0]  # Extract logits from (logits, state) tuple
                
                # Calculate loss with label smoothing
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                    label_smoothing=label_smoothing  # Apply label smoothing
                )
                
                # Update metrics
                metrics.update(name, loss, logits, labels)
                
                batch_count += 1
                if max_batches and batch_count >= max_batches:
                    break
    
    # Compute metrics
    results = metrics.compute()
    return results
