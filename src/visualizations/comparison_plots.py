"""Visualization utilities for model comparison."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any

import torch

from src.models.transformer_arch.titans_transformer import TransformerState
from src.models.transformer_arch.vanilla_transformer import VanillaTransformer, VanillaLM
from src.models.transformer_arch.titans_transformer import TitansTransformer, TitansLM


def plot_training_comparison(
    histories: Dict[str, Dict[str, List]],
    metrics: List[str] = ['loss', 'perplexity', 'accuracy'],
    title: str = 'Model Training Comparison',
    figsize: Tuple[int, int] = (18, 12),
    save_path: Optional[str] = None,
    log_scale: bool = True
):
    """Plot training metrics comparison between models.
    
    Args:
        histories: Dictionary of training histories for each model
        metrics: List of metrics to plot
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure
        log_scale: Whether to use log scale for y-axis (for loss and perplexity)
    """
    model_names = list(histories.keys())
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        for model in model_names:
            steps = histories[model]['steps']
            metric_data = histories[model][metric]
            
            # More robust handling of different data types
            try:
                # Try to treat the data as a list of tuples
                values = []
                for item in metric_data:
                    # Check if this item is a tuple with at least 2 elements
                    if isinstance(item, tuple) and len(item) >= 2:
                        step, value = item
                        if step in steps:
                            values.append(value)
                    else:
                        # If not a tuple, just use as is
                        values.append(item)
            except Exception as e:
                print(f"Warning: Error processing metric data: {e}")
                values = metric_data  # Fallback to using the data as is

            # Ensure we have data to plot
            if len(steps) > 1 and len(values) > 1:
                # Make sure values length matches steps length
                if len(values) == len(steps):
                    # Calculate rolling average for the metric values
                    rolling_window = len(steps) // 50 if len(steps) > 10 else 1
                    rolling_avg = pd.Series(values).rolling(max(1, rolling_window), min_periods=1).mean()

                    # Plot only the rolling average, not the raw data
                    ax.plot(steps, rolling_avg, label=model, linestyle='-', alpha=0.8)
                else:
                    print(f"Warning: Mismatch in steps and values for {model} - {metric}")
                    print(f"Steps length: {len(steps)}, Values length: {len(values)}")
                    # Use the shorter length to avoid errors
                    min_len = min(len(steps), len(values))
                    # Ensure rolling window is at least 1
                    rolling_window = max(1, min(min_len // 10, 5)) if min_len > 5 else 1
                    rolling_values = pd.Series(values[:min_len]).rolling(rolling_window, min_periods=1).mean()
                    ax.plot(steps[:min_len], rolling_values, label=model, alpha=0.8)
            else:
                print(f"Warning: Not enough data points for {model} - {metric}")
        
        ax.set_title(f'{metric.capitalize()}')
        ax.set_xlabel('Steps')
        ax.set_ylabel(metric.capitalize())
        ax.grid(True, alpha=0.3)
        
        if log_scale and metric in ['loss', 'perplexity']:
            ax.set_yscale('log')
        
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()


def plot_inference_metrics(
    inference_metrics: Dict[str, Dict[str, List]],
    title: str = 'Model Inference Comparison',
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
):
    """Plot inference metrics comparison between models.
    
    Args:
        inference_metrics: Dictionary of inference metrics for each model
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure
    """
    model_names = list(inference_metrics.keys())
    
    # Convert to pandas DataFrame for easier plotting
    data = []
    for model in model_names:
        times = inference_metrics[model]['time_per_token']
        memories = inference_metrics[model]['memory_usage']
        
        if len(times) > 0:
            data.append({
                'model': model,
                'time_per_token': np.mean(times),
                'std_time': np.std(times),
                'memory_usage': np.mean(memories),
                'std_memory': np.std(memories)
            })
    
    if not data:
        print("No inference data available for plotting.")
        return
    
    df = pd.DataFrame(data)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    # Plot time per token
    sns.barplot(x='model', y='time_per_token', data=df, ax=ax1)
    ax1.set_title('Average Time per Token')
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Time (s)')
    
    # Add error bars
    for i, row in df.iterrows():
        ax1.errorbar(i, row['time_per_token'], yerr=row['std_time'], fmt='o', color='black')
    
    # Plot memory usage
    sns.barplot(x='model', y='memory_usage', data=df, ax=ax2)
    ax2.set_title('Average Memory Usage')
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Memory (MB)')
    
    # Add error bars
    for i, row in df.iterrows():
        ax2.errorbar(i, row['memory_usage'], yerr=row['std_memory'], fmt='o', color='black')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()


def plot_comparative_metrics(
    results: Dict[str, Dict[str, Any]],
    baseline_model: str = 'vanilla',
    comparison_model: str = 'titans',
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
):
    """Plot comparative metrics between baseline and improved model.
    
    Args:
        results: Dictionary of model results
        baseline_model: Name of baseline model
        comparison_model: Name of model being compared
        figsize: Figure size
        save_path: Path to save the figure
    """
    print(results)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Comparative Analysis: {baseline_model} vs {comparison_model}', fontsize=16)
    
    # Extract comparative metrics
    comp_metrics = results.get('comparative', {})
    
    # Plot perplexity comparison
    perplexities = [results[baseline_model].get('perplexity', 0), 
                   results[comparison_model].get('perplexity', 0)]
    
    axes[0, 0].bar([baseline_model, comparison_model], perplexities, color=['lightgray', 'lightblue'])
    axes[0, 0].set_title('Perplexity (lower is better)')
    if perplexities[0] > 0 and perplexities[1] > 0:
        percent_change = (perplexities[1] - perplexities[0]) / perplexities[0] * 100
        axes[0, 0].text(1, perplexities[1], f'{percent_change:.1f}%', 
                       ha='center', va='bottom', fontweight='bold')
    
    # Plot accuracy comparison
    accuracies = [results[baseline_model].get('accuracy', 0) * 100, 
                 results[comparison_model].get('accuracy', 0) * 100]  # Convert to percentage
    
    axes[0, 1].bar([baseline_model, comparison_model], accuracies, color=['lightgray', 'lightblue'])
    axes[0, 1].set_title('Accuracy (%)')
    if accuracies[0] > 0 and accuracies[1] > 0:
        percent_change = (accuracies[1] - accuracies[0])  # Already percentage points
        axes[0, 1].text(1, accuracies[1], f'+{percent_change:.1f}pts', 
                       ha='center', va='bottom', fontweight='bold')
    
    # Plot inference time comparison
    if ('time_per_token' in results[baseline_model] and 
        'time_per_token' in results[comparison_model]):
        
        times = [results[baseline_model]['time_per_token'], 
                results[comparison_model]['time_per_token']]
        
        axes[1, 0].bar([baseline_model, comparison_model], times, color=['lightgray', 'lightblue'])
        axes[1, 0].set_title('Inference Time per Token (s)')
        if times[0] > 0 and times[1] > 0:
            percent_change = (times[1] - times[0]) / times[0] * 100
            axes[1, 0].text(1, times[1], f'{percent_change:.1f}%', 
                           ha='center', va='bottom', fontweight='bold')
    
    # Plot memory usage comparison
    if ('memory_usage' in results[baseline_model] and 
        'memory_usage' in results[comparison_model]):
        
        memories = [results[baseline_model]['memory_usage'], 
                   results[comparison_model]['memory_usage']]
        
        axes[1, 1].bar([baseline_model, comparison_model], memories, color=['lightgray', 'lightblue'])
        axes[1, 1].set_title('Memory Usage (MB)')
        if memories[0] > 0 and memories[1] > 0:
            percent_change = (memories[1] - memories[0]) / memories[0] * 100
            axes[1, 1].text(1, memories[1], f'{percent_change:.1f}%', 
                           ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()
