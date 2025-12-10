import numpy as np
import torch
import seaborn as sns
from typing import Optional, List, Dict, Any, Union, Tuple
import pandas as pd
from pathlib import Path
import os
import math
import logging
from tqdm import tqdm
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from src.models import NeuralMemoryState, TransformerState
from src.utils import get_logger, Logger

"""Visualization utilities for the Titans model."""

import matplotlib.pyplot as plt

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')

# Setup logger
logger = get_logger(__name__)

def plot_attention_patterns(
    attention_weights: torch.Tensor, 
    title: str = "Attention Patterns",
    save_path: Optional[str] = None, 
    show: bool = True,
    max_heads: int = 4,
    figsize: Tuple[int, int] = (12, 8)
):
    """Visualize attention patterns.
    
    Args:
        attention_weights: Attention weights [batch, heads, seq_len, seq_len]
        title: Plot title
        save_path: Path to save the figure (optional)
        show: Whether to display the plot
        max_heads: Maximum number of heads to plot
        figsize: Figure size
    """
    logger.debug(f"Plotting attention patterns with shape {attention_weights.shape}")
    
    # Convert to numpy if tensor
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    # Get dimensions
    batch_size, n_heads, seq_len, _ = attention_weights.shape
    
    # Only plot first batch element
    attn_weights = attention_weights[0]
    
    # Limit number of heads to plot
    n_heads_to_plot = min(n_heads, max_heads)
    logger.debug(f"Plotting {n_heads_to_plot} attention heads out of {n_heads}")
    
    # Create figure
    fig, axes = plt.subplots(1, n_heads_to_plot, figsize=figsize)
    if n_heads_to_plot == 1:
        axes = [axes]
    
    # Plot each head
    for i in range(n_heads_to_plot):
        im = axes[i].imshow(attn_weights[i], cmap='viridis', aspect='equal')
        axes[i].set_title(f"Head {i+1}")
        axes[i].set_xlabel("Key Position")
        axes[i].set_ylabel("Query Position")
        
        # Add colorbar
        divider = make_axes_locatable(axes[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
    
    # Set overall title
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        logger.debug(f"Saving attention plot to {save_path}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show or close
    if show:
        plt.show()
    else:
        plt.close()


def plot_memory_heatmap(
    memory_states: List[NeuralMemoryState],
    title: str = "Memory Activations",
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (10, 6)
):
    """Visualize memory activations as heatmaps.
    
    Args:
        memory_states: List of memory states from different layers
        title: Plot title
        save_path: Path to save the figure (optional)
        show: Whether to display the plot
        figsize: Figure size
    """
    if not memory_states:
        logger.debug("No memory states provided for heatmap, skipping")
        return
    
    logger.debug(f"Plotting memory heatmap for {len(memory_states)} layers")
    n_layers = len(memory_states)
    
    # Convert memories to numpy arrays
    memories = [state.memory.detach().cpu().numpy() for state in memory_states]
    
    # Log memory statistics for debugging
    for i, mem in enumerate(memories):
        logger.debug(f"Memory layer {i}: shape={mem.shape}, "
                     f"min={np.min(mem):.4f}, max={np.max(mem):.4f}, "
                     f"mean={np.mean(mem):.4f}, std={np.std(mem):.4f}")
    
    # Only plot first batch example if batched
    if len(memories[0].shape) > 2:
        memories = [mem[0] for mem in memories]
    
    # Create figure
    fig, axes = plt.subplots(n_layers, 1, figsize=figsize)
    if n_layers == 1:
        axes = [axes]
    
    # Plot each layer's memory
    for i, memory in enumerate(memories):
        im = axes[i].imshow(memory, cmap='plasma', aspect='auto')
        axes[i].set_title(f"Layer {i+1} Memory")
        axes[i].set_xlabel("Memory Dimension")
        
        # Add colorbar
        divider = make_axes_locatable(axes[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
    
    # Set overall title
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        logger.debug(f"Saving memory heatmap to {save_path}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show or close
    if show:
        plt.show()
    else:
        plt.close()


def plot_memory_updates(
    memory_states: List[NeuralMemoryState],
    title: str = "Memory Updates",
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (10, 6)
):
    """Visualize memory update magnitudes.
    
    Args:
        memory_states: List of memory states from different layers
        title: Plot title
        save_path: Path to save the figure (optional)
        show: Whether to display the plot
        figsize: Figure size
    """
    if not memory_states:
        return
    
    n_layers = len(memory_states)
    
    # Extract update norms for each layer
    update_norms = []
    for state in memory_states:
        if hasattr(state, 'updates') and state.updates is not None:
            # Compute L2 norm along the feature dimension
            norm = torch.norm(state.updates, dim=-1).detach().cpu().numpy()
            update_norms.append(norm)
        else:
            update_norms.append(np.zeros(1))
    
    # Only plot first batch example if batched
    if len(update_norms[0].shape) > 1:
        update_norms = [norm[0] for norm in update_norms]
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create bar plot for update magnitudes
    x = np.arange(n_layers)
    plt.bar(x, [norm.mean() for norm in update_norms], yerr=[norm.std() for norm in update_norms],
            capsize=5, color='skyblue', alpha=0.8)
    
    plt.xlabel('Layer')
    plt.ylabel('Average Update Magnitude')
    plt.title(title)
    plt.xticks(x, [f"Layer {i+1}" for i in range(n_layers)])
    plt.grid(axis='y', alpha=0.3)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show or close
    if show:
        plt.show()
    else:
        plt.close()


def plot_training_metrics(
    metrics: Dict[str, List[float]],
    steps: Optional[List[int]] = None,
    title: str = "Training Metrics",
    model_name: Optional[str] = None,
    save_path: Optional[str] = None, 
    show: bool = True,
    figsize: Tuple[int, int] = (12, 8),
    rolling_window: int = 50
):
    """Plot training metrics over time.
    
    Args:
        metrics: Dictionary of metric names to lists of values
        steps: List of step numbers (optional)
        title: Plot title
        model_name: Name of the model (will be added to title)
        save_path: Path to save the figure (optional)
        show: Whether to display the plot
        figsize: Figure size
        rolling_window: Window size for rolling average
    """
    # Guard against empty metrics dict to avoid StopIteration
    if not metrics or all(len(v) == 0 for v in metrics.values()):
        logger.warning("No training metrics to plot, skipping plot.")
        return
    
    # Create steps if not provided
    if steps is None:
        steps = list(range(len(next(iter(metrics.values())))))
    
    # Create figure
    num_metrics = len(metrics)
    fig, axes = plt.subplots(num_metrics, 1, figsize=figsize, sharex=True)
    if num_metrics == 1:
        axes = [axes]
    
    # Apply rolling average if specified
    smoothed_metrics = {}
    if rolling_window > 1:
        for key, values in metrics.items():
            smoothed = pd.Series(values).rolling(rolling_window, min_periods=1).mean().values
            smoothed_metrics[key] = smoothed
    else:
        smoothed_metrics = metrics
    
    # Plot each metric
    for i, (metric_name, values) in enumerate(smoothed_metrics.items()):
        axes[i].plot(steps, values, linewidth=2)
        axes[i].set_title(f"{metric_name.capitalize()}")
        axes[i].set_ylabel(metric_name)
        axes[i].grid(True, alpha=0.3)
    
    # Set x-label for the bottom plot
    axes[-1].set_xlabel("Steps")
    
    # Set overall title
    # Add model name to title if provided
    full_title = title
    if model_name:
        full_title = f"{title} - {model_name}"
    fig.suptitle(full_title, fontsize=16)
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show or close
    if show:
        plt.show()
    else:
        plt.close()


def plot_comparison_metrics(
    metrics_dict: Dict[str, Dict[str, List[float]]],
    metric_name: str,
    steps: Optional[List[int]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (10, 6),
    rolling_window: int = 10
):
    """Plot comparison of a metric across different models/runs.
    
    Args:
        metrics_dict: Dictionary of model names to their metrics dictionaries
        metric_name: Name of the metric to compare
        steps: List of step numbers (optional)
        title: Plot title
        save_path: Path to save the figure (optional)
        show: Whether to display the plot
        figsize: Figure size
        rolling_window: Window size for rolling average
    """
    plt.figure(figsize=figsize)
    
    # Default title if none provided
    if title is None:
        title = f"{metric_name.capitalize()} Comparison"
    
    # Plot for each model/run
    for model_name, metrics in metrics_dict.items():
        if metric_name not in metrics:
            continue
            
        values = metrics[metric_name]
        
        # Create steps if not provided
        if steps is None:
            steps = list(range(len(values)))
            
        # Apply rolling average if specified
        if rolling_window > 1:
            values = pd.Series(values).rolling(rolling_window, min_periods=1).mean().values
            
        plt.plot(steps, values, linewidth=2, label=model_name)
    
    plt.title(title)
    plt.xlabel("Steps")
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show or close
    if show:
        plt.show()
    else:
        plt.close()


def plot_text_generation(
    original_text: str,
    generated_text: str,
    tokenizer,
    title: str = "Text Generation Comparison",
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (12, 10),
    max_len: int = 100
):
    """Visualize and compare original and generated text.
    
    Args:
        original_text: Original text
        generated_text: Model-generated text
        tokenizer: Tokenizer for highlighting differences
        title: Plot title
        save_path: Path to save the figure (optional)
        show: Whether to display the plot
        figsize: Figure size
        max_len: Maximum text length to display
    """
    # Truncate texts if they're too long
    if len(original_text) > max_len:
        original_text = original_text[:max_len] + "..."
    if len(generated_text) > max_len:
        generated_text = generated_text[:max_len] + "..."
    
    # Tokenize both texts
    orig_tokens = tokenizer.tokenize(original_text)
    gen_tokens = tokenizer.tokenize(generated_text)
    
    # Find differences between token sequences
    orig_highlight = []
    gen_highlight = []
    
    # Simple diff algorithm (not optimal but works for visualization)
    for i in range(min(len(orig_tokens), len(gen_tokens))):
        if orig_tokens[i] != gen_tokens[i]:
            orig_highlight.append(i)
            gen_highlight.append(i)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Display original text
    ax1.axis('off')
    ax1.text(0, 0.5, original_text, wrap=True, fontsize=12)
    ax1.set_title("Original Text")
    
    # Display generated text
    ax2.axis('off')
    ax2.text(0, 0.5, generated_text, wrap=True, fontsize=12)
    ax2.set_title("Generated Text")
    
    # Set overall title
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show or close
    if show:
        plt.show()
    else:
        plt.close()


def visualize_model_state(
    state: TransformerState,
    title: str = "Model State Visualization",
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (15, 10)
):
    """Visualize the complete state of the transformer model.
    
    Args:
        state: TransformerState containing memory_states, kv_caches, etc.
        title: Plot title
        save_path: Path to save the figure (optional)
        show: Whether to display the plot
        figsize: Figure size
    """
    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 2, figure=fig)
    
    # Extract data from state
    memory_states = state.memory_states if hasattr(state, 'memory_states') else []
    kv_caches = state.kv_caches if hasattr(state, 'kv_caches') else []
    value_residuals = state.value_residuals if hasattr(state, 'value_residuals') else []
    
    # Plot memory states
    if memory_states:
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Extract and stack memory from all layers
        memories = [state.memory.detach().cpu().numpy() for state in memory_states if hasattr(state, 'memory')]
        if memories:
            # Only plot first batch example if batched
            if len(memories[0].shape) > 2:
                memories = [mem[0] for mem in memories]
            
            # Stack memories for all layers
            stacked_memory = np.vstack(memories)
            im = ax1.imshow(stacked_memory, cmap='plasma', aspect='auto')
            ax1.set_title("Memory States (all layers)")
            ax1.set_ylabel("Layer")
            ax1.set_xlabel("Memory Dimension")
            
            # Add colorbar
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
    
    # Plot KV cache information
    if kv_caches:
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Extract key norms from KV caches
        key_norms = []
        for k, v in kv_caches:
            if k is not None:
                norm = torch.norm(k, dim=-1).mean(-1).detach().cpu().numpy()
                key_norms.append(norm)
        
        if key_norms:
            # Only plot first batch example
            key_norms = [norm[0] if len(norm.shape) > 1 else norm for norm in key_norms]
            
            # Plot key norms for each layer
            for i, norm in enumerate(key_norms):
                ax2.plot(norm, label=f"Layer {i+1}")
                
            ax2.set_title("Key Norms in KV Cache")
            ax2.set_xlabel("Position")
            ax2.set_ylabel("Norm")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
    
    # Plot value residuals
    if value_residuals:
        ax3 = fig.add_subplot(gs[1, 0])
        
        # Extract value residual norms
        residual_norms = []
        for vr in value_residuals:
            if vr is not None:
                norm = torch.norm(vr, dim=-1).mean(-1).detach().cpu().numpy()
                residual_norms.append(norm)
        
        if residual_norms:
            # Only plot first batch example
            residual_norms = [norm[0] if len(norm.shape) > 1 else norm for norm in residual_norms]
            
            # Plot value residual norms for each layer
            for i, norm in enumerate(residual_norms):
                ax3.plot(norm, label=f"Layer {i+1}")
                
            ax3.set_title("Value Residual Norms")
            ax3.set_xlabel("Position")
            ax3.set_ylabel("Norm")
            ax3.legend()
            ax3.grid(True, alpha=0.3)
    
    # Plot memory update magnitudes
    if memory_states:
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Extract update norms for each layer
        update_norms = []
        for state in memory_states:
            if hasattr(state, 'updates') and state.updates is not None:
                # Compute L2 norm along the feature dimension
                norm = torch.norm(state.updates, dim=-1).detach().cpu().numpy()
                update_norms.append(norm.mean())
            else:
                update_norms.append(0)
        
        if update_norms:
            # Create bar plot for update magnitudes
            x = np.arange(len(update_norms))
            ax4.bar(x, update_norms, color='skyblue', alpha=0.8)
            
            ax4.set_xlabel('Layer')
            ax4.set_ylabel('Average Update Magnitude')
            ax4.set_title("Memory Update Magnitudes")
            ax4.set_xticks(x)
            ax4.set_xticklabels([f"{i+1}" for i in range(len(update_norms))])
            ax4.grid(axis='y', alpha=0.3)
    
    # Add memory efficiency metrics if available
    if hasattr(state, 'memory_efficiency_metrics'):
        ax5 = fig.add_subplot(gs[1, 1])
        metrics = state.memory_efficiency_metrics
        
        if metrics and 'relevance_scores' in metrics:
            relevance = metrics['relevance_scores']
            ax5.plot(relevance, marker='o', color='green', label='Memory Relevance')
            ax5.set_xlabel('Memory Index')
            ax5.set_ylabel('Relevance Score')
            ax5.set_title('Memory Usage Efficiency')
            ax5.grid(True, alpha=0.3)
            ax5.legend()
    
    # Plot memory momentum
    if memory_states:
        ax5 = fig.add_subplot(gs[2, :])
        
        # Extract momentum norms
        momentum_norms = []
        for state in memory_states:
            if hasattr(state, 'momentum') and state.momentum is not None:
                # Compute L2 norm along the feature dimension
                norm = torch.norm(state.momentum, dim=-1).detach().cpu().numpy()
                momentum_norms.append(norm)
            else:
                momentum_norms.append(np.zeros(1))
        
        if momentum_norms:
            # Only plot first batch example if batched
            momentum_norms = [norm[0] if len(norm.shape) > 1 else norm for norm in momentum_norms]
            
            # Plot momentum norms for each layer
            for i, norm in enumerate(momentum_norms):
                if len(norm.shape) > 0:  # If not a scalar
                    ax5.plot(norm, label=f"Layer {i+1}")
                    
            ax5.set_title("Memory Momentum Norms")
            ax5.set_xlabel("Memory Dimension")
            ax5.set_ylabel("Norm")
            ax5.legend()
            ax5.grid(True, alpha=0.3)
    
    # Set overall title
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show or close
    if show:
        plt.show()
    else:
        plt.close()


def compare_memory_states(
    states: Dict[str, TransformerState],
    title: str = "Memory State Comparison",
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (18, 14)
):
    """Compare memory states between different models.
    
    Args:
        states: Dictionary mapping model names to their states
        title: Plot title
        save_path: Path to save the figure
        show: Whether to display the plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    plt.suptitle(title, fontsize=16)
    
    # Set up the plotting grid
    n_models = len(states)
    gs = plt.GridSpec(3, n_models)
    
    model_names = list(states.keys())
    
    # Extract memory update norms for each model
    for i, (name, state) in enumerate(states.items()):
        memory_states = state.memory_states if hasattr(state, 'memory_states') else []
        
        # Plot memory update magnitudes
        if memory_states:
            ax1 = plt.subplot(gs[0, i])
            update_norms = []
            
            for mem_state in memory_states:
                if mem_state is not None and hasattr(mem_state, 'updates') and mem_state.updates is not None:
                    # Compute L2 norm along the feature dimension
                    norm = torch.norm(mem_state.updates, dim=-1).detach().cpu().numpy()
                    update_norms.append(norm.mean())
                else:
                    update_norms.append(0)
            
            if update_norms:
                x = np.arange(len(update_norms))
                ax1.bar(x, update_norms, alpha=0.8)
                ax1.set_title(f"{name} - Memory Update Magnitudes")
                ax1.set_xlabel('Layer')
                ax1.set_ylabel('Magnitude')
                ax1.set_xticks(x)
                ax1.set_xticklabels([f"{j+1}" for j in range(len(update_norms))])
                ax1.grid(True, alpha=0.3)
        
        # Plot KV cache norms
        kv_caches = state.kv_caches if hasattr(state, 'kv_caches') else []
        if kv_caches:
            ax2 = plt.subplot(gs[1, i])
            key_norms = []
            
            for k, v in kv_caches:
                if k is not None:
                    norm = torch.norm(k, dim=-1).mean(-1).detach().cpu().numpy()
                    if len(norm.shape) > 0:
                        key_norms.append(norm[0])  # Take first batch item
                    else:
                        key_norms.append(norm)
            
            if key_norms:
                for j, norm in enumerate(key_norms):
                    ax2.plot(norm, label=f"Layer {j+1}")
                
                ax2.set_title(f"{name} - Key Norms in KV Cache")
                ax2.set_xlabel('Position')
                ax2.set_ylabel('Norm')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
        
        # Plot value residuals
        value_residuals = state.value_residuals if hasattr(state, 'value_residuals') else []
        if value_residuals:
            ax3 = plt.subplot(gs[2, i])
            residual_norms = []
            
            for vr in value_residuals:
                if vr is not None:
                    norm = torch.norm(vr, dim=-1).mean(-1).detach().cpu().numpy()
                    residual_norms.append(norm)
            
            if residual_norms:
                for j, norm in enumerate(residual_norms):
                    if len(norm.shape) > 0:
                        ax3.plot(norm, label=f"Layer {j+1}")
                    else:
                        ax3.plot([norm], label=f"Layer {j+1}")
                
                ax3.set_title(f"{name} - Value Residual Norms")
                ax3.set_xlabel('Position')
                ax3.set_ylabel('Norm')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    if show:
        plt.show()


def plot_memory_efficiency(
    metrics: Dict[str, Any],
    title: str = 'Memory Efficiency Metrics',
    figsize: Tuple[int, int] = (18, 10),
    save_path: Optional[str] = None
):
    """Plot memory efficiency metrics over time.
    
    Args:
        metrics: Dictionary of memory efficiency metrics
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    # Extract steps and memory metrics
    steps = metrics.get('steps', [])
    memory_usage = metrics.get('memory_usage', [])
    update_magnitude = metrics.get('update_magnitude', [])
    memory_utilization = metrics.get('memory_utilization', [])
    retrieval_quality = metrics.get('retrieval_quality', [])
    
    # Process metrics for plotting
    metric_datasets = {
        'Memory Usage (MB)': memory_usage,
        'Update Magnitude': update_magnitude,
        'Memory Utilization (%)': memory_utilization,
        'Retrieval Quality': retrieval_quality
    }
    
    for i, (metric_name, metric_data) in enumerate(metric_datasets.items()):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        # Handle empty data
        if not metric_data or not steps:
            ax.text(0.5, 0.5, 'No data available', 
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax.transAxes)
            continue
        
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
            print(f"Warning: Error processing memory metric data: {e}")
            values = metric_data  # Fallback to using the data as is
            
        # Ensure we have data to plot
        if len(steps) > 1 and len(values) > 1:
            # Make sure values length matches steps length
            if len(values) == len(steps):
                # Calculate rolling average for smoother visualization
                rolling_window = len(steps) // 5 if len(steps) > 5 else 1
                if rolling_window > 1:
                    rolling_avg = pd.Series(values).rolling(rolling_window, min_periods=1).mean()
                    ax.plot(steps, rolling_avg, linestyle='-', color='blue', alpha=0.7, label='Rolling Avg')
                
                # Plot the actual values
                # ax.plot(steps, values, marker='o', markersize=3, alpha=0.5, label='Raw Data')
            else:
                print(f"Warning: Mismatch in steps and values for {metric_name}")
                print(f"Steps length: {len(steps)}, Values length: {len(values)}")
                # Use the shorter length to avoid errors
                min_len = min(len(steps), len(values))
                ax.plot(steps[:min_len], values[:min_len], marker='o', markersize=3, alpha=0.7)
        else:
            print(f"Warning: Not enough data points for {metric_name}")
            
        ax.set_title(metric_name)
        ax.set_xlabel('Steps')
        ax.grid(True, alpha=0.3)
        
        # Add legend if we have multiple lines
        if len(steps) > 20 and len(values) > 20:
            ax.legend(loc='best')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()


def save_animation_frames(
    frames_data: List[Any],
    plot_func: callable,
    output_dir: str,
    filename_prefix: str = "frame",
    **plot_kwargs
):
    """Save a series of frames for creating animations.
    
    Args:
        frames_data: List of data for each frame
        plot_func: Function to plot each frame
        output_dir: Directory to save frames
        filename_prefix: Prefix for frame filenames
        **plot_kwargs: Additional arguments for the plot function
    """
    logger.debug(f"Saving {len(frames_data)} animation frames to {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each frame
    for i, data in enumerate(tqdm(frames_data, desc="Saving frames")):
        frame_path = os.path.join(output_dir, f"{filename_prefix}_{i:04d}.png")
        try:
            plot_func(data, save_path=frame_path, show=False, **plot_kwargs)
        except Exception as e:
            logger.error(f"Error saving frame {i}: {e}")

    logger.info(f"Saved {len(frames_data)} frames to {output_dir}")
    logger.info("You can create an animation using: ffmpeg -r 10 -i "
              f"{output_dir}/{filename_prefix}_%04d.png -c:v libx264 -pix_fmt yuv420p animation.mp4")