"""Visualization utilities for the Titans model.

This module provides functions for visualizing various aspects of Transformers
such as attention patterns, memory states, training metrics, and text generation.
"""

from src.visualizations.plots import (
    plot_attention_patterns,
    plot_memory_heatmap,
    plot_memory_updates,
    plot_training_metrics,
    plot_comparison_metrics,
    plot_text_generation,
    visualize_model_state,
    save_animation_frames
)

from src.visualizations.attention_viz import (
    AttentionVisualizer,
    visualize_generation_attention
)

__all__ = [
    'plot_attention_patterns',
    'plot_memory_heatmap',
    'plot_memory_updates',
    'plot_training_metrics',
    'plot_comparison_metrics',
    'plot_text_generation',
    'visualize_model_state',
    'save_animation_frames',
    'AttentionVisualizer',
    'visualize_generation_attention'
]