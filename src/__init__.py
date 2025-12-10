"""Titans: A neural network implementation focusing on memory-augmented transformers with segmented attention mechanisms."""

__version__ = "0.1.0"

# Import model components
from src.models import (
    # Core model classes
    TitansTransformer,
    TitansLM,
    TransformerBlock,
    TransformerState,
    
    # Attention mechanism
    SegmentedAttention,
    FeedForward,
    ContinuousAxialPositionalEmbedding,
    
    # Memory components
    NeuralMemory,
    NeuralMemoryState,
    
    # Utility functions
    exists,
    default,
    create_sliding_window_mask,
    create_block_diagonal_mask,
    init_weights,
    get_slopes,
    chunk_sequence,
    merge_chunks,
    top_k_top_p_filtering,
    create_causal_mask
)

# Import data utilities
from src.data import (
    TextDataset,
    WikiTextDataset,
    CollatorForLanguageModeling,
    get_dataloader,
    prepare_datasets,
)

# Import training utilities
from src.training import (
    Trainer,
    DistributedTrainer,
    LanguageModelMetrics,
    MemoryMetrics,
    GenerationMetrics,
    MetricsTracker,
    compute_metrics_from_batch,
)

# Import visualization utilities
from src.visualizations import (
    plot_attention_patterns,
    plot_memory_heatmap,
    plot_memory_updates,
    plot_training_metrics,
    plot_comparison_metrics,
    plot_text_generation,
    visualize_model_state,
    save_animation_frames,
)

# Import configuration
from src.config import (
    ModelConfig,
    TrainingConfig,
    DefaultConfig,
)

# Import logging utilities
from src.utils import (
    get_logger,
    Logger,
    log_step_info,
    log_model_info,
    log_memory_usage,
    debug_tensor
)

# Define what's available via "from titans import *"
__all__ = [
    # Version
    "__version__",
    
    # Core model classes
    "TitansTransformer",
    "TitansLM",
    "TransformerBlock",
    "TransformerState",
    
    # Attention components
    "SegmentedAttention",
    "FeedForward",
    "ContinuousAxialPositionalEmbedding",
    
    # Memory components
    "NeuralMemory",
    "NeuralMemoryState",
    
    # Utility functions
    "exists",
    "default",
    "create_sliding_window_mask",
    "create_block_diagonal_mask",
    "init_weights",
    "get_slopes",
    "chunk_sequence",
    "merge_chunks",
    "top_k_top_p_filtering",
    "create_causal_mask",
    
    # Data utilities
    "TextDataset",
    "WikiTextDataset",
    "CollatorForLanguageModeling",
    "get_dataloader",
    "prepare_datasets",
    
    # Training utilities
    "Trainer",
    "DistributedTrainer",
    "LanguageModelMetrics",
    "MemoryMetrics",
    "GenerationMetrics",
    "MetricsTracker",
    "compute_metrics_from_batch",
    
    # Visualization utilities
    "plot_attention_patterns",
    "plot_memory_heatmap",
    "plot_memory_updates",
    "plot_training_metrics",
    "plot_comparison_metrics",
    "plot_text_generation",
    "visualize_model_state",
    "save_animation_frames",
    
    # Configuration
    "ModelConfig",
    "TrainingConfig",
    "DefaultConfig",
    
    # Logging utilities
    "get_logger",
    "Logger",
    "log_step_info",
    "log_model_info",
    "log_memory_usage",
    "debug_tensor"
]