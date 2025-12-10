"""Model components for the Titans architecture."""

__version__ = "0.1.0"

# Transformer components
from .transformer_arch import (
    TransformerState,
    TransformerBlock,
    TitansTransformer,
    TitansLM,

    VanillaTransformer,
    VanillaLM,

    KRONOSTransformer,
    KRONOSLM,

    create_model_pair,
)

# Attention mechanisms
from .attention import (
    SegmentedAttention,
    FeedForward,
    ContinuousAxialPositionalEmbedding
)

# Memory components
from .memory import (
    NeuralMemoryState,
    NeuralMemory
)

# Utility functions
from .utils import (
    exists,
    default,
    create_sliding_window_mask,
    create_block_diagonal_mask,
    init_weights,
    get_slopes,
    chunk_sequence,
    merge_chunks,
    top_k_top_p_filtering,
    create_causal_mask,
    calculate_entropy,
    cosine_similarity,
)

# Define what's available when using "from titans.models import *"
__all__ = [
    # Transformer
    "TransformerState",
    "TransformerBlock",
    "TitansTransformer",
    "TitansLM",
    
    "VanillaTransformer",
    "VanillaLM",

    "KRONOSTransformer",
    "KRONOSLM",

    "create_model_pair",
    
    # Attention
    "SegmentedAttention",
    "FeedForward",
    "ContinuousAxialPositionalEmbedding",
    
    # Memory
    "NeuralMemoryState",
    "NeuralMemory",
    
    # Utils
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
    "calculate_entropy",
    "cosine_similarity",
]