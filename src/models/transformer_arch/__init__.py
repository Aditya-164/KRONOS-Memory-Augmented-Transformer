# filepath: /home/prakhar/courses/INLP/NaturalStupidity/titans/models/transformer_arch/__init__.py
"""Transformer architecture components for the project.

This file exposes:
- Titans transformer components from titans_transformer.py
- Vanilla transformer components from vanilla_transformer.py
- Factory functions from factory.py
"""
__version__ = "0.1.0"

# Titans transformer components
from .titans_transformer import (
    TransformerState,
    TransformerBlock,
    TitansTransformer,
    TitansLM,
)

# Vanilla transformer components
from .vanilla_transformer import (
    VanillaTransformer,
    VanillaLM,
)

# KRONOS transformer components
from .kronos_transformer import (
    KRONOSTransformer,
    KRONOSLM,
) 

# Factory functions for model creation and initialization
from .factory import (
    create_model_pair,
    apply_improved_initialization,
)

__all__ = [
    # Titans transformer
    "TransformerState",
    "TransformerBlock",
    "TitansTransformer",
    "TitansLM",

    # Vanilla transformer
    "VanillaTransformer",
    "VanillaLM",

    # KRONOS transformer
    "KRONOSTransformer",
    "KRONOSLM",
    
    # Factory functions
    "create_model_pair",
    "apply_improved_initialization",
]