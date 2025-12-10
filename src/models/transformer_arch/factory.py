"""Factory functions for creating models with consistent configurations."""

from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn

from src.models.transformer_arch.titans_transformer import TitansTransformer, TitansLM
from src.models.transformer_arch.vanilla_transformer import VanillaTransformer, VanillaLM
from src.models.transformer_arch.kronos_transformer import KRONOSTransformer, KRONOSLM
from src.models.utils import init_weights


def apply_improved_initialization(model: nn.Module, scale: float = 1.0):
    """Apply improved initialization to model weights.
    
    Uses Xavier uniform for linear layers and normal for embeddings.
    
    Args:
        model: Model to initialize
        scale: Scaling factor for initialization
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=scale)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    return model


def create_model_pair(
    model_type: str,
    dim: int,
    depth: int,
    vocab_size: int,
    seq_len: int,
    dim_head: int = 64,
    heads: int = 8,
    segment_len: int = 512,
    ff_mult: int = 4,
    attn_dropout: float = 0.0,
    ff_dropout: float = 0.0,
    emb_dropout: float = 0.0,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    memory_kwargs: Optional[Dict[str, Any]] = None,
    use_memory: bool = True,
    use_flex_attn: bool = True,
    **kwargs
) -> Tuple:
    """Create a pair of models (Titans and baseline) with consistent configurations.
    
    Args:
        model_type: Type of model to create ('titans', 'vanilla', or 'both')
        dim: Model dimension
        depth: Number of transformer layers
        vocab_size: Vocabulary size
        seq_len: Maximum sequence length
        dim_head: Dimension of each attention head
        heads: Number of attention heads
        segment_len: Length of segments for segmented attention
        ff_mult: Multiplier for feed-forward dimension
        attn_dropout: Dropout rate for attention
        ff_dropout: Dropout rate for feed-forward network
        emb_dropout: Dropout rate for embeddings
        temperature: Temperature for generation
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
        memory_kwargs: Additional arguments for memory module
        use_memory: Whether to use memory in Titans model
        use_flex_attn: Whether to use flexible attention in Titans model
        
    Returns:
        Tuple containing Kronos, Titans, and Vanilla models when model_type is 'all'; or the requested subset.
    """
    memory_kwargs = memory_kwargs or {}
    # Create Kronos model
    kronos_model = None
    if model_type in ['kronos', 'all']:
        kronos_transformer = KRONOSTransformer(
            dim=dim,
            depth=depth,
            vocab_size=vocab_size,
            seq_len=seq_len,
            dim_head=dim_head,
            heads=heads,
            segment_len=segment_len,
            **kwargs
        )
        kronos_transformer = apply_improved_initialization(kronos_transformer)
        kronos_model = KRONOSLM(
            transformer=kronos_transformer,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
    # Create Titans model
    titans_model = None
    if model_type in ['titans', 'both', 'all']:
        titans_transformer = TitansTransformer(
            dim=dim,
            depth=depth,
            vocab_size=vocab_size,
            seq_len=seq_len,
            dim_head=dim_head,
            heads=heads,
            segment_len=segment_len,
            ff_mult=ff_mult,
            use_memory=use_memory,
            use_flex_attn=use_flex_attn,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            emb_dropout=emb_dropout,
            memory_kwargs=memory_kwargs,
            **kwargs
        )
        
        titans_transformer = apply_improved_initialization(titans_transformer)
        
        titans_model = TitansLM(
            transformer=titans_transformer,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
    # Create vanilla model
    vanilla_model = None
    if model_type in ['vanilla', 'both', 'all']:
        vanilla_transformer = VanillaTransformer(
            dim=dim,
            depth=depth,
            vocab_size=vocab_size,
            seq_len=seq_len,
            dim_head=dim_head,
            heads=heads,
            ff_mult=ff_mult,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            emb_dropout=emb_dropout,
            **kwargs
        )
        
        vanilla_transformer = apply_improved_initialization(vanilla_transformer)
        
        vanilla_model = VanillaLM(
            transformer=vanilla_transformer,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
    # Return models based on requested type
    if model_type == 'all':
        return kronos_model, titans_model, vanilla_model
    if model_type == 'both':
        return titans_model, vanilla_model
    if model_type == 'kronos':
        return kronos_model
    if model_type == 'titans':
        return titans_model
    if model_type == 'vanilla':
        return vanilla_model
    # Default fallback
    return titans_model, vanilla_model
