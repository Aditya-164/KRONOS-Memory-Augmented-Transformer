"""Helper functions for the Titans model implementation."""

import torch
from typing import Optional, Tuple, List, Dict, Any, Union, Callable
import math
from einops import rearrange, repeat


import torch.nn as nn
import torch.nn.functional as F


# Used utility function
def exists(val: Any) -> bool:
    """Check if a value is not None.
    
    Args:
        val: Value to check
        
    Returns:
        bool: True if the value is not None, False otherwise
    """
    return val is not None


def default(val: Any, d: Any) -> Any:
    """Return the value if it exists, otherwise return the default.
    
    Args:
        val: Value to check
        d: Default value
        
    Returns:
        Either val if it exists, or the default value d
    """
    return val if exists(val) else d


def create_sliding_window_mask(
    seq_len: int, 
    window_size: int, 
    device: torch.device
) -> torch.Tensor:
    """Create a sliding window attention mask.
    
    Args:
        seq_len: Length of sequence
        window_size: Size of the sliding window
        device: Device to create mask on
        
    Returns:
        mask: Boolean mask tensor [seq_len, seq_len]
    """
    # Create indices
    indices = torch.arange(seq_len, device=device)
    
    # Create a distance matrix
    distances = indices.unsqueeze(1) - indices.unsqueeze(0)
    
    # Create mask: attend within window_size
    mask = distances.abs() <= (window_size // 2)
    
    return mask


def calculate_entropy(self, probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Calculates entropy of a probability distribution."""
    log_probs = torch.log(probs + eps)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy


def cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity between two tensors.
    
    Args:
        x: First tensor with shape (..., D)
        y: Second tensor with shape (..., D)
        
    Returns:
        Tensor of cosine similarities with shape (...,)
    """
    x_norm = F.normalize(x, p=2, dim=-1)
    y_norm = F.normalize(y, p=2, dim=-1)
    return torch.matmul(x_norm, y_norm.transpose(-2, -1))


# Unused utility function
def create_block_diagonal_mask(
    seq_len: int, 
    block_size: int, 
    device: torch.device,
    mem_tokens: int = 0
) -> torch.Tensor:
    """Create a block diagonal attention mask with optional memory tokens.
    
    Args:
        seq_len: Length of sequence
        block_size: Size of each attention block
        device: Device to create mask on
        mem_tokens: Number of memory tokens that can attend globally
        
    Returns:
        mask: Boolean mask tensor [seq_len, seq_len]
    """
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
    
    # Allow attention within blocks
    for i in range(0, seq_len, block_size):
        end = min(i + block_size, seq_len)
        mask[i:end, i:end] = True
    
    # Allow memory tokens to attend to everything and vice versa
    if mem_tokens > 0:
        mask[:mem_tokens, :] = True
        mask[:, :mem_tokens] = True
    
    return mask


def init_weights(module: nn.Module, scale: float = 1.0):
    """Initialize the weights of the neural network module.
    
    Uses a scaled Xavier uniform initialization for linear layers
    and zeros for biases.
    
    Args:
        module: PyTorch module to initialize
        scale: Scaling factor for the weights
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=scale)
        if exists(module.bias):
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=0.02)


def get_slopes(n_heads: int) -> List[float]:
    """Get attention slopes for rotary position embeddings.
    
    Args:
        n_heads: Number of attention heads
        
    Returns:
        List of slopes for each head
    """
    def get_slopes_power_of_2(n):
        start = 2**(-(2**-(math.log2(n)-3)))
        ratio = start
        return [start * (ratio ** i) for i in range(n)]

    if math.log2(n_heads).is_integer():
        return get_slopes_power_of_2(n_heads)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes(n_heads - closest_power_of_2)


def chunk_sequence(x: torch.Tensor, chunk_size: int) -> List[torch.Tensor]:
    """Split a sequence tensor into chunks of specified size.
    
    Args:
        x: Input tensor [batch, seq_len, dim]
        chunk_size: Size of each chunk
        
    Returns:
        List of chunked tensors, each of shape [batch, chunk_size, dim]
    """
    batch, seq_len, dim = x.shape
    
    # Calculate padding needed to make seq_len divisible by chunk_size
    padding = (chunk_size - seq_len % chunk_size) % chunk_size
    
    if padding > 0:
        # Pad the sequence
        x = F.pad(x, (0, 0, 0, padding), value=0)
        seq_len = seq_len + padding
    
    # Reshape into chunks
    chunks = rearrange(x, 'b (n c) d -> b n c d', c=chunk_size)
    
    # Convert to list of tensors
    return [chunks[:, i, :, :] for i in range(chunks.shape[1])]


def merge_chunks(chunks: List[torch.Tensor], original_length: int) -> torch.Tensor:
    """Merge a list of chunk tensors back into a single sequence.
    
    Args:
        chunks: List of chunk tensors, each of shape [batch, chunk_size, dim]
        original_length: Original sequence length before chunking
        
    Returns:
        Merged tensor [batch, original_length, dim]
    """
    # Stack chunks along a new dimension
    stacked = torch.stack(chunks, dim=1)
    
    # Reshape to flattened sequence
    batch, n_chunks, chunk_size, dim = stacked.shape
    merged = rearrange(stacked, 'b n c d -> b (n c) d')
    
    # Truncate to original length
    return merged[:, :original_length, :]


def top_k_top_p_filtering(
    logits: torch.Tensor, 
    top_k: Optional[int] = None, 
    top_p: Optional[float] = None, 
    filter_value: float = -float('Inf')
) -> torch.Tensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.
    
    Args:
        logits: Logits distribution of shape [batch, vocab]
        top_k: Keep only the top-k tokens with highest probability
        top_p: Keep the top tokens with cumulative probability >= top_p
        filter_value: Value to assign to filtered tokens
        
    Returns:
        Filtered logits
    """
    if not exists(top_k) and not exists(top_p):
        return logits
    
    logits_modified = logits.clone()
    
    if exists(top_k):
        # Remove all tokens with a probability less than the
        # last token of the top-k
        indices_to_remove = logits_modified < torch.topk(logits_modified, top_k)[0][..., -1, None]
        logits_modified[indices_to_remove] = filter_value
        
    if exists(top_p) and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits_modified, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits_modified[indices_to_remove] = filter_value
        
    return logits_modified


def create_causal_mask(
    seq_len: int, 
    device: torch.device
) -> torch.Tensor:
    """Create a causal attention mask.
    
    Args:
        seq_len: Length of sequence
        device: Device to create mask on
        
    Returns:
        mask: Boolean causal mask tensor [seq_len, seq_len]
    """
    indices = torch.arange(seq_len, device=device)
    mask = indices.unsqueeze(0) >= indices.unsqueeze(1)
    return mask