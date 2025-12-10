"""Vanilla Transformer implementation for baseline comparison."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List, Callable

from src.models.utils import exists, default
from src.models.attention import ContinuousAxialPositionalEmbedding
from src.models.transformer_arch.titans_transformer import TransformerState


class VanillaTransformerBlock(nn.Module):
    """Standard Transformer block with self-attention and feed-forward network."""
    
    def __init__(
        self,
        dim: int,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = dim_head * heads
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=attn_dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(ff_dropout)
        )
    
    def forward(self, x, attn_mask=None):
        """Forward pass through the transformer block."""
        # Self-attention
        residual = x
        x = self.norm1(x)
        x_attn, _ = self.attention(x, x, x, attn_mask=attn_mask)
        x = residual + x_attn
        
        # Feed-forward network
        residual = x
        x = self.norm2(x)
        x = residual + self.ff(x)
        
        return x


class VanillaTransformer(nn.Module):
    """Standard Transformer model without memory components."""
    
    def __init__(
        self,
        dim: int,
        depth: int,
        vocab_size: int,
        seq_len: int,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        emb_dropout: float = 0.0,
        tie_word_embeddings: bool = False,
        use_axial_pos_emb: bool = True,
        num_axial_dims: int = 2,
    ):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        # Token embedding
        self.token_emb = nn.Embedding(vocab_size, dim)
        
        # Positional embedding
        self.pos_emb = None
        if use_axial_pos_emb:
            self.pos_emb = ContinuousAxialPositionalEmbedding(dim, num_axial_dims=num_axial_dims)
        else:
            self.pos_emb = nn.Embedding(seq_len, dim)
        
        self.emb_dropout = nn.Dropout(emb_dropout)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            VanillaTransformerBlock(
                dim=dim,
                dim_head=dim_head,
                heads=heads,
                ff_mult=ff_mult,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout
            )
            for _ in range(depth)
        ])
        
        # Final normalization and output projection
        self.norm = nn.LayerNorm(dim)
        
        # Output head
        if tie_word_embeddings:
            self.to_logits = lambda x: x @ self.token_emb.weight.t()
        else:
            self.to_logits = nn.Linear(dim, vocab_size)
    
    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the transformer."""
        device = x.device
        batch_size, seq_len = x.shape
        
        # Get token embeddings
        x = self.token_emb(x)
        
        # Add positional embeddings
        if exists(self.pos_emb):
            if isinstance(self.pos_emb, ContinuousAxialPositionalEmbedding):
                if exists(positions):
                    pos_emb = self.pos_emb.forward_with_seq_len(seq_len)
                    pos_emb = pos_emb[positions]
                else:
                    pos_emb = self.pos_emb.forward_with_seq_len(seq_len)
                    pos_emb = pos_emb.unsqueeze(0).expand(batch_size, -1, -1)
            else:
                pos = positions if exists(positions) else torch.arange(seq_len, device=device)
                pos_emb = self.pos_emb(pos)
            
            x = x + pos_emb
        
        x = self.emb_dropout(x)
        
        # Create causal mask if no mask is provided
        if attn_mask is None and seq_len > 1:
            attn_mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1)
        
        # Process through transformer blocks
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        
        # Final normalization
        x = self.norm(x)
        
        # Project to logits
        logits = self.to_logits(x)
        
        return logits

    # Adding state handling for compatibility with Titans
    def forward_with_state(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        state: Optional[TransformerState] = None,
        return_state: bool = False,
        disable_flex_attn: bool = False,
        flex_attn_fn: Optional[Callable] = None,
    ) -> Tuple[torch.Tensor, Optional[TransformerState]]:
        """Forward pass with state tracking for compatibility with Titans interface."""
        logits = self.forward(x, positions)
        
        if return_state:
            # Create empty state for compatibility
            empty_state = TransformerState(
                memory_states=[None] * len(self.layers),
                kv_caches=[None] * len(self.layers),
                value_residuals=[None] * len(self.layers)
            )
            return logits, empty_state
        
        return logits


class VanillaLM(nn.Module):
    """Vanilla Language Model for text generation."""
    
    def __init__(
        self,
        transformer: VanillaTransformer,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ):
        super().__init__()
        self.transformer = transformer
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
    
    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        return_state: bool = False,
    ) -> torch.Tensor:
        """Forward pass."""
        return self.transformer(x, positions)
    
    def generate(
        self,
        x: torch.Tensor,
        max_length: int,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Generate text autoregressively."""
        
        temperature = default(temperature, self.temperature)
        top_k = default(top_k, self.top_k)
        top_p = default(top_p, self.top_p)
        
        # Initialize output with input
        out = x
        
        # Store logits if requested
        all_logits = []
        
        # Generate tokens
        for _ in range(max_length):
            # Get next token logits
            logits = self.transformer(out)
            
            # Get logits for the last token
            next_token_logits = logits[:, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-k sampling
            if exists(top_k):
                values, indices = next_token_logits.topk(top_k, dim=-1)
                next_token_logits.zero_()
                next_token_logits.scatter_(-1, indices, values)
            
            # Apply top-p (nucleus) sampling
            if exists(top_p) and top_p < 1.0:
                sorted_logits, sorted_indices = next_token_logits.sort(descending=True, dim=-1)
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
                next_token_logits[indices_to_remove] = -float('Inf')
            
            # Convert to probabilities and sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Save logits if requested
            if return_logits:
                all_logits.append(logits)
            
            # Append to output
            out = torch.cat((out, next_token), dim=1)
        
        out = out.squeeze(dim=0)

        if return_logits:
            return out, torch.cat(all_logits, dim=1)
        
        return out, None
