"""Segmented attention mechanisms for the Titans model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable, Dict, Any
from einops import rearrange

from .utils import exists, create_sliding_window_mask

class SegmentedAttention(nn.Module):
    """Segmented attention with cross-segment context integration for Titans model."""
    def __init__(
        self,
        dim: int,
        dim_head: int = 64,
        heads: int = 8,
        segment_len: int = 512,
        use_flex_attn: bool = False,
        accept_value_residual: bool = False,
        num_longterm_mem_tokens: int = 0,
        num_persist_mem_tokens: int = 0,
        sliding: bool = False,
        attn_dropout: float = 0.0,
        output_dropout: float = 0.0,
        cross_segment_fusion: bool = True,
        global_tokens_ratio: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.dim_head = dim_head
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.sliding = sliding
        self.segment_len = segment_len
        self.use_flex_attn = use_flex_attn
        self.accept_value_residual = accept_value_residual
        self.num_longterm_mem_tokens = num_longterm_mem_tokens
        self.num_persist_mem_tokens = num_persist_mem_tokens
        self.cross_segment_fusion = cross_segment_fusion
        self.global_tokens_ratio = global_tokens_ratio

        # Input projections
        self.to_q = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(dim, self.inner_dim, bias=False)
        
        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, dim),
            nn.Dropout(output_dropout)
        )
        
        # Attention dropout
        self.attn_dropout = nn.Dropout(attn_dropout)
        
        # Cross-segment context fusion layers
        if cross_segment_fusion:
            self.global_context_norm = nn.LayerNorm(dim_head)
            self.global_context_gate = nn.Sequential(
                nn.Linear(dim_head * 2, dim_head),
                nn.Sigmoid()
            )

    def forward(
        self,
        x: torch.Tensor,
        value_residual: Optional[torch.Tensor] = None,
        disable_flex_attn: bool = False,
        flex_attn_fn: Optional[Callable] = None,
        output_gating: Optional[torch.Tensor] = None,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_attn_scores: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass.

        Returns:
            out: [batch, seq_len, dim]
            cache_dict: {'cache': (k, v), 'value_residual': tensor or None}
        """
        b, seq_len, _ = x.shape
        use_flex = self.use_flex_attn and not disable_flex_attn
        total_mem = self.num_longterm_mem_tokens + self.num_persist_mem_tokens

        # Projections
        q = rearrange(self.to_q(x), 'b n (h d) -> b h n d', h=self.heads)
        k_new = rearrange(self.to_k(x), 'b n (h d) -> b h n d', h=self.heads)
        v_new = rearrange(self.to_v(x), 'b n (h d) -> b h n d', h=self.heads)

        # KV caching
        if cache is not None:
            k_old, v_old = cache
            # check shapes
            if k_old.shape[:3] == k_new.shape[:3]:
                k = torch.cat([k_old, k_new], dim=2)
                v = torch.cat([v_old, v_new], dim=2)
                if self.sliding:
                    max_win = self.segment_len + total_mem
                    k = k[:, :, -max_win:]
                    v = v[:, :, -max_win:]
            else:
                k, v = k_new, v_new
        else:
            k, v = k_new, v_new

        # Update cache (detach to avoid grads)
        new_cache = (k.detach(), v.detach())

        # Local attention scores
        scale = self.dim_head ** -0.5
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        if self.sliding:
            mask = create_sliding_window_mask(seq_len, self.segment_len, device=x.device)
            scores = scores.masked_fill(~mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        local_out = torch.matmul(attn, v)

        # Cross-segment fusion
        if self.cross_segment_fusion:
            num_global = max(1, int(seq_len * self.global_tokens_ratio))
            key_norms = k.norm(dim=-1)  # [b, h, seq_len]
            top_idx = key_norms.topk(num_global, dim=-1)[1]
            global_k = torch.gather(k, 2, top_idx.unsqueeze(-1).expand(-1,-1,-1,self.dim_head))
            global_v = torch.gather(v, 2, top_idx.unsqueeze(-1).expand(-1,-1,-1,self.dim_head))

            g_scores = torch.matmul(q, global_k.transpose(-1, -2)) * scale
            g_attn = F.softmax(g_scores, dim=-1)
            global_out = torch.matmul(g_attn, global_v)

            # normalize & gate
            global_out = self.global_context_norm(global_out)
            cat = torch.cat([local_out, global_out], dim=-1)   # [b,h,n,2*d]
            gate = self.global_context_gate(cat)
            local_out = gate * local_out + (1 - gate) * global_out

        # reshape
        out = rearrange(local_out, 'b h n d -> b n (h d)')

        # value residual
        val_res = None
        if value_residual is not None and self.accept_value_residual:
            out = out + value_residual
            val_res = out.detach()

        # output gating
        if output_gating is not None:
            out = out * output_gating

        # final projection
        out = self.to_out(out)

        if return_attn_scores:
            return out, (val_res, new_cache), attn

        return out, (val_res, new_cache)

class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""
    
    def __init__(
        self,
        dim: int,
        mult: int = 4,
        dropout: float = 0.0
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(x)


class ContinuousAxialPositionalEmbedding(nn.Module):
    """Continuous axial positional embedding"""
    def __init__(
        self,
        dim: int,
        num_axial_dims: int = 2
    ):
        super().__init__()
        assert dim % num_axial_dims == 0, \
            f"Dimension {dim} must be divisible by number of axial dims {num_axial_dims}"
        self.dim = dim
        self.num_axial_dims = num_axial_dims
        # Create embeddings for each axial axis
        self.embeddings = nn.ModuleList([
            nn.Linear(1, dim // num_axial_dims) 
            for _ in range(num_axial_dims)
        ])

    def maybe_derive_outer_dims(self, seq_len: int, chunk_sizes: Optional[Tuple[int, ...]] = None) -> Tuple[int, ...]:
        # Try to factor seq_len into num_axial_dims axes
        if chunk_sizes:
            # validate provided tuple
            prod = 1
            for c in chunk_sizes:
                prod *= c
            if prod > seq_len:
                raise ValueError(f"Product {prod} exceeds seq_len {seq_len}")
            # compute residual dims to exactly match seq_len
            residual = seq_len // prod
            if prod * residual != seq_len:
                raise ValueError(f"Given chunk_sizes {chunk_sizes} cannot exactly tile seq_len {seq_len}")
            return (*chunk_sizes, residual) if residual > 1 else chunk_sizes
        # no chunk_sizes: split roughly evenly
        axis_len = int(seq_len ** (1 / self.num_axial_dims))
        outer = [axis_len] * self.num_axial_dims
        # adjust last dimension to match exactly
        prod = 1
        for d in outer[:-1]: prod *= d
        outer[-1] = seq_len // prod
        return tuple(outer)

    def _compute_embeddings(self, outer_dims: Tuple[int, ...]) -> torch.Tensor:
        device = next(self.parameters()).device
        if len(outer_dims) > self.num_axial_dims:
            raise ValueError(f"outer_dims {outer_dims} too many for {self.num_axial_dims} axes")
        # generate per-axis positions in [0,1]
        axis_positions = [torch.linspace(0, 1, steps=d, device=device) for d in outer_dims]
        embeds = []
        # axis_positions: A list of 1D tensors, where axis_positions[i] contains the coordinates for the i-th axial dimension. 
        # outer_dims: A tuple defining the shape of the spatial/sequential dimensions, e.g., (H, W) or (SeqDim1, SeqDim2)

        for axis_idx, pos_coords_for_this_axis in enumerate(axis_positions):
            # pos_coords_for_this_axis has shape [outer_dims[axis_idx]]
            
            # 1. Prepare input for the linear layer for this axis
            p = pos_coords_for_this_axis.unsqueeze(-1) # shape: [outer_dims[axis_idx], 1]
            
            # 2. Get the embedding slice for this axis from its dedicated linear layer
            axial_embedding_slice = self.embeddings[axis_idx](p) # shape: [outer_dims[axis_idx], self.dim // self.num_axial_dims]
            
            # Store the embedding dimension for this slice (e.g., 192)
            dim_per_axis_slice = axial_embedding_slice.shape[-1]

            # 3. Reshape for broadcasting:
            # The goal is to make axial_embedding_slice have a shape like:
            # [1, ..., 1, outer_dims[axis_idx], 1, ..., 1, dim_per_axis_slice]
            
            # Construct the target shape for the .view() operation
            view_shape = [1] * len(outer_dims)      # Initialize with 1s for all axial dimensions
            view_shape[axis_idx] = outer_dims[axis_idx] # Set the current axis's dimension to its actual size
            view_shape.append(dim_per_axis_slice)   # Append the embedding dimension of the slice
            
            # Reshape the axial_embedding_slice using .view() to this target shape
            reshaped_for_expansion = axial_embedding_slice.view(view_shape)
            
            # 4. Expand the reshaped embedding to the full target shape for this axial component's contribution
            expanded_embedding = reshaped_for_expansion.expand(*outer_dims, dim_per_axis_slice) # The target shape for expansion is [*outer_dims, dim_per_axis_slice]
            
            embeds.append(expanded_embedding)
        # pad unused axes with zeros
        if len(outer_dims) < self.num_axial_dims:
            pad_dim = (self.num_axial_dims - len(outer_dims)) * (self.dim // self.num_axial_dims)
            zeros = torch.zeros(*outer_dims, pad_dim, device=device)
            embeds.append(zeros)
        # concat
        combined = torch.cat(embeds, dim=-1)
        return combined

    def forward_with_seq_len(
        self,
        seq_len: int,
        chunk_sizes: Optional[Tuple[int, ...]] = None
    ) -> torch.Tensor:
        """Compute [seq_len, dim] embeddings without truncation/padding artifacts."""
        outer = self.maybe_derive_outer_dims(seq_len, chunk_sizes)
        emb = self._compute_embeddings(outer)  # shape [*outer, dim]
        # reshape exactly to seq_len
        total = 1
        for d in outer:
            total *= d
        emb = emb.reshape(total, self.dim)
        # if over, truncate; if under, pad
        if total != seq_len:
            if total > seq_len:
                emb = emb[:seq_len]
            else:
                pad = torch.zeros(seq_len - total, self.dim, device=emb.device)
                emb = torch.cat([emb, pad], dim=0)
        return emb

    def forward(
        self,
        outer_dims: Tuple[int, ...],
        return_factorized: bool = False
    ) -> torch.Tensor:
        emb = self._compute_embeddings(outer_dims)
        if return_factorized:
            return emb
        # reshape to [seq_len, dim]
        total = 1
        for d in outer_dims:
            total *= d
        return emb.reshape(total, self.dim)