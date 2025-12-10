"""Transformer implementation for the Titans model."""

import torch
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List, Callable
from src.models.memory import NeuralMemory, NeuralMemoryState


import torch.nn as nn
import torch.nn.functional as F

from src.models.attention import (
    SegmentedAttention, 
    FeedForward, 
    ContinuousAxialPositionalEmbedding,
)

from src.models.utils import (
    default,
    exists,
)


@dataclass
class TransformerState:
    """State of the transformer model for stateful execution."""
    memory_states: List[NeuralMemoryState]
    kv_caches: List[Tuple[torch.Tensor, torch.Tensor]]
    value_residuals: List[torch.Tensor]

class TransformerBlock(nn.Module):
    """Enhanced transformer block with segmented attention and optimized memory."""
    
    def __init__(
        self,
        dim: int,
        dim_head: int = 64,
        heads: int = 8,
        segment_len: int = 512,
        ff_mult: int = 4,
        memory_dim: Optional[int] = None,
        chunk_size: Optional[int] = None,
        use_memory: bool = True,
        use_flex_attn: bool = False,
        accept_value_residual: bool = False,
        num_longterm_mem_tokens: int = 0,
        num_persist_mem_tokens: int = 0,
        sliding: bool = False,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        memory_kwargs: Dict[str, Any] = None,
        cross_segment_fusion: bool = False,  
        adaptive_forgetting: bool = False,   
        layer_gate: bool = True,            
    ):
        super().__init__()
        self.dim = dim
        self.use_memory = use_memory
        self.segment_len = segment_len
        self.layer_gate = layer_gate
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim) if use_memory else None
        
        # Attention mechanism
        self.attention = SegmentedAttention(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            segment_len=segment_len,
            use_flex_attn=use_flex_attn,
            accept_value_residual=accept_value_residual,
            num_longterm_mem_tokens=num_longterm_mem_tokens,
            num_persist_mem_tokens=num_persist_mem_tokens,
            sliding=sliding,
            attn_dropout=attn_dropout,
            output_dropout=ff_dropout,
            cross_segment_fusion=cross_segment_fusion 
        )
        
        # Feed-forward network
        self.ff = FeedForward(
            dim=dim,
            mult=ff_mult,
            dropout=ff_dropout
        )
        
        # Memory module
        self.memory = None
        if use_memory:
            memory_kwargs = default(memory_kwargs, {})
            chunk_size = default(chunk_size, segment_len)
            memory_dim = default(memory_dim, dim)
            
            # Add adaptive forgetting to memory kwargs
            if "adaptive_forgetting" not in memory_kwargs:
                memory_kwargs["adaptive_forgetting"] = adaptive_forgetting
            
            self.memory = NeuralMemory(
                dim=dim,
                chunk_size=chunk_size,
                memory_dim=memory_dim,
                **memory_kwargs
            )
        
        # Layer gating mechanism - dynamically controls information flow
        if layer_gate:
            self.gate_net = nn.Sequential(
                nn.Linear(dim, dim // 4),
                nn.SiLU(),
                nn.Linear(dim // 4, 3 if use_memory else 2),
                nn.Sigmoid()
            )
    
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[Optional[NeuralMemoryState], Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]] = None,
        disable_flex_attn: bool = False,
        flex_attn_fn: Optional[Callable] = None,
        output_gating: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[Optional[NeuralMemoryState], Tuple[torch.Tensor, torch.Tensor], torch.Tensor]]:
        """Forward pass with enhanced layer gating and optimized information flow.
        
        Args:
            x: Input tensor [batch, seq_len, dim]
            state: Previous state (memory_state, kv_cache, value_residual)
            disable_flex_attn: Whether to disable flex attention
            flex_attn_fn: Custom attention function
            output_gating: Output gating tensor
            
        Returns:
            output: Transformed output [batch, seq_len, dim]
            state: Updated state (memory_state, kv_cache, value_residual)
        """
        memory_state, kv_cache, value_residual = None, None, None
        if exists(state):
            memory_state, kv_cache, value_residual = state
        
        # Calculate layer-specific gates if enabled
        gates = None
        if self.layer_gate:
            # Create gates for attention, FF, and memory (if used)
            # This allows the model to dynamically control information flow
            gates = self.gate_net(x.mean(dim=1, keepdim=True))
            attn_gate, ff_gate = gates[:, :, 0:1], gates[:, :, 1:2]
            
            # Extract memory gate if using memory
            memory_gate = gates[:, :, 2:3] if self.use_memory and gates.shape[-1] > 2 else None
        
        # Self-attention with dynamic gating
        residual = x
        x_attn = self.norm1(x)
        
        attn_output, (new_value_residual, new_kv_cache) = self.attention(
            x_attn,
            value_residual=value_residual,
            disable_flex_attn=disable_flex_attn,
            flex_attn_fn=flex_attn_fn,
            output_gating=output_gating,
            cache=kv_cache
        )
        
        # Apply attention gate if enabled
        if self.layer_gate:
            attn_output = attn_output * attn_gate
        
        x = residual + attn_output
        
        # Feed-forward network with dynamic gating
        residual = x
        x_ff = self.norm2(x)
        ff_output = self.ff(x_ff)
        
        # Apply FF gate if enabled
        if self.layer_gate:
            ff_output = ff_output * ff_gate
            
        x = residual + ff_output
        
        # Memory module with dynamic gating
        new_memory_state = None
        if self.use_memory and exists(self.memory):
            residual = x
            x_mem = self.norm3(x)
            memory_output, new_memory_state = self.memory(
                x_mem,
                state=memory_state,
                prev_weights=None if not exists(memory_state) else memory_state.updates
            )
            
            # Apply memory gate if enabled
            if self.layer_gate and exists(memory_gate):
                memory_output = memory_output * memory_gate
                
            x = residual + memory_output
        
        return x, (new_memory_state, new_kv_cache, new_value_residual)

class TitansTransformer(nn.Module):
    """Titans Transformer model with segmented attention and neural memory."""
    def __init__(
        self,
        dim: int,
        depth: int,
        vocab_size: int,
        seq_len: int,
        dim_head: int = 64,
        heads: int = 8,
        segment_len: int = 512,
        ff_mult: int = 4,
        memory_dim: Optional[int] = None,
        chunk_size: Optional[int] = None,
        use_memory: bool = False,
        use_flex_attn: bool = False,
        accept_value_residual: bool = False,
        num_longterm_mem_tokens: int = 0,
        num_persist_mem_tokens: int = 0,
        sliding: bool = False,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        emb_dropout: float = 0.0,
        memory_kwargs: Dict[str, Any] = None,
        tie_word_embeddings: bool = False,
        use_axial_pos_emb: bool = True,
        num_axial_dims: int = 2,
    ):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.tie_word_embeddings = tie_word_embeddings

        # Token embedding
        self.token_emb = nn.Embedding(vocab_size, dim)

        # Positional embedding
        if use_axial_pos_emb:
            self.pos_emb = ContinuousAxialPositionalEmbedding(dim, num_axial_dims=num_axial_dims)
            # Precompute full-range embeddings once -> causes tensors grad to be placed on cpu (not shifted during model.to call)
            # full_pe = self.pos_emb.forward_with_seq_len(seq_len)  # [seq_len, dim]
            # self.register_buffer('full_pos_emb', full_pe)        # buffer [seq_len, dim]
        else:
            self.pos_emb = nn.Embedding(seq_len, dim)

        self.emb_dropout = nn.Dropout(emb_dropout)

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                dim_head=dim_head,
                heads=heads,
                segment_len=segment_len,
                ff_mult=ff_mult,
                memory_dim=memory_dim,
                chunk_size=chunk_size,
                use_memory=use_memory,
                use_flex_attn=use_flex_attn,
                accept_value_residual=accept_value_residual,
                num_longterm_mem_tokens=num_longterm_mem_tokens,
                num_persist_mem_tokens=num_persist_mem_tokens,
                sliding=sliding,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                memory_kwargs=memory_kwargs
            ) for _ in range(depth)]
        )

        # Normalization & output
        self.norm = nn.LayerNorm(dim)
        if tie_word_embeddings:
            self.to_logits = lambda x: x @ self.token_emb.weight.t()
        else:
            self.to_logits = nn.Linear(dim, vocab_size)

    def forward_with_state(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        state: Optional[TransformerState] = None,
        return_state: bool = False,
        disable_flex_attn: bool = False,
        flex_attn_fn: Optional[Callable] = None,
    ) -> Tuple[torch.Tensor, Optional[TransformerState]]:
        batch_size, seq_len = x.shape
        device = x.device

        # Unpack prior state
        memory_states = state.memory_states if state is not None else None
        kv_caches     = state.kv_caches     if state is not None else None
        value_resids  = state.value_residuals if state is not None else None

        # Token embeddings
        x = self.token_emb(x)  # [batch, seq_len, dim]

        # Positional embeddings
        if isinstance(self.pos_emb, ContinuousAxialPositionalEmbedding):  
            if positions is not None:
                max_abs_idx_in_batch = positions.max().item()
                required_pe_table_len = max_abs_idx_in_batch + 1 # the length L must be at least `max_abs_idx_in_batch + 1`
                pe_table = self.pos_emb.forward_with_seq_len(required_pe_table_len) # shape [required_pe_table_len, dim].
                # positions: [batch, seq_len] of absolute indices
                pos_emb = pe_table[positions]          # [batch, seq_len, dim]
            else:
                # no explicit positions: use 0..seq_len-1
                pe_table_for_input_chunk = self.pos_emb.forward_with_seq_len(seq_len) # Shape: [seq_len, dim]
                pos_emb = pe_table_for_input_chunk.unsqueeze(0).expand(batch_size, -1, -1) # Shape: [B, seq_len, D]
        else:
            # nn.Embedding case
            pos_idx = positions if positions is not None else torch.arange(seq_len, device=device)
            pos_emb = self.pos_emb(pos_idx).unsqueeze(0)     # [1, seq_len, dim]
            if batch_size > 1:
                pos_emb = pos_emb.expand(batch_size, -1, -1)

        x = x + pos_emb
        x = self.emb_dropout(x)

        # Pass through layers
        new_mem_states, new_kv_caches, new_val_resids = [], [], []
        for i, layer in enumerate(self.layers):
            prev_mem = memory_states[i] if memory_states is not None else None
            prev_kv  = kv_caches[i]     if kv_caches is not None else None
            prev_val = value_resids[i]  if value_resids is not None else None

            x, (m, k, v) = layer(
                x,
                state=(prev_mem, prev_kv, prev_val),
                disable_flex_attn=disable_flex_attn,
                flex_attn_fn=flex_attn_fn
            )
            new_mem_states.append(m)
            new_kv_caches.append(k)
            new_val_resids.append(v)

        x = self.norm(x)
        logits = self.to_logits(x)

        if return_state:
            new_state = TransformerState(
                memory_states=new_mem_states,
                kv_caches=new_kv_caches,
                value_residuals=new_val_resids
            )
            return logits, new_state

        return logits

    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward_with_state(x, positions=positions, return_state=False)

class TitansLM(nn.Module):
    """Titans Language Model for text generation with KV cache optimization."""
    
    def __init__(
        self,
        transformer: nn.Module,     # should implement forward_with_state
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
        state: Optional[dict] = None,
        return_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        return self.transformer.forward_with_state(
            x, positions=positions, state=state, return_state=return_state
        )
    
    def generate(
        self,
        x: torch.Tensor,
        max_length: int,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Generate text autoregressively using KV-caching.

        Args:
          x: [batch, seq_len] input token IDs
          max_length: number of tokens to generate
          temperature, top_k, top_p: sampling parameters
          return_logits: if True, also return tensor of shape
                         [batch, generated_length, vocab_size]
        """
        device = x.device
        batch_size, seq_len = x.shape
        temperature = default(temperature, self.temperature)
        top_k       = default(top_k,       self.top_k)
        top_p       = default(top_p,       self.top_p)
        
        out = x                   # [batch, seq_len]
        state = None
        all_logits = []
        curr_len = seq_len        # absolute position of next token
        
        for _ in range(max_length):
            if state is None:
                # first pass: feed entire prompt
                input_ids = out                        # [batch, curr_len]
                # positions = [[0,1,2,...,curr_len-1]] broadcast to batch
                positions = torch.arange(curr_len, device=device).unsqueeze(0).expand(batch_size, -1)
            else:
                # subsequent passes: only the last token
                input_ids = out[:, -1:].contiguous()   # [batch,1]
                # position = [[curr_len]] broadcast
                positions = torch.full((batch_size, 1), curr_len, device=device, dtype=torch.long)
            
            # forward step with KV cache
            logits, state = self.transformer.forward_with_state(
                input_ids,
                positions=positions,
                state=state,
                return_state=True
            )
            # logits: [batch, seq_len' , vocab] where seq_len'=curr_len or 1
            
            # take logits for the last token
            next_logits = logits[:, -1, :]             # [batch, vocab]
            if temperature != 1.0:
                next_logits = next_logits / temperature
            
            # top-k
            if top_k is not None:
                vals, idx = next_logits.topk(top_k, dim=-1)
                mask = torch.full_like(next_logits, float('-inf'))
                next_logits = mask.scatter_(-1, idx, vals)
            
            # top-p
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_idx = next_logits.sort(descending=True, dim=-1)
                probs = F.softmax(sorted_logits, dim=-1)
                cum_probs = probs.cumsum(dim=-1)
                # mask tokens above threshold
                remove = cum_probs > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0]  = False
                next_logits.scatter_(-1, sorted_idx, torch.where(remove, float('-inf'), sorted_logits))
            
            # sample
            probs = F.softmax(next_logits, dim=-1)
            probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
            probs = torch.clamp(probs, min=1e-8)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            next_token = torch.multinomial(probs, num_samples=1)  # [batch,1]
            
            # save logits if requested
            if return_logits:
                # make sure logits is [batch,1,vocab]
                all_logits.append(logits[:, -1:, :])
            
            # append new token
            out = torch.cat([out, next_token], dim=1)  # [batch, curr_len+1]
            curr_len += 1
        
        out = out.squeeze(dim=0) # sequeeze to make dimesion from (1,1,) to (1, ) (flatten in dim 1)

        if return_logits:
            # concatenate along seq dim → [batch, generated_length, vocab]
            return out, torch.cat(all_logits, dim=1)
        
        return out, None