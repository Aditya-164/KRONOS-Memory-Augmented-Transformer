"""Transformer implementation for the KRONOS model with Coconut (Chain of Continuous Thought)
   and integrated Dual Memory architecture (NeuralMemory and PersistentMemory)."""

import warnings
import torch
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List, Callable, Union
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

from src.models.persistent_memory import (
    PersistentMemory,
    PersistentMemoryConfig,
    PersistentMemoryState,
    BucketCreationPolicy,
    BucketStorageStrategy
)


@dataclass
class TransformerState:
    """Updated state of the transformer model for stateful execution with dual memory systems."""
    memory_states: List[NeuralMemoryState]
    kv_caches: List[Tuple[torch.Tensor, torch.Tensor]]
    value_residuals: List[torch.Tensor]
    # Track latent mode state
    in_latent_mode: bool = False
    last_hidden_state: Optional[torch.Tensor] = None
    persistent_memory_states: Optional[List[PersistentMemoryState]] = None  # Added persistent memory states

# Define a type for block state with dual memory systems
DualMemoryBlockState = Tuple[
    Optional[NeuralMemoryState],    # neural_memory_state
    Optional[PersistentMemoryState], # persistent_memory_state
    Optional[Tuple[torch.Tensor, torch.Tensor]], # kv_cache
    Optional[torch.Tensor]           # value_residual
]
class TransformerBlock(nn.Module):
    """Enhanced transformer block with segmented attention and dual memory architecture."""
    
    def __init__(
        self,
        dim: int,
        dim_head: int = 64,
        heads: int = 8,
        segment_len: int = 512,
        ff_mult: int = 4,
        memory_dim: Optional[int] = None,
        chunk_size: Optional[int] = None,
        use_neural_memory: bool = True,
        use_persistent_memory: bool = False,
        neural_memory_kwargs: Dict[str, Any] = None,
        persistent_memory_kwargs: Dict[str, Any] = None,
        use_flex_attn: bool = False,
        accept_value_residual: bool = False,
        num_longterm_mem_tokens: int = 0,
        num_persist_mem_tokens: int = 0,
        sliding: bool = False,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        cross_segment_fusion: bool = True,  
        adaptive_forgetting: bool = True,   
        layer_gate: bool = True,         
    ):
        super().__init__()
        self.dim = dim
        self.use_neural_memory = use_neural_memory
        self.use_persistent_memory = use_persistent_memory
        self.segment_len = segment_len
        self.layer_gate = layer_gate
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)  # For attention
        self.norm2 = nn.LayerNorm(dim)  # For FF
        
        # Memory-specific normalization
        self.norm_nm = nn.LayerNorm(dim) if use_neural_memory else None
        self.norm_pm = nn.LayerNorm(dim) if use_persistent_memory else None
        
        # Common normalization for memory input (optional)
        self.norm_mem_input = nn.LayerNorm(dim) if use_neural_memory or use_persistent_memory else None
        
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
        
        # Neural Memory module
        self.neural_memory = None
        if use_neural_memory:
            nm_kwargs = default(neural_memory_kwargs, {})
            chunk_size = default(chunk_size, segment_len)
            memory_dim = default(memory_dim, dim)
            
            # Add parameter configurations for NeuralMemory initialization
            if "adaptive_forgetting" not in nm_kwargs:
                nm_kwargs["adaptive_forgetting"] = adaptive_forgetting
            
            # Add other parameters that were previously passed to forward() method
            if "use_enhanced_surprise" not in nm_kwargs:
                nm_kwargs["use_enhanced_surprise"] = True
            
            if "use_adaptive_surprise_threshold" not in nm_kwargs:
                nm_kwargs["use_adaptive_surprise_threshold"] = True
            
            self.neural_memory = NeuralMemory(
                dim=dim,
                chunk_size=chunk_size,
                memory_dim=memory_dim,
                **nm_kwargs
            )
        
        # Persistent Memory module
        self.persistent_memory = None
        if use_persistent_memory:
            pm_kwargs = default(persistent_memory_kwargs, {})
            pm_concept_dim = pm_kwargs.get('concept_embedding_dim', default(memory_dim, dim))
            pm_value_dim = pm_kwargs.get('value_dim', default(memory_dim, dim))
            
            pm_config = PersistentMemoryConfig(
                input_dim=dim,
                concept_embedding_dim=pm_concept_dim,
                value_dim=pm_value_dim,
                bucket_creation_policy=pm_kwargs.get('bucket_creation_policy', BucketCreationPolicy.THRESHOLD),
                storage_strategy=pm_kwargs.get('bucket_storage_strategy', BucketStorageStrategy.FIFO),
                max_buckets=pm_kwargs.get('max_buckets', 500),
                new_concept_similarity_threshold=pm_kwargs.get('new_concept_similarity_threshold', 0.85),
                # Remove the unsupported parameters
                max_items_per_bucket=pm_kwargs.get('max_items_per_bucket', 64),
                use_layernorm=pm_kwargs.get('use_layernorm', True),
                dropout_rate=pm_kwargs.get('dropout_rate', 0.1),
            )
            
            initial_core_concepts = pm_kwargs.get('initial_core_concept_embeddings', None)
            self.persistent_memory = PersistentMemory(
                config=pm_config,
                initial_core_concept_embeddings=initial_core_concepts
            )
        
        # Layer gating mechanism for dynamic control of information flow
        if layer_gate:
            # Determine number of gates needed: attention, FF, and each enabled memory system
            num_gates = 2  # Attention and FF base gates
            if use_neural_memory:
                num_gates += 1
            if use_persistent_memory:
                num_gates += 1
                
            self.gate_net = nn.Sequential(
                nn.Linear(dim, dim // 4),
                nn.SiLU(),
                nn.Linear(dim // 4, num_gates),
                nn.Sigmoid()
            )
    
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[DualMemoryBlockState] = None,
        disable_flex_attn: bool = False,
        flex_attn_fn: Optional[Callable] = None,
        output_gating: Optional[torch.Tensor] = None,
        input_self_information: Optional[torch.Tensor] = None,
        pm_retrieval_strategy: Optional[str] = None,
        pm_retrieval_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, DualMemoryBlockState]:
        """Forward pass with enhanced dual memory architecture.
        
        Args:
            x: Input tensor [batch, seq_len, dim]
            state: Previous state tuple (nm_state, pm_state, kv_cache, value_residual)
            disable_flex_attn: Whether to disable flex attention
            flex_attn_fn: Custom attention function
            output_gating: Output gating tensor
            input_self_information: Self-information metrics for neural memory [batch, seq_len]
            pm_retrieval_strategy: Strategy for persistent memory retrieval
            pm_retrieval_params: Parameters for persistent memory retrieval
            
        Returns:
            output: Transformed output [batch, seq_len, dim]
            state: Updated state tuple (nm_state, pm_state, kv_cache, value_residual)
        """
        neural_mem_state, persistent_mem_state, kv_cache, value_residual = None, None, None, None
        if exists(state):
            neural_mem_state, persistent_mem_state, kv_cache, value_residual = state
        
        # --- Gating ---
        gates = None
        attn_gate, ff_gate, nm_gate, pm_gate = None, None, None, None
        if self.layer_gate:
            # Calculate gates from input
            gates_input = x.mean(dim=1, keepdim=True) if x.ndim == 3 and x.shape[1] > 0 else x.mean(dim=0, keepdim=True) if x.ndim == 2 else x
            
            all_gates = self.gate_net(gates_input)
            current_gate_idx = 0
            
            # Assign gates based on enabled components
            attn_gate = all_gates[..., current_gate_idx:current_gate_idx+1]; current_gate_idx += 1
            ff_gate = all_gates[..., current_gate_idx:current_gate_idx+1]; current_gate_idx += 1
            
            if self.use_neural_memory:
                nm_gate = all_gates[..., current_gate_idx:current_gate_idx+1]; current_gate_idx += 1
                
            if self.use_persistent_memory:
                pm_gate = all_gates[..., current_gate_idx:current_gate_idx+1]; current_gate_idx += 1
        
        # --- Self-Attention ---
        residual = x
        x_attn = self.norm1(x)
        
        # Fix parameter name: return_attn_score -> return_attn_scores
        attn_output, (new_value_residual, new_kv_cache), attn_score = self.attention(
            x_attn,
            value_residual=value_residual,
            disable_flex_attn=disable_flex_attn,
            flex_attn_fn=flex_attn_fn,
            output_gating=output_gating,
            cache=kv_cache,
            return_attn_scores=True
        )
        
        # Apply attention gate if enabled
        if self.layer_gate and exists(attn_gate):
            attn_output = attn_output * attn_gate
        
        x = residual + attn_output
        
        # --- Feed-forward network ---
        residual = x
        x_ff = self.norm2(x)
        ff_output = self.ff(x_ff)
        
        # Apply FF gate if enabled
        if self.layer_gate and exists(ff_gate):
            ff_output = ff_output * ff_gate
            
        x = residual + ff_output
        
        # --- Memory Modules ---
        new_neural_mem_state, new_persistent_mem_state = neural_mem_state, persistent_mem_state
        
        if self.use_neural_memory or self.use_persistent_memory:
            # Common normalized input for memory modules
            x_mem_input = self.norm_mem_input(x) if exists(self.norm_mem_input) else x
            memory_contributions = torch.zeros_like(x)
            
            # -- Neural Memory --
            if self.use_neural_memory and exists(self.neural_memory):
                nm_input = self.norm_nm(x_mem_input) if exists(self.norm_nm) else x_mem_input
                mem_attention_probs = attn_score.mean(dim=1) if exists(attn_score) else None
                
                # Fix: Only pass parameters that NeuralMemory.forward() actually accepts
                nm_output, new_neural_mem_state = self.neural_memory(
                    nm_input,
                    state=neural_mem_state,
                    prev_weights=None if not exists(neural_mem_state) else neural_mem_state.updates,
                    attention_probs=mem_attention_probs,
                    input_self_information=input_self_information
                )
                
                # Apply memory gate if enabled
                if self.layer_gate and exists(nm_gate):
                    nm_output = nm_output * nm_gate
                    
                memory_contributions = memory_contributions + nm_output
            
            # -- Persistent Memory --
            if self.use_persistent_memory and exists(self.persistent_memory):
                pm_input = self.norm_pm(x_mem_input) if exists(self.norm_pm) else x_mem_input
                
                pm_output, new_persistent_mem_state = self.persistent_memory(
                    pm_input,
                    state=persistent_mem_state,
                    retrieval_strategy=pm_retrieval_strategy,
                    retrieval_params=pm_retrieval_params
                )
                
                # Apply memory gate if enabled
                if self.layer_gate and exists(pm_gate):
                    pm_output = pm_output * pm_gate
                    
                memory_contributions = memory_contributions + pm_output
            
            # Add combined memory contributions
            x = x + memory_contributions
        
        return x, (new_neural_mem_state, new_persistent_mem_state, new_kv_cache, new_value_residual)

class KRONOSTransformer(nn.Module):
    """Titans Transformer model with segmented attention, dual memory architecture, and Coconut."""
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
        # Updated memory parameters for dual memory architecture
        use_neural_memory: bool = True,
        use_persistent_memory: bool = False,
        neural_memory_kwargs: Dict[str, Any] = None,
        persistent_memory_kwargs: Dict[str, Any] = None,
        use_flex_attn: bool = False,
        accept_value_residual: bool = False,
        num_longterm_mem_tokens: int = 0,
        num_persist_mem_tokens: int = 0,
        sliding: bool = False,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        emb_dropout: float = 0.0,
        tie_word_embeddings: bool = False,
        use_axial_pos_emb: bool = True,
        num_axial_dims: int = 2,
        # New parameters
        cross_segment_fusion: bool = True,
        adaptive_forgetting: bool = True,
        layer_gate: bool = True,
        # Coconut parameters
        use_continuous_thought: bool = True,
        bot_token_id: int = None,  # Beginning of thought token ID
        eot_token_id: int = None,  # End of thought token ID
    ):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.tie_word_embeddings = tie_word_embeddings
        
        # Coconut parameters
        self.use_continuous_thought = use_continuous_thought
        self.bot_token_id = bot_token_id  # Beginning of thought
        self.eot_token_id = eot_token_id  # End of thought

        # Token embedding
        self.token_emb = nn.Embedding(vocab_size, dim)

        # Positional embedding
        if use_axial_pos_emb:
            self.pos_emb = ContinuousAxialPositionalEmbedding(dim, num_axial_dims=num_axial_dims)
        else:
            self.pos_emb = nn.Embedding(seq_len, dim)

        self.emb_dropout = nn.Dropout(emb_dropout)

        # Transformer blocks with dual memory architecture
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                dim_head=dim_head,
                heads=heads,
                segment_len=segment_len,
                ff_mult=ff_mult,
                memory_dim=memory_dim,
                chunk_size=chunk_size,
                use_neural_memory=use_neural_memory,
                use_persistent_memory=use_persistent_memory,
                neural_memory_kwargs=neural_memory_kwargs,
                persistent_memory_kwargs=persistent_memory_kwargs,
                use_flex_attn=use_flex_attn,
                accept_value_residual=accept_value_residual,
                num_longterm_mem_tokens=num_longterm_mem_tokens,
                num_persist_mem_tokens=num_persist_mem_tokens,
                sliding=sliding,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                cross_segment_fusion=cross_segment_fusion,
                adaptive_forgetting=adaptive_forgetting,
                layer_gate=layer_gate
            ) for _ in range(depth)]
        )

        # Normalization & output
        self.norm = nn.LayerNorm(dim)
        if tie_word_embeddings:
            self.to_logits = lambda x: x @ self.token_emb.weight.t()
        else:
            self.to_logits = nn.Linear(dim, vocab_size)

    def add_special_tokens(self, bot_token_id: int, eot_token_id: int):
        """Set special token IDs for continuous thought."""
        self.bot_token_id = bot_token_id
        self.eot_token_id = eot_token_id

    def forward_with_state(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        state: Optional[TransformerState] = None,
        return_state: bool = False,
        disable_flex_attn: bool = False,
        flex_attn_fn: Optional[Callable] = None,
        input_self_information: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[TransformerState]]:
        batch_size, seq_len = x.shape
        device = x.device

        # Unpack prior state or create a new one
        if state is not None:
            memory_states = state.memory_states
            persistent_memory_states = state.persistent_memory_states
            kv_caches = state.kv_caches
            value_resids = state.value_residuals
            in_latent_mode = state.in_latent_mode
            last_hidden_state = state.last_hidden_state
        else:
            memory_states = None
            persistent_memory_states = None
            kv_caches = None
            value_resids = None
            in_latent_mode = False
            last_hidden_state = None

        # Check for continuous thought mode transitions
        if self.use_continuous_thought and self.bot_token_id is not None and self.eot_token_id is not None:
            # Detect Beginning of Thought token
            if not in_latent_mode and seq_len == 1 and x[0, 0].item() == self.bot_token_id:
                in_latent_mode = True
                
            # Detect End of Thought token
            elif in_latent_mode and seq_len == 1 and x[0, 0].item() == self.eot_token_id:
                in_latent_mode = False
                last_hidden_state = None

        # Token embeddings
        if in_latent_mode and last_hidden_state is not None and seq_len == 1:
            # In latent mode: use last hidden state as embedding
            x = last_hidden_state.unsqueeze(1)  # [batch, 1, dim]
        else:
            # In language mode: use token embeddings
            x = self.token_emb(x)  # [batch, seq_len, dim]

        # Positional embeddings
        if isinstance(self.pos_emb, ContinuousAxialPositionalEmbedding):  
            if positions is not None:
                max_abs_idx_in_batch = positions.max().item()
                required_pe_table_len = max_abs_idx_in_batch + 1 # the length L must be at least max_abs_idx_in_batch + 1
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

        # Skip adding positional embeddings in latent mode for continuous thought
        if not (in_latent_mode and last_hidden_state is not None and seq_len == 1):
            x = x + pos_emb
            
        x = self.emb_dropout(x)

        # Pass through layers
        new_neural_mem_states, new_persistent_mem_states, new_kv_caches, new_val_resids = [], [], [], []
        for i, layer in enumerate(self.layers):
            prev_neural_mem = None
            prev_persistent_mem = None
            prev_kv = None
            prev_val = None
            
            if memory_states is not None:
                prev_neural_mem = memory_states[i]
            if persistent_memory_states is not None:
                prev_persistent_mem = persistent_memory_states[i]
            if kv_caches is not None:
                prev_kv = kv_caches[i]
            if value_resids is not None:
                prev_val = value_resids[i]

            x, (nm, pm, k, v) = layer(
                x,
                state=(prev_neural_mem, prev_persistent_mem, prev_kv, prev_val),
                disable_flex_attn=disable_flex_attn,
                flex_attn_fn=flex_attn_fn,
                input_self_information=input_self_information,
            )
            new_neural_mem_states.append(nm)
            new_persistent_mem_states.append(pm)
            new_kv_caches.append(k)
            new_val_resids.append(v)

        # Apply final normalization
        x = self.norm(x)
        
        # Store the hidden state for Coconut continuous thought
        if in_latent_mode:
            last_hidden_state = x[:, -1].detach().clone()  # [batch, dim]
            
        # Convert to logits (not needed in latent mode for intermediate steps)
        if not in_latent_mode or (seq_len == 1 and x[0,0].item() != self.bot_token_id):
            logits = self.to_logits(x)
        else:
            # Dummy logits during latent mode
            logits = torch.zeros((batch_size, seq_len, self.vocab_size), device=device)

        if return_state:
            new_state = TransformerState(
                memory_states=new_neural_mem_states,
                persistent_memory_states=new_persistent_mem_states,
                kv_caches=new_kv_caches,
                value_residuals=new_val_resids,
                in_latent_mode=in_latent_mode,
                last_hidden_state=last_hidden_state
            )
            return logits, new_state

        return logits

    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        input_self_information: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward_with_state(x, positions=positions, input_self_information=input_self_information, return_state=False)

class KRONOSLM(nn.Module):
    """Titans Language Model for text generation with KV cache, dual memory architecture, and Coconut continuous thought."""
    
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
        state: Optional[TransformerState] = None,
        return_state: bool = False,
        input_self_information: Optional[torch.Tensor] = None,
        **kwargs  # swallow any extra training args
    ) -> Tuple[torch.Tensor, Optional[TransformerState]]:
        # call the core transformer
        out = self.transformer.forward_with_state(
            x, positions=positions, state=state, return_state=return_state
        )
        # always normalize to (logits, state)
        if not isinstance(out, tuple):
            return out, None
        logits = out[0]
        new_state = out[1] if len(out) > 1 else None
        return logits, new_state