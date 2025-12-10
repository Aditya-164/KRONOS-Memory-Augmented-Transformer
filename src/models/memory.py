"""Neural Long-Term Memory Module with weight decay, momentum gate transitions, 
gradient norm clamping, and soft clamping mechanisms.Includes optional Persistent Memory"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple
from einops import rearrange, repeat

from .utils import exists, default

@dataclass
class NeuralMemoryState:
    """State of the neural memory module."""
    memory: torch.Tensor  # Current memory state
    momentum: torch.Tensor  # Momentum term for memory updates
    updates: Optional[torch.Tensor] = None  # Weight updates for memory

class NeuralMemory(nn.Module):
    """Neural Long-Term Memory module"""
    def __init__(
        self,
        dim: int,
        chunk_size: int,
        hidden_dim: Optional[int] = None,
        key_dim: Optional[int] = None,
        value_dim: Optional[int] = None,
        memory_dim: Optional[int] = None, 
        model: Optional[nn.Module] = None,
        momentum_decay: float = 0.9,
        forgetting_factor: float = 0.1,
        learning_rate: float = 0.1,
        weight_decay: float = 1e-4,
        gradient_clamp: Optional[float] = None,
        soft_clamp_threshold: Optional[float] = None,
        accept_weight_residual: bool = False,    
        adaptive_forgetting: bool = False,
        use_enhanced_surprise: bool = False,
        surprise_alpha_weights: Optional[Tuple[float, float, float, float]] = None, # Weights for [PredError, AttEntropy, GradDiversity, SelfInfo]
        content_importance_threshold: float = 0.5,
        use_adaptive_surprise_threshold: bool = False,
        surprise_beta_sensitivity: float = 1.0,  # The 'β' parameter
        surprise_ema_decay: float = 0.99,       # For updating running stats
        surprise_floor: float = 0.1,            # Minimum modulation factor if thresholding
        surprise_cap: float = 2.0,              # Maximum modulation factor (optional)
        surprise_modulates_update: bool = True, # If true, surprise score scales w_upd
        surprise_gates_update: bool = False,    # If true, surprise score (vs threshold) gates update
    ):
        super().__init__()
        hidden_dim = default(hidden_dim, dim)
        key_dim    = default(key_dim, dim)
        value_dim  = default(value_dim, dim)
        memory_dim = default(memory_dim, dim)

        # Projections
        self.to_key   = nn.Linear(dim, key_dim)
        self.to_value = nn.Linear(dim, value_dim)
        self.to_query = nn.Linear(dim, key_dim)
        self.to_out   = nn.Linear(value_dim, dim)

        # Internal model mapping (key + memory -> value)
        self.model = model if exists(model) else nn.Sequential(
            nn.Linear(key_dim + memory_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, value_dim)
        )

        # Hyperparameters
        self.eta            = momentum_decay
        self.alpha          = forgetting_factor
        self.theta          = learning_rate
        self.weight_decay   = weight_decay
        self.gradient_clamp = gradient_clamp
        self.soft_clamp_threshold = soft_clamp_threshold
        self.accept_weight_residual = accept_weight_residual
        self.adaptive_forgetting     = adaptive_forgetting
        self.content_importance_threshold = content_importance_threshold

        # Adaptive forgetting network
        if self.adaptive_forgetting:
            self.importance_net = nn.Sequential(
                nn.Linear(memory_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )

        # Momentum gating
        self.momentum_gate_net = nn.Sequential(
            nn.Linear(key_dim, memory_dim),
            nn.Sigmoid()
        )

        # Initial states
        self.register_buffer('initial_memory', torch.zeros(1, memory_dim))
        self.register_buffer('initial_momentum', torch.zeros(1, memory_dim))

        # Projection from value updates to memory space
        self.update_proj = nn.Linear(value_dim, memory_dim)

        # Enhanced suprise initilization
        self.use_enhanced_surprise = use_enhanced_surprise
        if self.use_enhanced_surprise:
            self.surprise_alpha_weights = default(surprise_alpha_weights, (1.0, 0.0, 0.0, 0.0)) # Default to only PredError
            self.use_adaptive_surprise_threshold = use_adaptive_surprise_threshold
            self.surprise_beta_sensitivity = surprise_beta_sensitivity
            self.surprise_ema_decay = surprise_ema_decay
            self.surprise_floor = surprise_floor
            self.surprise_cap = surprise_cap # New
            self.surprise_modulates_update = surprise_modulates_update
            self.surprise_gates_update = surprise_gates_update

            if len(self.surprise_alpha_weights) != 4:
                raise ValueError("surprise_alpha_weights must be a tuple of 4 floats.")

            if self.use_adaptive_surprise_threshold:
                self.register_buffer('running_surprise_mean', torch.zeros(1))
                self.register_buffer('running_surprise_std', torch.ones(1)) # Initialize std to 1 to avoid division by zero early on
            
            # Ensure only one gating/modulation strategy is primary if both flags true
            if self.surprise_modulates_update and self.surprise_gates_update:
                print("Warning: Both surprise_modulates_update and surprise_gates_update are True. Modulation will be applied first, then gating.")

    def soft_clamp(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        return threshold * torch.tanh(x / threshold)

    def _compute_enhanced_surprise(
        self,
        predicted_values: torch.Tensor, # (b*n, d_v) from internal model
        actual_values: torch.Tensor,    # (b*n, d_v) flat_v
        grad_mem: torch.Tensor,         # (b, d_mem) gradient w.r.t memory
        attention_probs: Optional[torch.Tensor] = None, # (b, num_heads, seq, seq) or similar, needs processing
        input_self_information: Optional[torch.Tensor] = None # (b,) or (b,n) pre-calculated -logP(x_t)
    ) -> torch.Tensor:
        """
        Computes the multi-dimensional surprise metric.
        All components should be reduced to a per-batch-item scalar before weighted sum.
        """
        b = grad_mem.shape[0]
        device = grad_mem.device
        surprise_components = []

        # 1. Prediction Error Magnitude (L2 norm between predicted and actual outputs)
        # Reshape to (b, n, d_v) to average over sequence length n for a per-batch item score
        if self.surprise_alpha_weights[0] > 0:
            # Assuming predicted_values and actual_values are (B*N, D_value)
            # We want a per-batch item error, so we average over N (chunk_size)
            num_elements_in_chunk = predicted_values.shape[0] // b
            pred_error_per_element = F.mse_loss(predicted_values, actual_values, reduction='none').sum(dim=-1) # (B*N)
            pred_error_per_batch_item = rearrange(pred_error_per_element, '(b n) -> b n', b=b).mean(dim=1) # (B,)
            # Normalize or scale if necessary, L2 norm of error vector might be too large
            # Alternative: use the `loss_sig` directly if it's representative
            pred_err_val = pred_error_per_batch_item
        else:
            # Create a zero tensor with the right shape and device
            pred_err_val = torch.zeros(b, device=device)
        surprise_components.append(pred_err_val * self.surprise_alpha_weights[0])

        # 2. Attention Entropy
        if self.surprise_alpha_weights[1] > 0 and exists(attention_probs):
            # attention_probs might be (B, H, N, N) from a self-attention layer.
            # We need to decide how to aggregate this. Example: mean entropy over heads and positions.
            # This is highly dependent on how attention_probs are structured and passed.
            # Assuming attention_probs are (B, N_att) representing some salient attention distributions
            # For simplicity, let's assume it's already processed to (B,) or (B, N) then averaged.
            # This part requires careful thought on what attention_probs represents.
            # SegmentedAttention -> the entropy of attention over memory tokens, for example.
            if attention_probs.ndim == 3: # (B, num_dist, prob_dim)
                avg_entropy = self._calculate_entropy(attention_probs).mean(dim=1) # (B,)
            elif attention_probs.ndim == 2 and attention_probs.shape[-1] > 1: # (B, prob_dim) - single distribution per batch item
                avg_entropy = self._calculate_entropy(attention_probs) # (B,)
            elif attention_probs.ndim == 1: # Already (B,) precomputed
                avg_entropy = attention_probs
            else: # Default to 0 if format is not recognized
                avg_entropy = torch.zeros(b, device=device)
            att_entropy_val = avg_entropy
        else:
            # Create a zero tensor with the right shape and device
            att_entropy_val = torch.zeros(b, device=device)
        surprise_components.append(att_entropy_val * self.surprise_alpha_weights[1])

        # 3. Gradient Diversity (e.g., variance or entropy of gradient elements)
        if self.surprise_alpha_weights[2] > 0:
            # grad_mem is (B, D_mem)
            # Variance of gradient elements for each batch item
            # grad_diversity = torch.var(grad_mem, dim=-1) # (B,)
            # Alternative: Entropy of normalized absolute gradient values (treating them as a distribution)
            abs_grad_mem = torch.abs(grad_mem)
            dist_grad_mem = abs_grad_mem / (torch.sum(abs_grad_mem, dim=-1, keepdim=True) + 1e-12) # (B, D_mem)
            grad_diversity = self._calculate_entropy(dist_grad_mem) # (B,)
            grad_diversity_val = grad_diversity
        else:
            # Create a zero tensor with the right shape and device
            grad_diversity_val = torch.zeros(b, device=device)
        surprise_components.append(grad_diversity_val * self.surprise_alpha_weights[2])

        # 4. Self-Information (-log P(x_t))
        if self.surprise_alpha_weights[3] > 0 and exists(input_self_information):
            # input_self_information should be (B,) or (B,N) then averaged
            if input_self_information.ndim == 2: # (B,N)
                self_info_val = input_self_information.mean(dim=1) # (B,)
            else: # Assuming (B,)
                self_info_val = input_self_information
        else:
            # Create a zero tensor with the right shape and device
            self_info_val = torch.zeros(b, device=device)
        surprise_components.append(self_info_val * self.surprise_alpha_weights[3])
        
        # Ensure all components are tensors with shape (B,)
        # Stack and sum the components
        total_surprise = torch.stack(surprise_components, dim=0).sum(dim=0) # (B,)

        # alphas handle scaling.
        return total_surprise # (B,)

    def _compute_content_importance(self, memory: torch.Tensor) -> torch.Tensor:
        if not self.adaptive_forgetting:
            return torch.ones(memory.size(0), 1, device=memory.device)
        return self.importance_net(memory)

    def _compute_loss_signal(
        self,
        memory: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor
    ) -> torch.Tensor:
        b, n, _ = keys.shape
        flat_k = rearrange(keys, 'b n d -> (b n) d')
        flat_v = rearrange(values, 'b n d -> (b n) d')
        expanded_mem = repeat(memory, 'b d -> (b n) d', n=n)
        inp = torch.cat([flat_k, expanded_mem], dim=-1)
        pred = self.model(inp)
        mse = F.mse_loss(pred, flat_v, reduction='none')
        p_norm = F.normalize(pred, dim=-1)
        v_norm = F.normalize(flat_v, dim=-1)
        cos_dist = (1 - (p_norm * v_norm).sum(dim=-1, keepdim=True)) * 0.5
        loss = mse + cos_dist
        loss = rearrange(loss, '(b n) d -> b n d', b=b)
        weights = F.softmax(loss.mean(dim=-1, keepdim=True), dim=1)
        return (loss * weights).sum(dim=1)

    def _update_memory(
        self,
        memory: torch.Tensor,
        momentum: torch.Tensor,
        keys: torch.Tensor,    # (b, n, d_k)
        values: torch.Tensor,  # (b, n, d_v)
        prev_weights: Optional[torch.Tensor],
        # --- New arguments for surprise ---
        attention_probs: Optional[torch.Tensor] = None,
        input_self_information: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mem_orig_device = memory.device
        # Ensure any prev_weights is on the same device as memory
        if exists(prev_weights):
            prev_weights = prev_weights.to(mem_orig_device)

        # Detach the memory before using it in internal computations
        # This prevents incorrect gradient flow during the main model backward pass
        mem = memory.detach().clone().requires_grad_(True)
        
        # --- Original loss computation and gradient ---
        b_n, d_k_val = keys.shape[0] * keys.shape[1], values.shape[-1]
        flat_k_for_pred = rearrange(keys, 'b n d -> (b n) d')
        expanded_mem_for_pred = repeat(mem, 'b d -> (b n) d', n=keys.shape[1])
        inp_for_pred = torch.cat([flat_k_for_pred, expanded_mem_for_pred], dim=-1)
        predicted_values_internal = self.model(inp_for_pred)  # (b*n, d_v) - used for PredError
        flat_v_for_pred = rearrange(values, 'b n d -> (b n) d')  # (b*n, d_v) - used for PredError

        loss_sig = self._compute_loss_signal(mem, keys, values)
        
        # Use create_graph=True when computing gradients used within forward pass
        # This ensures proper handling of higher-order gradients during backprop
        loss_sum_for_grad = self._compute_loss_signal(mem, keys, values).sum()
        grad_mem = torch.autograd.grad(
            loss_sum_for_grad, 
            mem, 
            create_graph=True,  # Enable higher-order gradients
            retain_graph=True   # Keep the graph for subsequent backward passes
        )[0]

        # --- Enhanced Surprise Calculation and Application ---
        surprise_modulation_factor = torch.ones(memory.shape[0], 1, device=mem_orig_device)  # (B, 1)
        if self.use_enhanced_surprise:
            current_surprise = self._compute_enhanced_surprise(
                predicted_values_internal.to(mem_orig_device),
                flat_v_for_pred.to(mem_orig_device),
                grad_mem.to(mem_orig_device),
                attention_probs.to(mem_orig_device) if exists(attention_probs) else None,
                input_self_information.to(mem_orig_device) if exists(input_self_information) else None
            ).unsqueeze(-1)

            if self.use_adaptive_surprise_threshold:
                # Copy buffers to avoid in-place operations
                running_surprise_mean = self.running_surprise_mean.clone().to(current_surprise.device)
                running_surprise_std = self.running_surprise_std.clone().to(current_surprise.device)

                batch_mean = current_surprise.mean()
                batch_std = current_surprise.std() if current_surprise.shape[0] > 1 else torch.tensor(1.0, device=current_surprise.device)

                # Avoid in-place update by using proper assignment
                new_mean = self.surprise_ema_decay * running_surprise_mean + (1 - self.surprise_ema_decay) * batch_mean
                new_std = self.surprise_ema_decay * running_surprise_std + (1 - self.surprise_ema_decay) * batch_std
                
                # Update buffers after computation
                with torch.no_grad():
                    self.running_surprise_mean.copy_(new_mean.detach())
                    self.running_surprise_std.copy_(new_std.detach())

                surprise_threshold = running_surprise_mean + self.surprise_beta_sensitivity * running_surprise_std

                if self.surprise_gates_update:
                    surprise_modulation_factor = torch.where(
                        current_surprise > surprise_threshold,
                        torch.ones_like(current_surprise),
                        torch.full_like(current_surprise, self.surprise_floor)
                    )
                elif self.surprise_modulates_update:
                    scaled_surprise_diff = (current_surprise - surprise_threshold) / (running_surprise_std + 1e-6)
                    surprise_modulation_factor = torch.sigmoid(scaled_surprise_diff)
                    surprise_modulation_factor = self.surprise_floor + (self.surprise_cap - self.surprise_floor) * surprise_modulation_factor
            elif self.surprise_modulates_update:
                surprise_modulation_factor = torch.clamp(current_surprise, min=self.surprise_floor, max=self.surprise_cap)

        # --- Original update logic, now modulated by surprise ---
        # Avoid in-place operations by creating new tensors
        w_upd = -self.theta * grad_mem.clone()  # Use clone instead of modifying in-place

        if self.use_enhanced_surprise and (self.surprise_modulates_update or self.surprise_gates_update):
            w_upd = w_upd * surprise_modulation_factor  # This is fine as it creates a new tensor

        if exists(prev_weights) and self.accept_weight_residual:
            w_upd = w_upd + prev_weights  # This is fine as it creates a new tensor

        if self.gradient_clamp is not None:
            norm = w_upd.norm(dim=-1, keepdim=True)
            factor = (self.gradient_clamp / (norm + 1e-6)).clamp(max=1.0)
            w_upd = w_upd * factor  # This is fine as it creates a new tensor

        w_upd = self.update_proj(w_upd)

        # Momentum gating
        agg_key = keys.mean(dim=1)
        gate = self.momentum_gate_net(agg_key)
        # Avoid in-place ops - create new tensors at each step
        scaled_momentum = self.eta * momentum.clone()
        new_mom = gate * scaled_momentum + (1 - gate) * w_upd  # This creates a new tensor

        # Update memory - avoid in-place operations
        if self.adaptive_forgetting:
            imp = self._compute_content_importance(memory)
            alpha_forget = self.alpha * (1 - imp).clamp(0, 1) + 0.01
            scaled_memory = (1 - alpha_forget) * memory.clone()
            weight_decay_term = self.weight_decay * memory.clone()
            new_mem = scaled_memory + new_mom - weight_decay_term  # Creates a new tensor
        else:
            scaled_memory = (1 - self.alpha) * memory.clone()
            weight_decay_term = self.weight_decay * memory.clone()
            new_mem = scaled_memory + new_mom - weight_decay_term  # Creates a new tensor

        if self.soft_clamp_threshold is not None:
            new_mem = self.soft_clamp(new_mem, self.soft_clamp_threshold)

        # Ensure final results are detached to avoid leaking gradients incorrectly
        # Return new tensors, not modified versions of input tensors
        return new_mem.detach().to(mem_orig_device), new_mom.detach().to(mem_orig_device), w_upd.detach().to(mem_orig_device)

    def _retrieve_from_memory(
        self,
        memory: torch.Tensor,
        queries: torch.Tensor
    ) -> torch.Tensor:
        b, n, _ = queries.shape
        flat_q = rearrange(queries, 'b n d -> (b n) d')
        exp_mem = repeat(memory, 'b d -> (b n) d', n=n)
        inp = torch.cat([flat_q, exp_mem], dim=-1)
        ret = self.model(inp)
        return rearrange(ret, '(b n) d -> b n d', b=b)

    def forward(
        self,
        x: torch.Tensor, # Input to the memory module (e.g., hidden states from Transformer)
        state: Optional[NeuralMemoryState] = None,
        prev_weights: Optional[torch.Tensor] = None,
        # --- New optional inputs for surprise ---
        attention_probs: Optional[torch.Tensor] = None,
        input_self_information: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, NeuralMemoryState]:
        b, _, _ = x.shape
        keys    = self.to_key(x)
        values  = self.to_value(x)
        queries = self.to_query(x)
        if state is None:
            mem = repeat(self.initial_memory, '1 d -> b d', b=b)
            mom = repeat(self.initial_momentum, '1 d -> b d', b=b)
        else:
            mem = state.memory
            mom = state.momentum
        new_mem, new_mom, w_upd = self._update_memory(
            mem, mom, keys, values, prev_weights,
            attention_probs=attention_probs if self.use_enhanced_surprise else None,
            input_self_information=input_self_information if self.use_enhanced_surprise else None
        )
        retrieved = self._retrieve_from_memory(new_mem, queries)
        out = self.to_out(retrieved)
        new_state = NeuralMemoryState(memory=new_mem, momentum=new_mom, updates=w_upd)
        return out, new_state
