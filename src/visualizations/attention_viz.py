"""Attention visualization module for Titans transformer.

This module provides functions to visualize attention patterns and memory
activations during text generation, helping to understand how the model processes
information and utilizes its memory components.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation
from pathlib import Path
import os
from tqdm import tqdm
import seaborn as sns
from src.models import TransformerState, TitansLM
from src.utils import get_logger

# Setup logger
logger = get_logger(__name__)

# Define custom colormaps for attention visualization
ATTENTION_COLORMAP = sns.color_palette("viridis", as_cmap=True)
MEMORY_COLORMAP = sns.color_palette("plasma", as_cmap=True)


class AttentionVisualizer:
    """Visualize attention patterns and memory activations during text generation."""
    
    def __init__(
        self,
        model: TitansLM,
        tokenizer: Any,
        output_dir: str = "./attention_viz",
        max_len: int = 50,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize the attention visualizer.
        
        Args:
            model: The TitansLM model to visualize
            tokenizer: Tokenizer to decode tokens
            output_dir: Directory to save visualizations
            max_len: Maximum sequence length for visualization
            device: Device to run the model on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_len = max_len
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Store generation history
        self.token_history = []
        self.state_history = []
        self.attention_history = []
        self.memory_history = []
        
        # For tracking frame data when generating animations
        self.frames = []
        
        logger.info(f"AttentionVisualizer initialized with model on {self.device}")
    
    def generate_with_history(
        self, 
        prompt: str, 
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        store_attention: bool = True,
    ) -> Tuple[str, List[int]]:
        """Generate text while capturing attention and memory states.
        
        Args:
            prompt: Text prompt
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            store_attention: Whether to store attention for visualization
            
        Returns:
            generated_text: Full generated text (prompt + continuation)
            token_ids: List of token IDs
        """
        self.token_history = []
        self.state_history = []
        self.attention_history = []
        self.memory_history = []
        
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        input_len = input_ids.size(1)
        
        # Store initial tokens in history
        self.token_history = input_ids.tolist()[0]
        
        # Initialize state
        state = None
        
        # For storing current sequence
        out = input_ids
        
        # Record attention and memory for initial prompt
        if store_attention:
            # Process the prompt
            _, state = self.model.transformer.forward_with_state(
                out, state=None, return_state=True
            )
            self.state_history.append(state)
            
            # Record attention patterns and memory for the prompt
            self._record_attention_and_memory(state)
        
        # Generate new tokens one by one
        for _ in range(max_new_tokens):
            # Get next token logits
            logits, state = self.model.transformer.forward_with_state(
                out, state=state, return_state=True
            )
            
            # Get logits for the last token
            next_token_logits = logits[:, -1, :].clone()
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply sampling methods
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')
            
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    -1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = -float('Inf')
            
            # Convert to probabilities and sample
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            
            # Handle potential numerical instabilities
            probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
            probs = torch.clamp(probs, min=1e-8)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to the sequence
            out = torch.cat((out, next_token), dim=1)
            
            # Store token in history
            self.token_history.append(next_token.item())
            
            # Store state in history
            self.state_history.append(state)
            
            # Record attention patterns and memory for this token
            if store_attention:
                self._record_attention_and_memory(state)
            
            # Stop if EOS token is generated
            if self.tokenizer.eos_token_id is not None and next_token.item() == self.tokenizer.eos_token_id:
                break
        
        # Decode generated text
        generated_text = self.tokenizer.decode(self.token_history)
        generated_tokens = self.token_history
        
        return generated_text, generated_tokens
    
    def _record_attention_and_memory(self, state: TransformerState) -> None:
        """Record attention patterns and memory states for visualization."""
        # Extract attention patterns if available
        if hasattr(state, 'kv_caches') and state.kv_caches:
            # Create storage for attention weights
            attention_weights = []
            for layer_idx, (keys, values) in enumerate(state.kv_caches):
                if keys is not None:
                    # Compute approximate attention weights from keys and queries
                    # Note: This doesn't incorporate masking or softmax, but helps visualization
                    batch_size, seq_len, n_heads, head_dim = keys.shape
                    last_key = keys[:, -1, :, :]  # Last token's key
                    attention = torch.einsum('bhd,bshd->bhs', last_key, keys)
                    # Normalize to approximate attention weights
                    attention = torch.nn.functional.softmax(attention / (head_dim ** 0.5), dim=-1)
                    attention_weights.append(attention.detach().cpu().numpy())
                else:
                    attention_weights.append(None)
                    
            self.attention_history.append(attention_weights)
        
        # Extract memory states if available
        if hasattr(state, 'memory_states') and state.memory_states:
            memory_states = []
            for layer_idx, mem_state in enumerate(state.memory_states):
                if mem_state is not None:
                    mem_data = {}
                    if hasattr(mem_state, 'memory'):
                        mem_data['memory'] = mem_state.memory.detach().cpu().numpy()
                    if hasattr(mem_state, 'updates'):
                        mem_data['updates'] = mem_state.updates.detach().cpu().numpy()
                    if hasattr(mem_state, 'gates'):
                        mem_data['gates'] = mem_state.gates.detach().cpu().numpy()
                    memory_states.append(mem_data)
                else:
                    memory_states.append(None)
            
            self.memory_history.append(memory_states)
    
    def visualize_token_generation(
        self, 
        step_idx: int,
        layer_idx: int = 0,
        head_idx: int = 0,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """Visualize attention and memory for a specific token generation step.
        
        Args:
            step_idx: Generation step index
            layer_idx: Model layer index to visualize
            head_idx: Attention head index to visualize
            figsize: Figure size
            save_path: Path to save the figure (optional)
            show: Whether to display the figure
        """
        if step_idx >= len(self.token_history):
            logger.error(f"Step index {step_idx} exceeds history length {len(self.token_history)}")
            return
        
        # Get token sequence up to this step
        tokens = self.token_history[:step_idx + 1]
        token_labels = [self.tokenizer.decode([t]) for t in tokens]
        
        # Create figure with subplots for attention and memory
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[3, 1])
        
        # Plot attention pattern
        if self.attention_history and step_idx < len(self.attention_history):
            layer_attention = self.attention_history[step_idx][layer_idx]
            if layer_attention is not None:
                ax_attn = fig.add_subplot(gs[0, 0])
                attention_weights = layer_attention[0, head_idx, :step_idx+1]
                im = ax_attn.imshow(
                    attention_weights.reshape(1, -1), 
                    cmap=ATTENTION_COLORMAP,
                    aspect='auto'
                )
                ax_attn.set_title(f"Layer {layer_idx+1}, Head {head_idx+1} Attention")
                ax_attn.set_yticks([])
                ax_attn.set_xticks(range(len(token_labels)))
                ax_attn.set_xticklabels(token_labels, rotation=45, ha="right")
                plt.colorbar(im, ax=ax_attn, orientation="vertical", pad=0.01)
        
        # Plot memory activation
        if self.memory_history and step_idx < len(self.memory_history):
            layer_memory = self.memory_history[step_idx][layer_idx]
            if layer_memory and 'memory' in layer_memory:
                ax_mem = fig.add_subplot(gs[0, 1])
                memory = layer_memory['memory'][0]  # First batch item
                im = ax_mem.imshow(
                    memory, 
                    cmap=MEMORY_COLORMAP,
                    aspect='auto'
                )
                ax_mem.set_title(f"Layer {layer_idx+1} Memory")
                ax_mem.set_yticks([])
                plt.colorbar(im, ax=ax_mem, orientation="vertical", pad=0.01)
        
        # Plot token probabilities (from last step)
        if step_idx > 0 and self.state_history:
            ax_prob = fig.add_subplot(gs[1, :])
            
            # Get logits from previous step and create token probabilities
            prev_logits, _ = self.model.transformer.forward_with_state(
                torch.tensor([tokens[:-1]]).to(self.device),
                state=self.state_history[step_idx-1] if step_idx > 1 else None, 
                return_state=True
            )
            
            # Get probabilities for the last position
            last_logits = prev_logits[0, -1, :]
            probs = torch.nn.functional.softmax(last_logits, dim=-1).detach().cpu().numpy()
            
            # Get the top token probabilities
            top_k = 10
            top_indices = np.argsort(probs)[-top_k:][::-1]
            top_probs = probs[top_indices]
            top_tokens = [self.tokenizer.decode([idx]) for idx in top_indices]
            
            # Highlight the actual chosen token
            chosen_token = tokens[-1]
            chosen_idx = np.where(top_indices == chosen_token)[0]
            colors = ['skyblue'] * top_k
            if len(chosen_idx) > 0:
                colors[chosen_idx[0]] = 'coral'
            
            # Plot probabilities
            ax_prob.bar(top_tokens, top_probs, color=colors)
            ax_prob.set_title(f"Top {top_k} Token Probabilities")
            ax_prob.set_ylabel("Probability")
            ax_prob.set_xticklabels(top_tokens, rotation=45, ha="right")
        
        # Add generation text so far
        text_so_far = self.tokenizer.decode(tokens)
        plt.figtext(0.5, 0.95, f"Generated (step {step_idx}): {text_so_far}", 
                    ha="center", va="top", fontsize=12, wrap=True,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Saved visualization to {save_path}")
        
        # Show or close
        if show:
            plt.show()
        else:
            plt.close()
    
    def visualize_generation_animation(
        self,
        delay: int = 500,  # milliseconds between frames
        layer_idx: int = 0,
        head_idx: int = 0,
        save_path: Optional[str] = None,
    ) -> None:
        """Create an animation of attention and memory throughout generation.
        
        Args:
            delay: Delay between frames in milliseconds
            layer_idx: Model layer index to visualize
            head_idx: Attention head index to visualize
            save_path: Path to save the animation (optional)
        """
        # Create frames for each generation step
        frames = []
        for step_idx in range(len(self.token_history)):
            # Set up figure for this frame
            fig = plt.figure(figsize=(12, 8))
            gs = GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[3, 1])
            
            # Get token sequence up to this step
            tokens = self.token_history[:step_idx + 1]
            token_labels = [self.tokenizer.decode([t]) for t in tokens]
            
            # Plot attention pattern
            if self.attention_history and step_idx < len(self.attention_history):
                layer_attention = self.attention_history[step_idx][layer_idx]
                if layer_attention is not None:
                    ax_attn = fig.add_subplot(gs[0, 0])
                    attention_weights = layer_attention[0, head_idx, :step_idx+1]
                    im = ax_attn.imshow(
                        attention_weights.reshape(1, -1), 
                        cmap=ATTENTION_COLORMAP,
                        aspect='auto'
                    )
                    ax_attn.set_title(f"Layer {layer_idx+1}, Head {head_idx+1} Attention")
                    ax_attn.set_yticks([])
                    ax_attn.set_xticks(range(len(token_labels)))
                    ax_attn.set_xticklabels(token_labels, rotation=45, ha="right")
                    plt.colorbar(im, ax=ax_attn, orientation="vertical", pad=0.01)
            
            # Plot memory activation
            if self.memory_history and step_idx < len(self.memory_history):
                layer_memory = self.memory_history[step_idx][layer_idx]
                if layer_memory and 'memory' in layer_memory:
                    ax_mem = fig.add_subplot(gs[0, 1])
                    memory = layer_memory['memory'][0]  # First batch item
                    im = ax_mem.imshow(
                        memory, 
                        cmap=MEMORY_COLORMAP,
                        aspect='auto'
                    )
                    ax_mem.set_title(f"Layer {layer_idx+1} Memory")
                    ax_mem.set_yticks([])
                    plt.colorbar(im, ax=ax_mem, orientation="vertical", pad=0.01)
            
            # Add generation text so far
            text_so_far = self.tokenizer.decode(tokens)
            plt.figtext(0.5, 0.95, f"Generated (step {step_idx}): {text_so_far}", 
                        ha="center", va="top", fontsize=12, wrap=True,
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            
            plt.tight_layout()
            
            # Capture the frame
            frames.append([fig])
            plt.close()
        
        # Create animation
        if frames:
            fig = plt.figure(figsize=(12, 8))
            ani = animation.ArtistAnimation(fig, frames, interval=delay, blit=True, repeat_delay=1000)
            
            # Save if requested
            if save_path:
                if save_path.endswith('.gif'):
                    ani.save(save_path, writer='pillow', fps=1000/delay)
                else:
                    ani.save(save_path, writer='ffmpeg', fps=1000/delay)
                logger.info(f"Saved animation to {save_path}")
            
            plt.show()
    
    def save_attention_heatmaps(
        self, 
        output_dir: Optional[str] = None,
        layer_idx: int = 0,
        head_idx: int = 0,
    ) -> None:
        """Save attention heatmaps for each generation step.
        
        Args:
            output_dir: Directory to save heatmaps (default: self.output_dir/heatmaps)
            layer_idx: Model layer index to visualize
            head_idx: Attention head index to visualize
        """
        # Set up output directory
        if output_dir is None:
            output_dir = self.output_dir / "heatmaps"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save heatmap for each step
        for step_idx in range(len(self.token_history)):
            save_path = os.path.join(output_dir, f"step_{step_idx:03d}.png")
            self.visualize_token_generation(
                step_idx=step_idx,
                layer_idx=layer_idx,
                head_idx=head_idx,
                save_path=save_path,
                show=False
            )
        
        logger.info(f"Saved {len(self.token_history)} heatmaps to {output_dir}")
        logger.info("You can create a GIF with: convert -delay 100 -loop 0 heatmaps/step_*.png animation.gif")
    
    def visualize_attention_across_heads(
        self,
        step_idx: int,
        layer_idx: int = 0,
        figsize: Tuple[int, int] = (15, 10),
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """Visualize attention patterns across all heads for a specific step.
        
        Args:
            step_idx: Generation step index
            layer_idx: Model layer index to visualize
            figsize: Figure size
            save_path: Path to save the figure (optional)
            show: Whether to display the figure
        """
        if step_idx >= len(self.token_history):
            logger.error(f"Step index {step_idx} exceeds history length {len(self.token_history)}")
            return
        
        # Get token sequence up to this step
        tokens = self.token_history[:step_idx + 1]
        token_labels = [self.tokenizer.decode([t]) for t in tokens]
        
        # Get attention patterns for all heads
        if not self.attention_history or step_idx >= len(self.attention_history):
            logger.error(f"No attention history available for step {step_idx}")
            return
            
        layer_attention = self.attention_history[step_idx][layer_idx]
        if layer_attention is None:
            logger.error(f"No attention patterns available for layer {layer_idx}")
            return
            
        num_heads = layer_attention.shape[1]
        
        # Create subplots for each head
        fig, axes = plt.subplots(1, num_heads, figsize=figsize)
        if num_heads == 1:
            axes = [axes]
        
        # Plot each head's attention pattern
        for head_idx in range(num_heads):
            attention_weights = layer_attention[0, head_idx, :step_idx+1]
            im = axes[head_idx].imshow(
                attention_weights.reshape(1, -1),
                cmap=ATTENTION_COLORMAP,
                aspect='auto'
            )
            axes[head_idx].set_title(f"Head {head_idx+1}")
            axes[head_idx].set_yticks([])
            
            # Only show token labels on the first head
            if head_idx == 0:
                axes[head_idx].set_xticks(range(len(token_labels)))
                axes[head_idx].set_xticklabels(token_labels, rotation=45, ha="right")
            else:
                axes[head_idx].set_xticks([])
        
        # Add colorbar
        plt.colorbar(im, ax=axes, shrink=0.8, pad=0.01)
        
        # Add generation text so far
        text_so_far = self.tokenizer.decode(tokens)
        plt.figtext(0.5, 0.95, f"Generated (step {step_idx}): {text_so_far}", 
                   ha="center", va="top", fontsize=12, wrap=True,
                   bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Saved head visualization to {save_path}")
        
        # Show or close
        if show:
            plt.show()
        else:
            plt.close()
    
    def compare_memory_usage(
        self,
        steps: List[int],
        layer_idx: int = 0,
        figsize: Tuple[int, int] = (14, 8),
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """Compare memory usage across multiple generation steps.
        
        Args:
            steps: List of generation step indices to compare
            layer_idx: Model layer index to visualize
            figsize: Figure size
            save_path: Path to save the figure (optional)
            show: Whether to display the figure
        """
        if not self.memory_history:
            logger.error("No memory history available")
            return
            
        # Filter valid steps
        valid_steps = [s for s in steps if s < len(self.memory_history)]
        if not valid_steps:
            logger.error("No valid steps to compare")
            return
            
        # Create figure
        fig, axes = plt.subplots(len(valid_steps), 1, figsize=figsize, sharex=True)
        if len(valid_steps) == 1:
            axes = [axes]
        
        # Plot memory for each step
        for i, step_idx in enumerate(valid_steps):
            # Get token up to this step for labeling
            token = self.token_history[step_idx]
            token_str = self.tokenizer.decode([token])
            
            # Get memory for this step
            layer_memory = self.memory_history[step_idx][layer_idx]
            if layer_memory and 'memory' in layer_memory:
                memory = layer_memory['memory'][0]  # First batch item
                im = axes[i].imshow(
                    memory,
                    cmap=MEMORY_COLORMAP,
                    aspect='auto'
                )
                axes[i].set_title(f"Step {step_idx}: Token '{token_str}'")
                axes[i].set_ylabel("Memory Index")
            
        # Add colorbar to last subplot
        plt.colorbar(im, ax=axes, shrink=0.8, pad=0.01)
        
        # Set common x-axis label
        axes[-1].set_xlabel("Memory Dimension")
        
        plt.tight_layout()
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Saved memory comparison to {save_path}")
        
        # Show or close
        if show:
            plt.show()
        else:
            plt.close()


# Convenience functions for simple API access
def visualize_generation_attention(
    model: TitansLM,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 20,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    layer_idx: int = 0,
    head_idx: int = 0,
    output_dir: str = "./attention_viz",
    create_animation: bool = True,
    save_frames: bool = False,
) -> str:
    """Generate text and visualize attention patterns.
    
    Args:
        model: The TitansLM model to visualize
        tokenizer: Tokenizer to decode tokens
        prompt: Text prompt
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
        layer_idx: Model layer index to visualize
        head_idx: Attention head index to visualize
        output_dir: Directory to save visualizations
        create_animation: Whether to create an animation
        save_frames: Whether to save individual frames
        
    Returns:
        generated_text: The generated text
    """
    # Create visualizer
    visualizer = AttentionVisualizer(
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir
    )
    
    # Generate text with attention tracking
    generated_text, _ = visualizer.generate_with_history(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
    )
    
    # Save individual frames if requested
    if save_frames:
        visualizer.save_attention_heatmaps(
            layer_idx=layer_idx,
            head_idx=head_idx
        )
    
    # Create animation if requested
    if create_animation:
        animation_path = os.path.join(output_dir, "attention_animation.mp4")
        visualizer.visualize_generation_animation(
            layer_idx=layer_idx,
            head_idx=head_idx,
            save_path=animation_path
        )
    
    # Return generated text
    return generated_text