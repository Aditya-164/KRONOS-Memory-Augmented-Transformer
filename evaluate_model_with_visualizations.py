#!/usr/bin/env python
"""
Comprehensive Model Evaluation and Visualization Script for Titans Transformers.

This script provides a one-stop solution for evaluating a trained Titans model 
with various tests and visualizations:
- Language model evaluation metrics (perplexity, loss)
- Text generation quality and samples
- Attention pattern visualization
- Memory usage and efficiency analysis
- General model performance metrics

Usage:
    python evaluate_model_with_visualizations.py --checkpoint path/to/model.pt
"""

import os
import sys
import argparse
import torch
import json
import numpy as np
import random
import matplotlib
# Use non-interactive backend if no display is available
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer

# Import Titans modules - reuse imports from main.py
from titans import (
    TitansTransformer, 
    TitansLM,
    ModelConfig,
    TrainingConfig,
    DefaultConfig, 
    Trainer,
    WikiTextDataset,
    TextDataset,
    CollatorForLanguageModeling,
    get_dataloader,
    prepare_datasets,
    init_weights,
    get_logger,
    TitansLogger,
    log_step_info,
    log_model_info,
    log_memory_usage
)

# Import visualization modules
from src.visualizations import (
    plot_attention_patterns,
    plot_memory_heatmap,
    plot_text_generation,
    visualize_model_state,
    AttentionVisualizer,
    visualize_generation_attention
)

# Import training metrics and utilities
from titans.training.metrics import (
    LanguageModelMetrics, 
    MemoryMetrics, 
    GenerationMetrics, 
    MetricsTracker,
    compute_metrics_from_batch
)

# Set up logger
logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate and visualize a trained Titans model")
    
    # Model configuration
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--model_dim", type=int, default=512, help="Model dimension")
    parser.add_argument("--depth", type=int, default=12, help="Number of layers")
    parser.add_argument("--heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--dim_head", type=int, default=64, help="Dimension per attention head")
    parser.add_argument("--segment_len", type=int, default=512, help="Length of attention segments")
    parser.add_argument("--use_memory", action="store_true", help="Use neural memory in the model")
    parser.add_argument("--sliding_window", action="store_true", help="Use sliding window attention")
    
    # Data configuration
    parser.add_argument("--data_path", type=str, default=None, help="Path to data file/directory")
    parser.add_argument("--dataset", type=str, choices=["text", "wikitext"], default="text", 
                        help="Dataset type")
    parser.add_argument("--wikitext_version", type=str, choices=["2", "103"], default="103",
                        help="WikiText version")
    parser.add_argument("--tokenizer", type=str, default="gpt2", 
                        help="Tokenizer name or path")
    parser.add_argument("--seq_length", type=int, default=1024, 
                        help="Evaluation sequence length")
    parser.add_argument("--batch_size", type=int, default=8, help="Evaluation batch size")
    
    # Evaluation configuration
    parser.add_argument("--num_eval_samples", type=int, default=100, help="Number of evaluation samples")
    parser.add_argument("--num_gen_samples", type=int, default=5, help="Number of text generation samples")
    
    # Generation parameters
    parser.add_argument("--max_length", type=int, default=100, help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    
    # Visualization parameters
    parser.add_argument("--vis_attention", action="store_true", help="Visualize attention patterns")
    parser.add_argument("--vis_memory", action="store_true", help="Visualize memory patterns")
    parser.add_argument("--vis_generation", action="store_true", help="Visualize text generation process")
    parser.add_argument("--vis_all", action="store_true", help="Enable all visualizations")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="./model_evaluation", help="Output directory")
    parser.add_argument("--prompts_file", type=str, default=None, 
                        help="Path to file with prompts for text generation")
    
    # Misc configuration
    parser.add_argument("--device", type=str, default=None, 
                        help="Device to use (e.g., 'cuda', 'cpu')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.debug(f"Set random seed to {seed}")


def setup_device(device=None):
    """Set up device for computation."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    logger.info(f"Using device: {device}")
    return device

def load_or_create_tokenizer(tokenizer_name):
    """Load or create a tokenizer with BOS, EOS, and PAD tokens."""
    logger.info(f"Loading tokenizer: {tokenizer_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Ensure tokenizer has all required special tokens
        special_tokens_dict = {}
        
        # Check and set pad_token
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.debug(f"Set pad_token to eos_token ('{tokenizer.pad_token}')")
            else:
                special_tokens_dict['pad_token'] = '[PAD]'
                logger.debug("Added [PAD] token")
        
        # Check and set bos_token
        if tokenizer.bos_token is None:
            special_tokens_dict['bos_token'] = '[BOS]'
            logger.debug("Added [BOS] token")
        
        # Check and set eos_token
        if tokenizer.eos_token is None:
            special_tokens_dict['eos_token'] = '[EOS]'
            logger.debug("Added [EOS] token")
        
        # Add special tokens if needed
        if special_tokens_dict:
            num_added = tokenizer.add_special_tokens(special_tokens_dict)
            logger.debug(f"Added {num_added} special tokens to the tokenizer")
        
        logger.debug(f"Loaded tokenizer with vocab size: {len(tokenizer)}")
        logger.debug(f"Special tokens: BOS='{tokenizer.bos_token}', EOS='{tokenizer.eos_token}', PAD='{tokenizer.pad_token}'")
        
        return tokenizer
    except Exception as e:
        logger.error(f"Error loading tokenizer {tokenizer_name}: {e}")
        logger.info("Using default BERT tokenizer as fallback")
        
        # Create BERT tokenizer with all special tokens
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        special_tokens_dict = {}
        
        if tokenizer.bos_token is None:
            special_tokens_dict['bos_token'] = '[BOS]'
        if tokenizer.eos_token is None:
            special_tokens_dict['eos_token'] = '[EOS]'
        if tokenizer.pad_token is None:
            special_tokens_dict['pad_token'] = '[PAD]'
            
        if special_tokens_dict:
            tokenizer.add_special_tokens(special_tokens_dict)
            
        logger.debug(f"Created fallback tokenizer with BOS='{tokenizer.bos_token}', EOS='{tokenizer.eos_token}', PAD='{tokenizer.pad_token}'")
        return tokenizer


def create_model(args, vocab_size, device):
    """Create a Titans model based on configuration."""
    logger.info(f"Creating model with {args.depth} layers, {args.model_dim} dimensions")
    
    # Create default config for model parameters
    config = DefaultConfig()
    config.model.dim = args.model_dim
    config.model.depth = args.depth
    config.model.heads = args.heads
    config.model.dim_head = args.dim_head
    config.model.segment_len = args.segment_len
    
    # Create base transformer model
    model = TitansTransformer(
        dim=config.model.dim,
        depth=config.model.depth,
        vocab_size=vocab_size,
        seq_len=args.seq_length,
        dim_head=config.model.dim_head,
        heads=config.model.heads,
        segment_len=config.model.segment_len,
        use_memory=args.use_memory,
        sliding=args.sliding_window,
        num_longterm_mem_tokens=config.model.num_longterm_mem_tokens,
        num_persist_mem_tokens=config.model.num_persist_mem_tokens,
        use_flex_attn=config.model.use_flex_attn,
        accept_value_residual=config.model.neural_memory_add_value_residual
    )
    
    # Initialize weights
    model.apply(lambda m: init_weights(m))
    
    # Wrap in language model for generation capabilities
    language_model = TitansLM(
        transformer=model,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )
    
    # Log model architecture and parameter count
    log_model_info(logger, language_model)
    
    # Move model to device
    language_model = language_model.to(device)
    
    return language_model


def load_model_from_checkpoint(model, checkpoint_path, device):
    """Load model weights from checkpoint."""
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Check if the checkpoint contains model state_dict directly or has it under 'model_state_dict' key
        if 'model_state_dict' in checkpoint:
            missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            missing, unexpected = model.load_state_dict(checkpoint, strict=False)
        
        if missing:
            logger.warning(f"Missing keys when loading model: {missing}")
        elif unexpected:
            logger.warning(f"Unexpected keys when loading model: {unexpected}")
        else:
            logger.info("All model parameters loaded successfully")
        logger.info("Model loaded successfully from checkpoint")
        return True
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        return False


def prepare_evaluation_data(args, tokenizer):
    """Prepare dataset for evaluation."""
    if args.dataset == "wikitext":
        logger.info(f"Loading WikiText-{args.wikitext_version} dataset...")
        eval_dataset = WikiTextDataset(
            data_dir=args.data_path,
            split="test",
            tokenizer=tokenizer,
            seq_length=args.seq_length,
            version=args.wikitext_version
        )
    elif args.dataset == "text":
        logger.info(f"Loading text dataset from {args.data_path}...")
        
        # For text datasets, use a simple approach without monkey patching
        logger.info("Loading dataset...")
        with tqdm(total=100, desc="Loading dataset") as pbar:
            # Load the dataset normally
            _, eval_dataset, _ = prepare_datasets(
                data_path=args.data_path,
                tokenizer=tokenizer,
                train_seq_length=args.seq_length,
                val_seq_length=args.seq_length,
                val_split=0.1,
                test_split=0.0,
                seed=args.seed
            )
            pbar.update(100)  # Complete the progress bar
    else:
        raise ValueError(f"Unsupported dataset type: {args.dataset}")
        
    logger.info(f"Evaluation dataset size: {len(eval_dataset)}")
    
    # Create dataloader
    collator = CollatorForLanguageModeling(
        tokenizer
    )
    
    eval_dataloader = get_dataloader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
        drop_last=False
    )
    
    return eval_dataset, eval_dataloader

def evaluate_model(model, eval_dataloader, device, args=None):
    """Evaluate model on the dataset."""
    logger.info("Evaluating model...")
    model.eval()
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker()
    
    total_samples = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
            batch = {k: v.to(device) for k, v in batch.items()}
            input_ids = batch["input_ids"]
            labels = batch.get("labels", input_ids.clone())

            # Forward pass and memory state handling
            if isinstance(model, TitansLM):
                logits, state = model(input_ids, return_state=True)
                memory_states = getattr(state, 'memory_states', None)
            else:
                logits = model(input_ids)
                memory_states = None
            
            # Compute metrics with batch smoothing if provided
            label_smoothing = args.label_smoothing if args and hasattr(args, 'label_smoothing') else 0.0
            batch_metrics = compute_metrics_from_batch(
                logits=logits,
                targets=labels,
                memory_states=memory_states,
                prefix="",
                label_smoothing=label_smoothing
            )
            
            # Update metrics tracker
            batch_size = input_ids.size(0)
            metrics_tracker.update(batch_metrics, batch_size=batch_size)
            total_samples += batch_size
            
            # Log progress
            if (batch_idx + 1) % 10 == 0:
                logger.debug(f"Evaluated {batch_idx + 1} batches...")
            
            # Limit number of evaluation samples if specified
            if args and hasattr(args, 'num_eval_samples') and args.num_eval_samples > 0 and total_samples >= args.num_eval_samples:
                break
    
    # Get final metrics
    all_metrics = metrics_tracker.get_all_averages()
    all_metrics["total_samples"] = total_samples
    
    # Log results
    logger.info(f"Evaluation results:")
    for name, value in all_metrics.items():
        if isinstance(value, float):
            logger.info(f"  {name}: {value:.4f}")
        else:
            logger.info(f"  {name}: {value}")
    
    return all_metrics


def generate_text_samples(model, tokenizer, prompts, args, device):
    """Generate text samples from the model."""
    logger.info(f"Generating {len(prompts)} text samples...")
    model.eval()
    
    results = []
    
    for i, prompt in enumerate(prompts):
        logger.info(f"Generating sample {i+1}/{len(prompts)}")
        logger.info(f"Prompt: {prompt}")
        
        # Tokenize prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        # Generate text
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p
            )
        
        # Decode generated text
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Store results
        results.append({
            "prompt": prompt,
            "generated_text": generated_text,
            "num_tokens": len(output_ids[0]) - len(input_ids[0])
        })
        
        logger.info(f"Generated: {generated_text}")
        logger.info("-" * 50)
    
    return results


def visualize_attention_patterns(model, tokenizer, prompts, args, device, output_dir):
    """Visualize attention patterns during generation."""
    logger.info("Visualizing attention patterns...")
    model.eval()
    
    # Create visualizer
    visualizer = AttentionVisualizer(
        model=model,
        tokenizer=tokenizer,
        output_dir=os.path.join(output_dir, "attention_viz")
    )
    
    # Generate and visualize for each prompt
    for i, prompt in enumerate(prompts[:min(3, len(prompts))]):  # Limit to 3 prompts for attention viz
        logger.info(f"Visualizing attention for prompt {i+1}/{min(3, len(prompts))}")
        
        prompt_slug = prompt.replace(" ", "_")[:20]  # Create a short slug for the filename
        
        # Generate with history tracking for visualization
        generated_text, _ = visualizer.generate_with_history(
            prompt=prompt,
            max_new_tokens=min(30, args.max_length),  # Limit to 30 tokens for visualization
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
        
        # Save visualization frames
        vis_dir = os.path.join(output_dir, f"attention_viz_prompt_{i+1}")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Determine final step index
        final_step = len(visualizer.token_history) - 1
        # Ensure we have at least two time steps to visualize
        if final_step < 1:
            logger.warning(f"Not enough tokens generated to visualize attention for prompt {i+1}")
            continue
        
        # Visualize different layers and heads
        for layer_idx in range(min(3, args.depth)):  # First 3 layers
            for head_idx in range(min(4, args.heads)):  # First 4 heads per layer
                try:
                    visualizer.visualize_token_generation(
                        step_idx=final_step,
                        layer_idx=layer_idx,
                        head_idx=head_idx,
                        save_path=os.path.join(vis_dir, f"layer_{layer_idx}_head_{head_idx}.png"),
                        show=False
                    )
                except IndexError:
                    logger.warning(f"Unable to visualize token generation for layer {layer_idx}, head {head_idx} at step {final_step}")
                    continue

        # Create an animation for the first layer's first head
        try:
            visualizer.visualize_generation_animation(
                layer_idx=0,
                head_idx=0,
                save_path=os.path.join(vis_dir, "attention_animation.mp4"),
            )
        except Exception as e:
            logger.warning(f"Failed to create attention animation: {e}")

        # Compare heads for the final step
        try:
            visualizer.visualize_attention_across_heads(
                step_idx=final_step,
                layer_idx=0,
                save_path=os.path.join(vis_dir, "attention_across_heads.png"),
                show=False
            )
        except IndexError:
            logger.warning(f"Unable to visualize attention across heads at step {final_step}")
        
        logger.info(f"Saved attention visualizations to {vis_dir}")


def visualize_memory_patterns(model, tokenizer, prompts, args, device, output_dir):
    """Visualize memory patterns and efficiency."""
    logger.info("Visualizing memory patterns...")
    
    # Check if model uses memory
    has_memory = args.use_memory and hasattr(model, 'transformer') and hasattr(model.transformer, 'layers') and \
                any(hasattr(layer, 'memory') and layer.memory is not None for layer in model.transformer.layers)
    
    if not has_memory:
        logger.warning("Model doesn't use neural memory. Skipping memory visualization.")
        return
    
    # Prepare a memory visualization directory
    mem_vis_dir = os.path.join(output_dir, "memory_viz")
    os.makedirs(mem_vis_dir, exist_ok=True)
    
    # Process a few prompts
    for i, prompt in enumerate(prompts[:min(2, len(prompts))]):
        logger.info(f"Visualizing memory for prompt {i+1}/{min(2, len(prompts))}")
        
        # Tokenize prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        # Generate text while capturing memory states
        model.eval()
        with torch.no_grad():
            # First run model once to get states via TitansLM.forward
            logits, state = model(input_ids, return_state=True)
            
            # Plot memory heatmap if there are memory states
            if hasattr(state, 'memory_states') and state.memory_states:
                memory_states = state.memory_states
                plot_memory_heatmap(
                    memory_states,
                    title=f"Memory Activations for Prompt {i+1}",
                    save_path=os.path.join(mem_vis_dir, f"memory_heatmap_prompt_{i+1}.png"),
                    show=False
                )
                
                # Memory updates visualization
                plot_memory_updates(
                    memory_states,
                    title=f"Memory Updates for Prompt {i+1}",
                    save_path=os.path.join(mem_vis_dir, f"memory_updates_prompt_{i+1}.png"),
                    show=False
                )
                
                # Full model state visualization
                visualize_model_state(
                    state,
                    title=f"Model State for Prompt {i+1}",
                    save_path=os.path.join(mem_vis_dir, f"model_state_prompt_{i+1}.png"),
                    show=False
                )
                
                logger.info(f"Saved memory visualizations to {mem_vis_dir}")
            else:
                logger.warning("No memory states found in model state. Skipping memory visualization.")


def plot_memory_updates(memory_states, title="Memory Updates", save_path=None, show=True):
    """Plot memory update magnitudes across layers."""
    if not memory_states:
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    update_norms = []
    for i, state in enumerate(memory_states):
        if hasattr(state, 'updates'):
            # Compute norm of updates (vector magnitudes)
            update_norm = state.updates.norm(dim=-1).mean().cpu().numpy()
            update_norms.append((i, update_norm))
    
    if not update_norms:
        return
    
    layers, norms = zip(*update_norms)
    ax.bar(layers, norms, alpha=0.7)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Average Update Magnitude')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    if show:
        plt.show()
    else:
        plt.close()


def generate_evaluation_report(metrics, generation_results, output_dir, args):
    """Generate an evaluation report with all results."""
    logger.info("Generating evaluation report...")
    
    # Create report data structure
    report = {
        "model_info": {
            "checkpoint": args.checkpoint,
            "model_dim": args.model_dim,
            "depth": args.depth,
            "heads": args.heads,
            "dim_head": args.dim_head,
            "segment_len": args.segment_len,
            "use_memory": args.use_memory,
            "sliding_window": args.sliding_window,
        },
        "evaluation_metrics": metrics,
        "text_generation": {
            "samples": generation_results,
            "parameters": {
                "max_length": args.max_length,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "top_p": args.top_p
            }
        },
        "timestamp": datetime.now().isoformat()
    }
    
    # Save report as JSON
    report_path = os.path.join(output_dir, "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    
    # Also save a formatted text version
    text_report_path = os.path.join(output_dir, "evaluation_report.txt")
    with open(text_report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("TITANS MODEL EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Model info
        f.write("MODEL INFORMATION:\n")
        f.write("-" * 80 + "\n")
        for key, value in report["model_info"].items():
            f.write(f"{key}: {value}\n")
        f.write("\n\n")
        
        # Metrics
        f.write("EVALUATION METRICS:\n")
        f.write("-" * 80 + "\n")
        for key, value in report["evaluation_metrics"].items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")
        f.write("\n\n")
        
        # Generation samples
        f.write("TEXT GENERATION SAMPLES:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Parameters: temperature={args.temperature}, top_k={args.top_k}, top_p={args.top_p}\n\n")
        
        for i, sample in enumerate(report["text_generation"]["samples"]):
            f.write(f"Sample {i+1}:\n")
            f.write(f"Prompt: {sample['prompt']}\n")
            f.write(f"Generated: {sample['generated_text']}\n")
            f.write(f"Generated tokens: {sample['num_tokens']}\n")
            f.write("-" * 40 + "\n\n")
    
    logger.info(f"Evaluation report saved to {report_path} and {text_report_path}")
    return report


def load_prompts(args):
    """Load or generate prompts for text generation."""
    default_prompts = [
        "The best way to predict the future is to",
        "In a world where artificial intelligence has become",
        "The relationship between humans and machines",
        "Scientists recently discovered that",
        "When I look at the stars, I feel"
    ]
    
    if args.prompts_file:
        try:
            with open(args.prompts_file, "r", encoding="utf-8") as f:
                prompts = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(prompts)} prompts from {args.prompts_file}")
        except Exception as e:
            logger.error(f"Error loading prompts from {args.prompts_file}: {e}")
            logger.info("Using default prompts instead")
            prompts = default_prompts
    else:
        prompts = default_prompts
        logger.info("Using default prompts for text generation")
    
    # Limit to the specified number of generation samples
    prompts = prompts[:args.num_gen_samples]
    return prompts


def main(args):
    """Main entry point for the script."""
    # Set random seed
    set_seed(args.seed)
    
    # Configure device
    device = setup_device(args.device)
    
    # Set up output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"eval_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Results will be saved to {output_dir}")
    
    # Load tokenizer
    tokenizer = load_or_create_tokenizer(args.tokenizer)
    
    # Create model
    model = create_model(args, len(tokenizer), device)
    
    # Load checkpoint
    if not load_model_from_checkpoint(model, args.checkpoint, device):
        logger.error("Failed to load checkpoint. Exiting.")
        return 1
    
    # Prepare evaluation data
    eval_dataset, eval_dataloader = prepare_evaluation_data(args, tokenizer)
    
    # Evaluate model
    metrics = evaluate_model(model, eval_dataloader, device)
    
    # Load prompts for generation and visualization
    prompts = load_prompts(args)
    
    # Generate text samples
    generation_results = generate_text_samples(model, tokenizer, prompts, args, device)
    
    # Visualizations
    if args.vis_all or args.vis_attention:
        visualize_attention_patterns(model, tokenizer, prompts, args, device, output_dir)
    
    if args.vis_all or args.vis_memory:
        visualize_memory_patterns(model, tokenizer, prompts, args, device, output_dir)
        
    if args.vis_all or args.vis_generation:
        # Generation attention visualization is currently unstable; skipping
        logger.info("Skipping generation attention visualization (disabled)")
    
    # Generate evaluation report
    report = generate_evaluation_report(metrics, generation_results, output_dir, args)
    
    logger.info("Evaluation completed successfully!")
    return 0


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    try:
        sys.exit(main(args))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Error during execution: {e}")
        sys.exit(1)