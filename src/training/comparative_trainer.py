"""Comparative training module for multiple models."""

import torch
import time
import os
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from tqdm import tqdm
import torch.nn.utils as nn_utils
from torch.cuda.amp import autocast, GradScaler

from src.models.transformer_arch.titans_transformer import TitansLM
from src.models.transformer_arch.vanilla_transformer import VanillaLM
from src.training.comparison_metrics import ComparativeMetrics
from src.evaluation.comparison import evaluate_models, compare_models_inference
from src.training.utils import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_one_cycle_schedule,
    get_grad_norm
)


class ComparativeTrainer:
    """Trainer for comparing multiple model architectures."""
    
    def __init__(
        self,
        models: Dict[str, Union[TitansLM, VanillaLM]],
        optimizers: Dict[str, torch.optim.Optimizer],
        train_dataloader: torch.utils.data.DataLoader,
        eval_dataloader: torch.utils.data.DataLoader,
        metrics: ComparativeMetrics,
        pad_token_id: int = 0,  # For label masking
        device: str = 'cuda',
        log_interval: int = 100,
        eval_interval: int = 500,
        save_dir: str = './checkpoints',
        save_interval: int = 1000,
        label_smoothing: float = 0.1,
        gradient_accumulation_steps: int = 1,  # Added gradient accumulation
        max_grad_norm: float = 0.5,  # Tighter clipping
    ):
        """Initialize the comparative trainer."""
        self.models = models
        self.optimizers = optimizers
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.metrics = metrics
        self.pad_token_id = pad_token_id
        self.device = device
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_dir = save_dir
        self.save_interval = save_interval
        # Create GradScalers with device_type parameter to avoid deprecation warning
        self.scalers = {name: GradScaler(device_type=device) for name in models.keys()}  # For mixed precision
        self.label_smoothing = label_smoothing
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        os.makedirs(save_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize model parameters with Xavier uniform for consistency
        for name, model in self.models.items():
            # Use improved initialization method
            self._initialize_model_parameters(model)
            if name not in optimizers:
                raise ValueError(f"No optimizer for model '{name}'")

    def _initialize_model_parameters(self, model):
        """Initialize model parameters using Xavier uniform for consistency."""
        for p in model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def create_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 1000,
        max_steps: int = 100000,
        scheduler_type: str = 'cosine'
    ):
        """Create a learning rate scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            warmup_steps: Number of warmup steps
            max_steps: Total number of training steps
            scheduler_type: Type of scheduler ('cosine', 'polynomial', 'one_cycle')
            
        Returns:
            Learning rate scheduler
        """
        if scheduler_type == 'cosine':
            return get_cosine_schedule_with_warmup(
                optimizer, warmup_steps, max_steps, num_cycles=0.5
            )
        elif scheduler_type == 'polynomial':
            return get_polynomial_decay_schedule_with_warmup(
                optimizer, warmup_steps, max_steps, lr_end=1e-6, power=1.0
            )
        elif scheduler_type == 'one_cycle':
            return get_one_cycle_schedule(
                optimizer, max_steps, pct_ramp_up=0.3, div_factor=10.0
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    def train(
        self,
        num_steps: int,
        eval_steps: int = 100,
        warmup_steps: int = 1000,
        lr_scheduler: Optional[Dict[str, torch.optim.lr_scheduler._LRScheduler]] = None,
        use_fp16: bool = True,
        scheduler_type: str = 'cosine',
    ):
        """Train all models with improved stability measures."""
        self.logger.info(f"Starting training for: {', '.join(self.models.keys())}")
        
        global_step = 0
        train_iterator = iter(self.train_dataloader)
        accumulation_counters = {name: 0 for name in self.models}
        schedulers = {}
        
        # Create schedulers if not provided
        if lr_scheduler is None:
            for name, optimizer in self.optimizers.items():
                schedulers[name] = self.create_scheduler(
                    optimizer, 
                    warmup_steps=warmup_steps, 
                    max_steps=num_steps,
                    scheduler_type=scheduler_type
                )
        else:
            schedulers = lr_scheduler
        
        with tqdm(total=num_steps, desc="Training") as pbar:
            while global_step < num_steps:
                try:
                    batch = next(train_iterator)
                except StopIteration:
                    train_iterator = iter(self.train_dataloader)
                    batch = next(train_iterator)
                
                # Proper label shifting and masking
                original_input = batch['input_ids'].to(self.device)
                input_ids = original_input[:, :-1]
                labels = original_input[:, 1:].clone()
                labels[labels == self.pad_token_id] = -100  # Mask padding

                for name, model in self.models.items():
                    model.train()
                    optimizer = self.optimizers[name]
                    scaler = self.scalers[name]
                    
                    # Gradient accumulation setup
                    if accumulation_counters[name] % self.gradient_accumulation_steps == 0:
                        optimizer.zero_grad()

                    # Mixed precision training
                    if use_fp16:
                        with autocast():
                            out = model(input_ids)
                            # Handle models returning (logits, state)
                            logits = out[0] if isinstance(out, tuple) else out
                            
                            # Handle potential NaN in logits
                            if torch.isnan(logits).any():
                                self.logger.warning(f"NaN detected in logits for {name}, skipping batch")
                                continue
                            loss = torch.nn.functional.cross_entropy(
                                logits.view(-1, logits.size(-1)),
                                labels.view(-1),
                                ignore_index=-100,
                                label_smoothing=self.label_smoothing
                            )
                            # Scale loss for accumulation
                            scaled_loss = loss / self.gradient_accumulation_steps

                        # Backward pass with scaling
                        scaler.scale(scaled_loss).backward()
                        
                        # Step if accumulation complete
                        accumulation_counters[name] += 1
                        if accumulation_counters[name] % self.gradient_accumulation_steps == 0:
                            # Gradient clipping
                            scaler.unscale_(optimizer)
                            
                            # Calculate gradient norm for monitoring
                            grad_norm = get_grad_norm(model)
                            if grad_norm > 10.0:
                                self.logger.warning(f"High gradient norm in {name}: {grad_norm:.2f}")
                                
                            # Apply gradient clipping
                            nn_utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                            
                            # Optimizer and scaler step
                            scaler.step(optimizer)
                            scaler.update()
                            
                            # Reset counter
                            accumulation_counters[name] = 0
                    else:
                        # Standard training without mixed precision
                        out = model(input_ids)
                        logits = out[0] if isinstance(out, tuple) else out
                        loss = torch.nn.functional.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            labels.view(-1),
                            ignore_index=-100,
                            label_smoothing=self.label_smoothing
                        )
                        scaled_loss = loss / self.gradient_accumulation_steps
                        
                        # Backward pass
                        scaled_loss.backward()
                        
                        # Step if accumulation complete
                        accumulation_counters[name] += 1
                        if accumulation_counters[name] % self.gradient_accumulation_steps == 0:
                            # Gradient clipping
                            nn_utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                            
                            # Optimizer step
                            optimizer.step()
                            
                            # Reset counter
                            accumulation_counters[name] = 0
                    
                    # Update scheduler
                    if name in schedulers:
                        scheduler = schedulers[name]
                        if accumulation_counters[name] % self.gradient_accumulation_steps == 0:
                            scheduler.step()
                    
                    self.metrics.update(name, loss, logits, labels, global_step)

                # Log metrics
                if global_step % self.log_interval == 0:
                    metrics_dict = self.metrics.compute()
                    log_str = f"Step {global_step} | "
                    for name in self.models.keys():
                        log_str += f"{name} Loss: {metrics_dict[name]['loss']:.4f}, "
                        log_str += f"PPL: {metrics_dict[name]['perplexity']:.2f} | "
                        
                        # Add learning rate info
                        if name in schedulers:
                            lr = schedulers[name].get_last_lr()[0]
                            log_str += f"LR: {lr:.6f} | "
                    
                    self.logger.info(log_str)
                    pbar.set_postfix({
                        f"{name}_loss": f"{metrics_dict[name]['loss']:.4f}" for name in self.models.keys()
                    })
                
                # Evaluate models
                if global_step % self.eval_interval == 0:
                    self.logger.info(f"Evaluating models at step {global_step}")
                    eval_metrics = self.evaluate(max_batches=eval_steps)
                    
                    # Log evaluation metrics
                    log_str = f"Eval at step {global_step} | "
                    for name in self.models.keys():
                        log_str += f"{name} Loss: {eval_metrics[name]['loss']:.4f}, "
                        log_str += f"PPL: {eval_metrics[name]['perplexity']:.2f} | "
                    
                    self.logger.info(log_str)
                    
                    # Run inference comparison
                    self._run_inference_comparison(input_ids)
                
                # Save checkpoints
                if global_step % self.save_interval == 0:
                    self.save_checkpoints(global_step)
                
                global_step += 1
                pbar.update(1)
        
        # Final evaluation
        self.logger.info(f"Training completed. Running final evaluation.")
        eval_metrics = self.evaluate(max_batches=eval_steps)
        
        # Save final checkpoints
        self.save_checkpoints(global_step)
        
        return eval_metrics
    
    def _run_inference_comparison(self, sample_input):
        """Run inference comparison between models."""
        # Use first batch item, first 10 tokens
        sample_input = sample_input[:1, :10] if sample_input.size(1) > 10 else sample_input[:1, :]
        
        inference_results = compare_models_inference(
            self.models, sample_input, max_length=20, num_trials=3, warmup_trials=1
        )
        
        # Update inference metrics
        for name, results in inference_results.items():
            self.metrics.update_inference_metrics(
                name,
                results['time_per_token'],
                results['memory_usage']
            )
            
            # Log inference stats
            self.logger.info(
                f"{name} inference: {results['time_per_token']:.4f} ms/token, "
                f"{results['memory_usage']:.2f} MB memory usage"
            )
    
    def evaluate(self, max_batches: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
        """Evaluate all models on the evaluation dataset."""
        return evaluate_models(
            self.models,
            self.eval_dataloader,
            self.metrics,
            max_batches=max_batches,
            device=self.device,
            label_smoothing=self.label_smoothing  # Pass label smoothing for consistent evaluation
        )
    
    def save_checkpoints(self, step: int):
        """Save model checkpoints."""
        for name, model in self.models.items():
            checkpoint_path = os.path.join(self.save_dir, f"{name}_step_{step}.pt")
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': self.optimizers[name].state_dict()
            }, checkpoint_path)
            
            self.logger.info(f"Saved {name} checkpoint to {checkpoint_path}")
    
    def load_checkpoints(self, step: int = None, checkpoint_paths: Dict[str, str] = None):
        """Load model checkpoints.
        
        Args:
            step: Step to load (will look for {name}_step_{step}.pt)
            checkpoint_paths: Dictionary mapping model names to checkpoint paths
        """
        if checkpoint_paths is None and step is None:
            self.logger.error("Either step or checkpoint_paths must be provided")
            return
            
        for name, model in self.models.items():
            if checkpoint_paths and name in checkpoint_paths:
                path = checkpoint_paths[name]
            elif step is not None:
                path = os.path.join(self.save_dir, f"{name}_step_{step}.pt")
            else:
                self.logger.warning(f"No checkpoint path for {name}")
                continue
                
            if os.path.exists(path):
                checkpoint = torch.load(path, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizers[name].load_state_dict(checkpoint['optimizer_state_dict'])
                self.logger.info(f"Loaded checkpoint for {name} from {path}")
            else:
                self.logger.warning(f"Checkpoint {path} for {name} not found")
