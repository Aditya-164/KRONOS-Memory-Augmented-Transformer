"""Training utilities for Titans models.

This module provides utility functions for training Titans models, including
learning rate schedulers, gradient accumulation, and other training helpers.
"""

import math
import torch
import numpy as np
from typing import List, Dict, Any, Callable, Union, Optional
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR


def get_polynomial_decay_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    lr_end: float = 1e-7,
    power: float = 1.0,
    last_epoch: int = -1
) -> LambdaLR:
    """
    Create a schedule with polynomial decay and warmup.
    
    This scheduler provides a more stable training process by using:
    1. A linear warmup phase for 'num_warmup_steps' 
    2. A polynomial decay phase from peak learning rate to 'lr_end'
    
    Args:
        optimizer: The optimizer for which to schedule the learning rate
        num_warmup_steps: The number of steps for the warmup phase
        num_training_steps: The total number of training steps
        lr_end: Minimum learning rate
        power: Power factor for polynomial decay
        last_epoch: The index of the last epoch
        
    Returns:
        A LambdaLR scheduler with polynomial decay and warmup
    """
    # Validate inputs
    if num_warmup_steps < 0:
        raise ValueError(f"num_warmup_steps must be non-negative, got {num_warmup_steps}")
    if num_training_steps <= 0:
        raise ValueError(f"num_training_steps must be positive, got {num_training_steps}")
    if num_warmup_steps > num_training_steps:
        raise ValueError(
            f"num_warmup_steps must be less than or equal to num_training_steps, "
            f"got {num_warmup_steps} > {num_training_steps}"
        )
    if lr_end < 0:
        raise ValueError(f"lr_end must be non-negative, got {lr_end}")
    
    # Extract initial learning rates for each parameter group
    lr_init_values = [group['lr'] for group in optimizer.param_groups]
    
    def lr_lambda(current_step: int) -> float:
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Polynomial decay phase
        if current_step >= num_training_steps:
            return lr_end / lr_init_values[0]
        
        lr_range = lr_init_values[0] - lr_end
        decay_steps = num_training_steps - num_warmup_steps
        pct_remaining = 1.0 - float(current_step - num_warmup_steps) / float(decay_steps)
        decay = lr_range * pct_remaining ** power + lr_end
        
        return decay / lr_init_values[0]
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, 
    num_warmup_steps: int, 
    num_training_steps: int, 
    num_cycles: float = 0.5, 
    last_epoch: int = -1
) -> LambdaLR:
    """
    Create a cosine schedule with warmup.
    
    Args:
        optimizer: The optimizer for which to schedule the learning rate
        num_warmup_steps: The number of steps for the warmup phase
        num_training_steps: The total number of training steps
        num_cycles: The number of cycles for the cosine decay (0.5 = half cycle)
        last_epoch: The index of last epoch
        
    Returns:
        A LambdaLR scheduler with cosine decay and warmup
    """
    def lr_lambda(current_step):
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_one_cycle_schedule(
    optimizer: Optimizer,
    num_steps: int,
    lr_max: Optional[float] = None,
    pct_ramp_up: float = 0.3,
    div_factor: float = 25.0,
    final_div_factor: float = 1e4,
    last_epoch: int = -1
) -> LambdaLR:
    """
    Create a one-cycle learning rate schedule as described in the paper
    'Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates'.
    
    The schedule:
    1. Ramps up from lr_max/div_factor to lr_max during pct_ramp_up * num_steps steps
    2. Ramps down from lr_max to lr_max/final_div_factor during (1-pct_ramp_up) * num_steps steps
    
    Args:
        optimizer: The optimizer for which to schedule the learning rate
        num_steps: The total number of steps in the schedule
        lr_max: The maximum learning rate (if None, use optimizer's initial lr)
        pct_ramp_up: Percentage of steps spent ramping up
        div_factor: Initial learning rate = lr_max / div_factor
        final_div_factor: Final learning rate = lr_max / final_div_factor
        last_epoch: The index of last epoch
        
    Returns:
        A LambdaLR scheduler with one-cycle policy
    """
    if lr_max is None:
        lr_max = optimizer.param_groups[0]['lr']
    
    # Calculate key points in the cycle
    ramp_up_steps = int(pct_ramp_up * num_steps)
    ramp_down_steps = num_steps - ramp_up_steps
    
    def lr_lambda(step):
        if step < ramp_up_steps:
            # Linear ramp up phase
            pct_completed = float(step) / float(ramp_up_steps)
            return (1.0 - 1.0/div_factor) * pct_completed + 1.0/div_factor
        else:
            # Cosine ramp down phase
            pct_completed = float(step - ramp_up_steps) / float(ramp_down_steps)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * pct_completed))
            return (1.0 - 1.0/final_div_factor) * cosine_decay + 1.0/final_div_factor
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


class LinearWarmupCosineLRScheduler:
    """
    Creates a learning rate scheduler with linear warmup and cosine annealing.
    
    This scheduler is designed to give more precise control over the learning rate
    throughout training, with explicit warmup steps and a smooth cosine decay.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_factor: float = 0.1,
        base_lr: Optional[float] = None,
        warmup_start_lr_factor: float = 0.01,
    ):
        """
        Initialize the scheduler.
        
        Args:
            optimizer: The optimizer for which to schedule the learning rate
            warmup_steps: Number of steps for the warmup phase
            total_steps: Total number of training steps
            min_lr_factor: Minimum learning rate as a ratio of base LR
            base_lr: Base learning rate (if None, use optimizer's initial lr)
            warmup_start_lr_factor: Initial LR during warmup as a ratio of base LR
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_factor = min_lr_factor
        
        if base_lr is None:
            self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        else:
            self.base_lrs = [base_lr for _ in optimizer.param_groups]
            
        self.warmup_start_lr_factor = warmup_start_lr_factor
        self.step_count = 0
        
    def step(self) -> None:
        """Update the learning rate for each parameter group based on current step."""
        self.step_count += 1
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self._get_lr(self.step_count, self.base_lrs[i])
    
    def _get_lr(self, step: int, base_lr: float) -> float:
        """Calculate the learning rate for the current step."""
        if step < self.warmup_steps:
            # Linear warmup
            alpha = float(step) / float(max(1, self.warmup_steps))
            warmup_factor = self.warmup_start_lr_factor * (1.0 - alpha) + alpha
            return base_lr * warmup_factor
        else:
            # Cosine annealing
            progress = float(step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr_factor * base_lr + (base_lr - self.min_lr_factor * base_lr) * cosine_factor


def torch_seed_all(seed: int = 42) -> None:
    """Set random seed for all relevant RNG sources to ensure reproducibility.
    
    Args:
        seed: Random seed to set
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    
    # Make PyTorch operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_grad_norm(model: torch.nn.Module) -> float:
    """Calculate gradient norm for the model parameters.
    
    This is useful for monitoring gradient behavior during training.
    
    Args:
        model: PyTorch model
        
    Returns:
        Gradient norm (float)
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def log_training_info(
    logger,  # Logger object
    step: int,  # Current step
    train_loss: float,  # Training loss
    learning_rate: float,  # Current learning rate
    grad_norm: Optional[float] = None,  # Gradient norm (optional)
    batch_size: Optional[int] = None,  # Batch size (optional)
    epoch: Optional[int] = None,  # Current epoch (optional)
    additional_metrics: Optional[Dict[str, Any]] = None,  # Additional metrics (optional)
):
    """Log training information in a structured and consistent format.
    
    Args:
        logger: Logger object
        step: Current training step
        train_loss: Training loss
        learning_rate: Current learning rate
        grad_norm: Gradient norm (optional)
        batch_size: Batch size (optional)
        epoch: Current epoch (optional)
        additional_metrics: Additional metrics to log (optional)
    """
    # Basic info
    info = {
        "step": step,
        "loss": f"{train_loss:.4f}",
        "lr": f"{learning_rate:.8f}"
    }
    
    # Add optional info
    if grad_norm is not None:
        info["grad_norm"] = f"{grad_norm:.4f}"
    
    if batch_size is not None:
        info["batch_size"] = batch_size
    
    if epoch is not None:
        info["epoch"] = epoch
    
    # Add additional metrics
    if additional_metrics:
        for k, v in additional_metrics.items():
            if isinstance(v, float):
                info[k] = f"{v:.4f}"
            else:
                info[k] = v
    
    # Log info
    logger.info(f"Training: {', '.join([f'{k}={v}' for k, v in info.items()])}")