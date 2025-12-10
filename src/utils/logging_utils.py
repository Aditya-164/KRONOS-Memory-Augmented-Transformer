import os
import sys
import logging
import json
from datetime import datetime
from pathlib import Path
import inspect
from typing import Dict, Any, Optional, Union
import torch
import numpy as np

"""Logging utilities for the Titans model."""

class Logger:
    """Centralized logging utility for the Titans project."""
    
    # Track configured loggers to avoid duplicate setup
    _configured_loggers = {}
    
    @staticmethod
    def setup_logger(
        name: str,
        level: Union[str, int] = logging.INFO,
        log_file: Optional[str] = None,
        format_type: str = "detailed",
        console_output: bool = True,
    ) -> logging.Logger:
        """Set up a logger with file and console handlers.
        
        Args:
            name: Logger name (typically __name__)
            level: Logging level (DEBUG, INFO, etc.)
            log_file: Path to log file. If None, only console logging
            format_type: Format type ("simple", "detailed", "json")
            console_output: Whether to output logs to console
            
        Returns:
            Configured logger
        """
        # Convert string level to int if needed
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        
        # Check if this logger is already configured
        if name in Logger._configured_loggers:
            return Logger._configured_loggers[name]
        
        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = False  # Don't propagate to parent loggers
        
        # Remove any existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Select formatter based on format type
        if format_type == "simple":
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        elif format_type == "json":
            class JsonFormatter(logging.Formatter):
                def format(self, record):
                    log_record = {
                        "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                        "level": record.levelname,
                        "logger": record.name,
                        "module": record.module,
                        "function": record.funcName,
                        "line": record.lineno,
                        "message": record.getMessage()
                    }
                    # Add extra data if available
                    if hasattr(record, 'extras'):
                        log_record.update(record.extras)
                    return json.dumps(log_record)
            formatter = JsonFormatter()
        else:  # detailed format
            formatter = logging.Formatter(
                '%(asctime)s.%(msecs)03d | %(levelname)-8s | '
                '%(name)s:%(funcName)s:%(lineno)d | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        # Add console handler if requested
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # Add file handler if log_file provided
        if log_file:
            # Create directory if needed
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # Store logger in configured loggers
        Logger._configured_loggers[name] = logger
        return logger
    
    @staticmethod
    def log_with_context(
        logger: logging.Logger, 
        level: int, 
        message: str, 
        extras: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a message with additional context information.
        
        Args:
            logger: Logger to use
            level: Logging level (e.g., logging.DEBUG)
            message: Log message
            extras: Additional context data for structured logging
        """
        extras = extras or {}
        
        # Get caller information
        frame = inspect.currentframe().f_back
        module = inspect.getmodule(frame)
        module_name = module.__name__ if module else "unknown"
        function_name = frame.f_code.co_name
        lineno = frame.f_lineno
        
        # Create record directly
        record = logger.makeRecord(
            logger.name, level, frame.f_code.co_filename, lineno,
            message, (), None, function_name
        )
        record.extras = extras
        
        # Process record through all handlers
        logger.handle(record)
    
    @staticmethod
    def log_tensor_info(
        logger: logging.Logger,
        level: int,
        name: str,
        tensor: torch.Tensor,
        include_data: bool = False,
        max_elements: int = 10
    ) -> None:
        """Log tensor information for debugging.
        
        Args:
            logger: Logger to use
            level: Logging level
            name: Tensor name or description
            tensor: Tensor to log
            include_data: Whether to include actual tensor values
            max_elements: Maximum number of elements to log
        """
        if not logger.isEnabledFor(level):
            return
            
        info = {
            "name": name,
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "min": float(tensor.min().item()) if tensor.numel() > 0 else None,
            "max": float(tensor.max().item()) if tensor.numel() > 0 else None,
            "mean": float(tensor.mean().item()) if tensor.numel() > 0 else None,
            "std": float(tensor.std().item()) if tensor.numel() > 0 and tensor.dtype.is_floating_point else None,
            "has_nan": bool(torch.isnan(tensor).any().item()),
            "has_inf": bool(torch.isinf(tensor).any().item()),
        }
        
        if include_data and tensor.numel() > 0:
            # Get a sample of tensor values
            flat = tensor.detach().cpu().flatten()
            if flat.numel() <= max_elements:
                sample = flat
            else:
                indices = torch.randperm(flat.numel())[:max_elements]
                sample = flat[indices]
            
            info["sample"] = sample.tolist()
        
        Logger.log_with_context(
            logger, level, f"Tensor info for {name}:", {"tensor_info": info}
        )
    
    @staticmethod
    def log_model_grad_stats(
        logger: logging.Logger,
        model: torch.nn.Module,
        level: int = logging.DEBUG,
        sample_size: int = 5
    ) -> None:
        """Log gradient statistics for model parameters.
        
        Args:
            logger: Logger to use
            model: PyTorch model
            level: Logging level
            sample_size: Number of parameters to sample for detailed logging
        """
        if not logger.isEnabledFor(level):
            return
            
        # Collect gradient stats
        grad_norms = []
        grad_means = []
        grad_maxes = []
        has_nan = False
        has_inf = False
        param_names = []
        
        # Collect all parameters with gradients
        params_with_grad = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                params_with_grad.append((name, param))
                
                grad = param.grad
                grad_norms.append(grad.norm().item())
                grad_means.append(grad.abs().mean().item())
                grad_maxes.append(grad.abs().max().item())
                
                has_nan = has_nan or torch.isnan(grad).any().item()
                has_inf = has_inf or torch.isinf(grad).any().item()
                param_names.append(name)
        
        # Compute overall statistics
        if grad_norms:
            avg_norm = sum(grad_norms) / len(grad_norms)
            max_norm = max(grad_norms)
            
            # Log overall stats
            Logger.log_with_context(
                logger, level, 
                f"Gradient stats: avg_norm={avg_norm:.6f}, max_norm={max_norm:.6f}, "
                f"has_nan={has_nan}, has_inf={has_inf}",
                {
                    "grad_stats": {
                        "avg_norm": avg_norm,
                        "max_norm": max_norm,
                        "has_nan": has_nan,
                        "has_inf": has_inf,
                        "param_count": len(grad_norms)
                    }
                }
            )
            
            # Log sample of individual parameter gradients
            if sample_size > 0 and params_with_grad:
                sample = params_with_grad[:sample_size] if sample_size < len(params_with_grad) else params_with_grad
                for name, param in sample:
                    Logger.log_tensor_info(
                        logger, level, f"Gradient for {name}", param.grad, include_data=False
                    )
        else:
            logger.log(level, "No gradients found in model")
    
    @staticmethod
    def log_memory_usage(logger: logging.Logger, level: int = logging.DEBUG, tag: str = "") -> None:
        """Log current memory usage (if torch.cuda is available).
        
        Args:
            logger: Logger to use
            level: Logging level
            tag: Optional tag to identify the measurement point
        """
        if not logger.isEnabledFor(level):
            return
            
        mem_info = {}
        
        # Get CPU memory info
        try:
            import psutil
            process = psutil.Process(os.getpid())
            cpu_mem = process.memory_info().rss / (1024 * 1024)  # MB
            mem_info["cpu_mem_mb"] = cpu_mem
        except ImportError:
            pass
        
        # Get GPU memory info if available
        if torch.cuda.is_available():
            try:
                for i in range(torch.cuda.device_count()):
                    mem_info[f"gpu{i}_allocated_mb"] = torch.cuda.memory_allocated(i) / (1024 * 1024)
                    mem_info[f"gpu{i}_reserved_mb"] = torch.cuda.memory_reserved(i) / (1024 * 1024)
                    mem_info[f"gpu{i}_max_mb"] = torch.cuda.max_memory_allocated(i) / (1024 * 1024)
            except:
                pass
        
        # Log memory information
        prefix = f"{tag}: " if tag else ""
        if mem_info:
            Logger.log_with_context(
                logger, level, 
                f"{prefix}Memory usage: " + ", ".join([f"{k}={v:.2f}" for k, v in mem_info.items()]),
                {"memory_info": mem_info}
            )
        else:
            logger.log(level, f"{prefix}Memory usage information not available.")


def get_logger(
    name: str,
    level: Union[str, int] = None,
    log_file: Optional[str] = None,
    format_type: Optional[str] = None,
) -> logging.Logger:
    """Get a configured logger.
    
    This is a convenience function that checks global configuration before
    setting up a logger with Logger.
    
    Args:
        name: Logger name
        level: Logging level (defaults to global setting if None)
        log_file: Log file path (defaults to global setting if None)
        format_type: Format type (defaults to global setting if None)
        
    Returns:
        Configured logger
    """
    # Get global settings from environment or use defaults
    global_level = os.environ.get("LOG_LEVEL", "INFO")
    global_format = os.environ.get("LOG_FORMAT", "detailed")
    global_log_file = os.environ.get("LOG_FILE", None)
    
    # Use provided values or fallback to globals
    level = level or global_level
    format_type = format_type or global_format
    log_file = log_file or global_log_file
    
    return Logger.setup_logger(
        name=name,
        level=level,
        log_file=log_file,
        format_type=format_type
    )


def log_step_info(
    logger: logging.Logger, 
    step: int, 
    metrics: Dict[str, Any], 
    level: int = logging.INFO
) -> None:
    """Log step information during training or evaluation.
    
    Args:
        logger: Logger to use
        step: Current step number
        metrics: Metrics dictionary
        level: Logging level
    """
    # Format metrics for clean output
    metrics_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                           for k, v in metrics.items()])
    
    logger.log(level, f"Step {step}: {metrics_str}")
    
    # Add detailed metrics as extras for structured logging
    Logger.log_with_context(
        logger, level, f"Step {step} metrics", 
        {"step": step, "metrics": metrics}
    )


def log_model_info(
    logger: logging.Logger,
    model: torch.nn.Module,
    level: int = logging.DEBUG
) -> None:
    """Log model architecture and parameter information.
    
    Args:
        logger: Logger to use
        model: PyTorch model
        level: Logging level
    """
    if not logger.isEnabledFor(level):
        return
        
    # Count parameters
    total_params = 0
    trainable_params = 0
    non_trainable_params = 0
    
    # Collect parameter statistics by layer
    param_stats = {}
    
    for name, param in model.named_parameters():
        param_size = param.numel()
        total_params += param_size
        
        if param.requires_grad:
            trainable_params += param_size
        else:
            non_trainable_params += param_size
            
        # Extract layer name (everything before the last dot)
        layer_name = name.rsplit(".", 1)[0] if "." in name else name
        
        if layer_name not in param_stats:
            param_stats[layer_name] = {"trainable": 0, "non_trainable": 0}
            
        if param.requires_grad:
            param_stats[layer_name]["trainable"] += param_size
        else:
            param_stats[layer_name]["non_trainable"] += param_size
    
    # Log overall statistics
    Logger.log_with_context(
        logger, level,
        f"Model has {total_params:,} parameters "
        f"({trainable_params:,} trainable, {non_trainable_params:,} non-trainable)",
        {
            "model_stats": {
                "total_params": total_params,
                "trainable_params": trainable_params,
                "non_trainable_params": non_trainable_params,
                "param_stats": param_stats
            }
        }
    )
    
    # Log model architecture
    model_str = str(model)
    Logger.log_with_context(
        logger, level,
        f"Model architecture:\n{model_str}",
        {"model_architecture": model_str}
    )


def debug_tensor(
    tensor: torch.Tensor, 
    name: str = "tensor", 
    level: int = logging.DEBUG,
    logger: Optional[logging.Logger] = None
) -> torch.Tensor:
    """Debug a tensor by logging its stats and returning it unchanged.
    
    This can be inserted directly into computation: x = debug_tensor(x, "after_layer1")
    
    Args:
        tensor: Input tensor
        name: Name for debugging
        level: Logging level
        logger: Logger to use (creates a new one if None)
        
    Returns:
        Input tensor unchanged
    """
    if logger is None:
        logger = get_logger(__name__)
        
    if logger.isEnabledFor(level):
        Logger.log_tensor_info(logger, level, name, tensor)
        
    return tensor


def log_memory_usage(
    logger: logging.Logger, 
    level: int = logging.DEBUG, 
    tag: str = ""
) -> None:
    """Log current memory usage.
    
    Args:
        logger: Logger to use
        level: Logging level
        tag: Optional tag to identify the measurement point
    """
    Logger.log_memory_usage(logger, level, tag)