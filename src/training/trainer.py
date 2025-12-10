import os
import time
import random
import math
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union, Callable
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from src.models import TitansLM, KRONOSLM, VanillaLM  # add KRONOSLM here
from src.data.dataloader import get_dataloader, CollatorForLanguageModeling
from src.config import ModelConfig, TrainingConfig
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from src.utils import (
    get_logger, 
    Logger, 
    log_step_info,
    log_model_info,
    log_memory_usage
)
import torch.cuda.amp as amp
import torch.nn.functional as F

from src.training.utils import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_one_cycle_schedule,
    LinearWarmupCosineLRScheduler,
    get_grad_norm,
    log_training_info,
    torch_seed_all
)

from src.training.metrics import (
    LanguageModelMetrics, 
    MemoryMetrics, 
    GenerationMetrics, 
    MetricsTracker,
    compute_metrics_from_batch
)

device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False

logger = get_logger(__name__)

# Strategy interface and implementations

class BaseTrainingStrategy:
    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        raise NotImplementedError()

class DefaultTrainingStrategy(BaseTrainingStrategy):
    def __init__(self, trainer: "Trainer"):
        self.trainer = trainer

    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        # Delegate to the original implementation
        return self.trainer._default_training_step(batch)

class CoconutTrainingStrategy(BaseTrainingStrategy):
    """
    Implements training for KRONOSLM models using the Coconut scheme.
    Supports either annotated reasoning spans or a hybrid heuristic + random span selection when annotations are absent.
    Expects TrainingConfig to provide:
      - coconut_reasoning_start: Optional[int]
      - coconut_reasoning_end: Optional[int]
      - bot_token_id: int
      - eot_token_id: int
      - continuous_steps: int
      - use_annotated_spans: bool
      - hybrid_random_ratio: float      # fraction of batches to use random spans
      - random_span_length: Optional[int] # length of random span; defaults to continuous_steps
    """
    def __init__(self, trainer: "Trainer"):
        self.trainer = trainer
        self.cfg = trainer.config

    def convert_language_to_continuous(self, sequence: torch.Tensor, start: int, end: int) -> torch.Tensor:
        batch_size, _ = sequence.shape
        dev = sequence.device
        bot_id = self.cfg.bot_token_id
        eot_id = self.cfg.eot_token_id
        # Replace the reasoning span with <bot> and <eot>
        return torch.cat([
            sequence[:, :start],
            torch.full((batch_size, 1), bot_id, device=dev),
            torch.full((batch_size, 1), eot_id, device=dev),
            sequence[:, end+1:]
        ], dim=1)

    def select_span(self, seq_len: int) -> Tuple[int, int]:
        """
        Return (start, end) of reasoning span under the hybrid scheme:
        - If annotations enabled, use annotated values
        - Else: with probability (1 - random_ratio) use last-N heuristic,
                with probability random_ratio sample a random fixed-length span.
        """
        # annotated path
        if self.cfg.use_annotated_spans:
            return self.cfg.coconut_reasoning_start, self.cfg.coconut_reasoning_end

        # unannotated: hybrid heuristic + random
        M = getattr(self.cfg, 'random_span_length', self.cfg.continuous_steps)
        max_start = seq_len - M - 1
        # decide heuristic or random
        r = random.random()
        if r < (1.0 - self.cfg.hybrid_random_ratio):
            # last-N heuristic
            start = seq_len - M - 1
        else:
            # random fixed-length
            start = random.randint(0, max_start)
        end = start + M - 1
        return start, end

    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        t = self.trainer
        t.optimizer.zero_grad()

        # Move to device
        input_ids = batch['input_ids'].to(t.device)
        labels    = batch['labels'].to(t.device)
        seq_len   = input_ids.size(1)

        # Select span
        start, end = self.select_span(seq_len)

        # Convert to continuous-thought sequence
        coconut_input = self.convert_language_to_continuous(input_ids, start, end)

        # Forward with continuous thought steps
        logits, _ = t.model(
            coconut_input,
            continuous_thought_steps=self.cfg.continuous_steps
        )

        # Compute loss on tokens after thought markers
        post_start = start + 2  # account for <bot> and <eot>
        post_labels = labels[:, end+1:]
        post_logits = logits[:, post_start:post_start + post_labels.size(1), :]
        loss = F.cross_entropy(
            post_logits.reshape(-1, post_logits.size(-1)),
            post_labels.reshape(-1)
        )

        # Backward & step
        loss.backward(retain_graph=True)
        t.optimizer.step()

        # Compute and return full metrics for post-thought tokens
        metrics_dict = compute_metrics_from_batch(
            post_logits, post_labels,
            memory_states=None,
            prefix="", label_smoothing=self.trainer.label_smoothing
        )
        return metrics_dict
class Trainer:
    """Unified Trainer for different model types."""

    def __init__(
        self,
        model: Union[VanillaLM, TitansLM, KRONOSLM],
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[Optimizer] = None,
        lr_scheduler: Optional[LambdaLR] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        config: Optional[TrainingConfig] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        log_dir: Optional[Union[str, Path]] = None,
        fp16: bool = False,
        gradient_accumulation: int = 1,
        max_grad_norm: float = 1.0,
        log_every: int = 10,
        label_smoothing: float = 0.1,
        sliding_window: bool = False,  # New argument for sliding window
    ):
        # store constructor args
        self.train_dataloader = train_dataloader
        self.val_dataloader   = val_dataloader
        self.optimizer        = optimizer
        self.lr_scheduler     = lr_scheduler
        self.config           = config
        self.checkpoint_dir   = checkpoint_dir
        self.log_dir          = log_dir
        # store model and execution device
        self.model = model
        self.device = device
        # store new args
        self.fp16 = fp16
        self.grad_accum = gradient_accumulation
        self.sliding_window = sliding_window
        # Use the updated GradScaler constructor with device_type parameter
        self.scaler = GradScaler(device=device) if fp16 else None

        self.max_grad_norm = max_grad_norm
        self.log_every = log_every
        self.label_smoothing = label_smoothing  # Initialize label smoothing attribute

        self.global_step = 0
        self.best_val_loss = float('inf')
        self.epochs_trained = 0
        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()

        # move model onto device
        self.model.to(self.device)

        self.logger = logger

        try:
            self.writer = SummaryWriter(log_dir=self.log_dir)
            self.has_tensorboard = True
        except ImportError:
            self.has_tensorboard = False
            self.logger.warning("TensorBoard not available. Install with pip install tensorboard")

        self.logger.info(f"Trainer initialized with device={self.device}, fp16={self.fp16}, "
                         f"grad_accum={self.grad_accum}, sliding_window={self.sliding_window}")
        self.logger.debug(f"Max gradient norm: {self.max_grad_norm}")
        log_model_info(self.logger, self.model)
        log_memory_usage(self.logger, tag="Initial")

        # Select a training strategy based on model type
        if isinstance(model, KRONOSLM):
            self.strategy = CoconutTrainingStrategy(self)
            self.logger.info("Using Coconut training strategy for KRONOSLM model.")
        else:
            self.strategy = DefaultTrainingStrategy(self)
            self.logger.info("Using default training strategy.")

    @staticmethod
    def get_lr_scheduler(
        optimizer: Optimizer,
        warmup_steps: int = 10000,
        max_steps: int = 100000,
        min_lr_ratio: float = 0.1
    ) -> LambdaLR:
        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
            return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return LambdaLR(optimizer, lr_lambda)

    def save_checkpoint(self, name: str = "checkpoint", save_optimizer: bool = True) -> str:
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{name}_{self.global_step}.pt")
        self.logger.debug(f"Saving checkpoint to {checkpoint_path}")
        checkpoint_data = {
            "model_state_dict": self.model.state_dict(),
            "global_step": self.global_step,
            "epochs_trained": self.epochs_trained,
            "best_val_loss": self.best_val_loss,
        }
        if save_optimizer and self.optimizer is not None:
            checkpoint_data["optimizer_state_dict"] = self.optimizer.state_dict()
        if save_optimizer and self.lr_scheduler is not None:
            checkpoint_data["scheduler_state_dict"] = self.lr_scheduler.state_dict()
        torch.save(checkpoint_data, checkpoint_path)
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True) -> None:
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.global_step = checkpoint.get("global_step", 0)
        self.epochs_trained = checkpoint.get("epochs_trained", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        if load_optimizer and "optimizer_state_dict" in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if load_optimizer and "scheduler_state_dict" in checkpoint and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.logger.info(f"Loaded checkpoint from step {self.global_step}")
        log_memory_usage(self.logger, tag="After checkpoint load")

    def train(self, max_steps: Optional[int] = None, max_epochs: Optional[int] = None,
              eval_every: Optional[int] = None, save_every: Optional[int] = None) -> Dict[str, float]:
        max_steps = max_steps or self.config.max_steps
        eval_every = eval_every or self.config.eval_every
        save_every = save_every or self.config.save_every
        self.model.train()
        self.train_metrics.reset()
        total_batches = len(self.train_dataloader)
        epoch = self.epochs_trained
        step = self.global_step
        self.logger.info(f"Starting training at step {step}, epoch {epoch}")
        log_memory_usage(self.logger, tag="Training start")
        try:
            patience = 5
            best_val_loss = float("inf")
            no_improve_count = 0

            while True:
                epoch += 1
                if max_epochs is not None and epoch > max_epochs:
                    self.logger.info(f"Reached maximum epochs: {max_epochs}")
                    break
                epoch_start_time = time.time()
                self.logger.debug(f"Starting epoch {epoch}")

                for batch_idx, batch in enumerate(self.train_dataloader):
                    step += 1
                    if max_steps and step > max_steps:
                        self.logger.info(f"Reached maximum steps: {max_steps}")
                        break

                    # Delegate training step to the chosen strategy.
                    batch_metrics = self.strategy.training_step(batch)
                    # Update training metrics tracker
                    batch_size = batch['input_ids'].size(0)
                    self.train_metrics.update(batch_metrics, batch_size=batch_size)

                    if step % 50 == 0:
                        self.logger.debug(f"Detailed model state at step {step}")
                        Logger.log_model_grad_stats(self.logger, self.model)

                    if step % self.log_every == 0:
                        lr = self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler else self.config.learning_rate
                        log_output = f"Step {step} | Loss: {batch_metrics['loss']:.4f}"
                        if "perplexity" in batch_metrics:
                            log_output += f" | PPL: {batch_metrics['perplexity']:.4f}"
                        log_output += f" | LR: {lr:.2e}"
                        self.logger.info(log_output)
                        log_step_info(self.logger, step, {**batch_metrics, "lr": lr,
                                                          "epoch": epoch,
                                                          "batch": batch_idx,
                                                          "progress": f"{batch_idx}/{total_batches}"})
                        if self.has_tensorboard:
                            for name, value in batch_metrics.items():
                                self.writer.add_scalar(f"train/{name}", value, step)
                            self.writer.add_scalar("train/learning_rate", lr, step)
                            if torch.cuda.is_available():
                                self.writer.add_scalar("system/gpu_memory_allocated", 
                                                       torch.cuda.memory_allocated() / (1024 * 1024), 
                                                       step)

                    if eval_every > 0 and step % eval_every == 0 and self.val_dataloader is not None:
                        self.logger.debug(f"Starting evaluation at step {step}")
                        self.save_checkpoint(name="temp_model")
                        eval_metrics = self.evaluate()
                        val_loss = eval_metrics["loss"]
                        self.logger.info(f"Evaluation at step {step} | Loss: {val_loss:.4f} | PPL: {eval_metrics['perplexity']:.4f}")
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            no_improve_count = 0
                            self.save_checkpoint(name="best_model")
                            self.logger.info(f"New best validation loss: {best_val_loss:.4f}")
                        else:
                            no_improve_count += 1
                            self.logger.info(f"Validation loss did not improve for {no_improve_count}/{patience} evaluations.")
                        if self.has_tensorboard:
                            for name, value in eval_metrics.items():
                                self.writer.add_scalar(f"eval/{name}", value, step)
                        if no_improve_count >= patience:
                            self.logger.info("Early stopping triggered due to overfitting.")
                            break
                        self.model.train()
                        log_memory_usage(self.logger, tag="After evaluation")
                    if save_every > 0 and step % save_every == 0:
                        self.save_checkpoint()
                if no_improve_count >= patience:
                    break
                epoch_time = time.time() - epoch_start_time
                self.logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s | Steps: {step} | Loss: {self.train_metrics.get_average('loss'):.4f}")
                self.epochs_trained = epoch
                self.global_step = step
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Error during training: {e}", exc_info=True)
            try:
                self.save_checkpoint(name="error_recovery")
                self.logger.info("Saved recovery checkpoint")
            except:
                self.logger.error("Failed to save recovery checkpoint", exc_info=True)
            raise
        self.save_checkpoint(name="final")
        if self.val_dataloader is not None:
            self.logger.info("Running final evaluation...")
            final_metrics = self.evaluate()
            self.logger.info(f"Final evaluation | Loss: {final_metrics['loss']:.4f} | PPL: {final_metrics['perplexity']:.4f}")
        else:
            final_metrics = self.train_metrics.get_all_averages()
        if self.has_tensorboard:
            self.writer.close()
        self.logger.info("Training completed")
        log_memory_usage(self.logger, tag="Training end")
        return final_metrics

    def _default_training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        batch = {k: v.to(self.device) for k, v in batch.items()}
        # … zero grad / fp16 boilerplate …

        inputs = batch["input_ids"]
        targets = batch.get("labels", inputs.clone())
        # mask pad
        if "labels" not in batch:
            targets[targets == self.model.transformer.token_emb.padding_idx] = -100

        # Mixed precision forward: compute logits and loss tensor via language metrics
        with autocast(device_type=self.device, enabled=self.fp16):
            logits, _ = self.model(inputs, return_state=True)
            # Compute loss using shift-based LanguageModelMetrics with label smoothing
            loss = LanguageModelMetrics.compute_loss(
                logits,
                targets,
                ignore_index=-100,
                label_smoothing=self.label_smoothing
            ) / self.grad_accum

        # Compute full batch metrics from logits and targets, fallback to include all keys on error
        try:
            metrics_dict = compute_metrics_from_batch(
                logits, targets, memory_states=None,
                prefix="", label_smoothing=self.label_smoothing
            )
        except Exception as e:
            self.logger.warning(f"compute_metrics_from_batch failed: {e}")
            # Fallback values
            ppl_val = math.exp(loss.item()) if loss.item() < 20 else float('inf')
            metrics_dict = {
                'loss': loss.item(),
                'perplexity': ppl_val,
                'accuracy': 0.0,
                'total_tokens': 0,
                'token_entropy': 0.0
            }

        # backward pass on loss tensor
        if self.fp16:
            self.scaler.scale(loss).backward(retain_graph=True)
        else:
            loss.backward(retain_graph=True)

        # step & zero_grad every grad_accum steps
        self.global_step += 1
        if self.global_step % self.grad_accum == 0:
            if self.fp16:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()
            self.optimizer.zero_grad()
            # advance the configured scheduler
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # free up any fragmentation
        torch.cuda.empty_cache()

        return metrics_dict

    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        # Delegate the training step to the chosen strategy.
        return self.strategy.training_step(batch)

    def evaluate(self, dataloader: Optional[DataLoader] = None) -> Dict[str, float]:
        dataloader = dataloader or self.val_dataloader
        if dataloader is None:
            self.logger.warning("No validation dataloader provided for evaluation")
            return {}
        self.model.eval()
        self.val_metrics.reset()
        self.logger.info("Starting evaluation")
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            inputs = batch["input_ids"]
            targets = batch["labels"]
            if "labels" not in batch:
                targets = inputs.clone()
                if hasattr(self.model, 'pad_token_id'):
                    targets[targets == self.model.pad_token_id] = -100
            # Handle models that return (logits, state)
            if isinstance(self.model, (TitansLM, KRONOSLM)):
                logits, state = self.model(inputs, return_state=True)
                memory_states = state.memory_states if hasattr(state, "memory_states") else None
            else:
                logits = self.model(inputs)
                memory_states = None
            metrics = compute_metrics_from_batch(logits, targets, memory_states, prefix="", label_smoothing=self.label_smoothing)
            if batch_idx % 20 == 0:
                self.logger.debug(f"Evaluation batch {batch_idx}: " + 
                                  ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()]))
            batch_size = inputs.size(0)
            self.val_metrics.update(metrics, batch_size=batch_size)
        avg_metrics = self.val_metrics.get_all_averages()
        self.logger.debug("Evaluation metrics: " + ", ".join([f"{k}={v:.4f}" for k, v in avg_metrics.items()]))
        return avg_metrics

    def generate(self, input_ids: torch.Tensor, max_length: int = 100, temperature: float = 1.0,
                 top_k: Optional[int] = None, top_p: Optional[float] = None, return_logits: bool = False
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        self.model.eval()
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids)
        input_ids = input_ids.to(self.device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        self.logger.debug(f"Generating text with input shape: {input_ids.shape}, max_length: {max_length}, temperature: {temperature}")
        if isinstance(self.model, TitansLM):
            return self.model.generate(input_ids, max_length=max_length, temperature=temperature,
                                       top_k=top_k, top_p=top_p, return_logits=return_logits)
        else:
            generated = input_ids
            all_logits = []
            state = None
            for _ in range(max_length):
                if hasattr(self.model, "forward_with_state"):
                    logits, state = self.model.forward_with_state(generated, state=state, return_state=True)
                else:
                    logits = self.model(generated)
                next_token_logits = logits[:, -1, :]
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('Inf')
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                if return_logits:
                    all_logits.append(logits)
                generated = torch.cat((generated, next_token), dim=1)
            if return_logits:
                all_logits = torch.cat(all_logits, dim=1)
                return generated, all_logits
            return generated, None

class DistributedTrainer(Trainer):
    """Distributed trainer for multi-GPU training."""
    
    def __init__(
        self,
        model: Union[KRONOSLM, TitansLM, VanillaLM],
        train_dataloader: DataLoader,
        local_rank: int,
        world_size: int,
        **kwargs
    ):
        """Initialize distributed trainer."""
        super().__init__(model=model, train_dataloader=train_dataloader, **kwargs)
        
        self.local_rank = local_rank
        self.world_size = world_size
        
        # Wrap model in DistributedDataParallel
        self.model = DistributedDataParallel(
            self.model, 
            device_ids=[local_rank],
            output_device=local_rank
        )
        
        # Only log from rank 0
        if local_rank != 0:
            self.logger.setLevel(logging.ERROR)
            self.has_tensorboard = False
        
        self.logger.debug(f"Initialized DistributedTrainer on rank {local_rank}/{world_size}")
    
    def save_checkpoint(self, name: str = "checkpoint", save_optimizer: bool = True) -> str:
        """Save a model checkpoint from the main process only."""
        if self.local_rank == 0:
            return super().save_checkpoint(name, save_optimizer)
        return ""
    
    def evaluate(self, dataloader: Optional[DataLoader] = None) -> Dict[str, float]:
        """Evaluate the model on all processes and gather results."""
        local_metrics = super().evaluate(dataloader)
        
        # Gather metrics from all processes
        if self.world_size > 1:
            import torch.distributed as dist
            
            metrics_tensor = torch.zeros(len(local_metrics), device=self.device)
            metrics_keys = list(local_metrics.keys())
            
            for i, key in enumerate(metrics_keys):
                metrics_tensor[i] = torch.tensor(local_metrics[key], device=self.device)
            
            # Average metrics across processes
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
            metrics_tensor /= self.world_size
            
            # Update local metrics with gathered values
            for i, key in enumerate(metrics_keys):
                local_metrics[key] = metrics_tensor[i].item()
                
            self.logger.debug(f"Gathered evaluation metrics from {self.world_size} processes")
        
        return local_metrics
