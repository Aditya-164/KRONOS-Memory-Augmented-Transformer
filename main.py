import torch, random, argparse, os, platform, sys, numpy as np
from pathlib import Path

from src import (
    DefaultConfig,
    Trainer,
    WikiTextDataset,
    TextDataset,
    CollatorForLanguageModeling,
    get_dataloader,
    prepare_datasets,
    plot_training_metrics,
    init_weights,
    log_model_info,
)
from src.utils import get_logger, log_memory_usage
from src.models.transformer_arch.kronos_transformer import KRONOSTransformer, KRONOSLM
from src.models.transformer_arch.titans_transformer import TitansTransformer, TitansLM
from src.models.transformer_arch.vanilla_transformer import VanillaTransformer, VanillaLM
from transformers import AutoTokenizer


# Will be configured after parsing arguments
logger = None


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if logger:
        logger.debug(f"Set random seed to {seed}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train and evaluate Titans models")
    
    # Model configuration
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--model_dim", type=int, default=512, help="Model dimension")
    parser.add_argument("--depth", type=int, default=12, help="Number of layers")
    parser.add_argument("--heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--dim_head", type=int, default=64, help="Dimension per attention head")
    parser.add_argument("--segment_len", type=int, default=512, help="Length of attention segments")
    parser.add_argument("--use_memory", action="store_true", help="Use neural memory in the model")
    parser.add_argument("--sliding_window", action="store_true", help="Use sliding‐window segmented attention to save memory")
    
    # Data configuration
    parser.add_argument("--data_path", type=str, default=None, help="Path to data file/directory")
    parser.add_argument("--dataset", type=str, choices=["text", "wikitext"], default="text", 
                        help="Dataset type")
    parser.add_argument("--wikitext_version", type=str, choices=["2", "103"], default="103",
                        help="WikiText version")
    parser.add_argument(
        "--tokenizer", type=str, default="bert-base-uncased",
        help="HuggingFace tokenizer to use (always BERT)"
    )
    parser.add_argument("--seq_length", type=int, default=1024, 
                        help="Training sequence length")
    parser.add_argument("--val_seq_length", type=int, default=1024, 
                        help="Validation sequence length")
    
    # Training configuration
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--val_batch_size", type=int, default=4, help="Validation batch size")
    parser.add_argument("--val_split",    type=float, default=0.1, help="Fraction of data to hold out for validation")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=10000, help="Warmup steps")
    parser.add_argument("--max_steps", type=int, default=100000, help="Maximum training steps")
    parser.add_argument("--max_epochs", type=int, default=100000, help="Maximum training epochs")
    parser.add_argument("--eval_every", type=int, default=100, 
                        help="Evaluate every N steps")
    parser.add_argument("--save_every", type=int, default=100, 
                        help="Save checkpoint every N steps")
    parser.add_argument("--gradient_accumulation", type=int, default=1, 
                        help="Gradient accumulation steps")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable mixed‐precision (auto‐cast) training"
    )
    
    # Output directories
    parser.add_argument("--output_dir", type=str, default="./outputs", 
                        help="Output directory")
    parser.add_argument("--checkpoint_dir", type=str, default=None, 
                        help="Checkpoint directory")
    parser.add_argument("--log_dir", type=str, default=None, 
                        help="Log directory")
    
    # Execution modes
    parser.add_argument("--mode", type=str, choices=["train", "eval", "generate"], 
                        default="train", help="Execution mode")
    parser.add_argument("--checkpoint", type=str, default=None, 
                        help="Checkpoint to load for evaluation or generation")
    parser.add_argument("--prompt", type=str, default=None, 
                        help="Text prompt for generation")
    parser.add_argument("--max_length", type=int, default=100, 
                        help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=1.0, 
                        help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, 
                        help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.9, 
                        help="Nucleus sampling parameter")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--log_file", type=str, default=None, 
                        help="Path to log file (if not specified, logs to console only)")
    parser.add_argument("--log_format", type=str, choices=["simple", "detailed", "json"], 
                        default="detailed", help="Log format")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["kronos","titans","vanilla"],
        default="titans",
        help="Which model to train/eval: kronos enables Coconut (use_annotated_spans=True), titans (annotated=False), vanilla."
    )

    return parser.parse_args()


def setup_logging(args):
    """Set up logging based on command-line arguments."""
    global logger
    
    # Set environment variables for global logging configuration
    os.environ["TITANS_LOG_LEVEL"] = "DEBUG" if args.debug else "INFO"
    os.environ["TITANS_LOG_FORMAT"] = args.log_format
    
    if args.log_file:
        log_file = args.log_file
    elif args.log_dir:
        # Create log directory if it doesn't exist
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate log file name based on current time and mode
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"titans_{args.mode}_{timestamp}.log"
    else:
        log_file = None
        
    if log_file:
        os.environ["TITANS_LOG_FILE"] = str(log_file)
    
    # Create main logger
    level = "DEBUG" if args.debug else "INFO"
    logger = get_logger("titans.main", level=level, log_file=log_file, format_type=args.log_format)
    
    # Log startup information
    logger.info(f"Starting Titans in {args.mode} mode")
    logger.debug(f"Command-line arguments: {args}")
    
    if args.debug:
        logger.debug("Debug mode enabled")
    
    # Log system information
    import platform
    system_info = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda if torch.cuda.is_available() else "Not available",
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "platform": platform.platform(),
    }
    
    logger.info(f"System: Python {system_info['python']}, PyTorch {system_info['pytorch']}, "
                f"Device: {system_info['device']}")
    logger.debug(f"Full system info: {system_info}")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            logger.debug(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    return logger


def create_model(args, config, vocab_size):
    """
    Instantiate one of {Kronos, Titans, Vanilla} based on args.model_type.
    Also flips config.training.use_annotated_spans for Kronos vs Titans.
    """
    mt = args.model_type
    logger.info(f"Creating `{mt}` model: {config.model.depth} layers, dim={config.model.dim}")
    if mt == "kronos":
        # Kronos: Coconut strategy → annotated spans ON
        config.training.use_annotated_spans = True
        # ensure we have valid start/end for Coconut spans
        if config.training.coconut_reasoning_start is None:
            config.training.coconut_reasoning_start = 0
        if config.training.coconut_reasoning_end is None:
            config.training.coconut_reasoning_end = config.training.continuous_steps
        transformer = KRONOSTransformer(
            dim=config.model.dim,
            depth=config.model.depth,
            vocab_size=vocab_size,
            seq_len=args.seq_length,
            dim_head=config.model.dim_head,
            heads=config.model.heads,
            segment_len=config.model.segment_len,
            use_neural_memory=args.use_memory,
            use_persistent_memory=args.use_memory,
            sliding=args.sliding_window,
            num_longterm_mem_tokens=config.model.num_longterm_mem_tokens,
            num_persist_mem_tokens=config.model.num_persist_mem_tokens,
            use_flex_attn=config.model.use_flex_attn,
            accept_value_residual=config.model.neural_memory_add_value_residual,
            neural_memory_kwargs={'adaptive_forgetting': True}, 
            persistent_memory_kwargs={'adaptive_forgetting': True}, 
            use_continuous_thought=True,
            bot_token_id=config.training.bot_token_id,
            eot_token_id=config.training.eot_token_id
        )
        transformer.apply(lambda m: init_weights(m))
        model = KRONOSLM(transformer=transformer,
                         temperature=args.temperature,
                         top_k=args.top_k,
                         top_p=args.top_p)

    elif mt == "vanilla":
        # Vanilla baseline → no annotated spans
        config.training.use_annotated_spans = False
        transformer = VanillaTransformer(
            dim=config.model.dim,
            depth=config.model.depth,
            vocab_size=vocab_size,
            seq_len=args.seq_length
        )
        transformer.apply(lambda m: init_weights(m))
        model = VanillaLM(transformer=transformer,
                          temperature=args.temperature,
                          top_k=args.top_k,
                          top_p=args.top_p)

    else:  # "titans"
        # Titans → no Coconut annotations
        config.training.use_annotated_spans = False
        transformer = TitansTransformer(
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
            accept_value_residual=config.model.neural_memory_add_value_residual,
            memory_kwargs={'adaptive_forgetting': False}  # disable adaptive forgetting for Titans
        )
        transformer.apply(lambda m: init_weights(m))
        model = TitansLM(transformer=transformer,
                         temperature=args.temperature,
                         top_k=args.top_k,
                         top_p=args.top_p)

    log_model_info(logger, model)
    return model


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


def train(args, config):
    # sliding window is only supported for segmented attention models
    if args.model_type == 'vanilla' and args.sliding_window:
        logger.warning("Sliding window mode is unsupported for vanilla model; disabling sliding_window flag.")
        args.sliding_window = False
    #  """Train a Titans model."""
     # Set random seed
        set_seed(args.seed)    
    
    # Load tokenizer
    tokenizer = load_or_create_tokenizer(args.tokenizer)
    
    # Create output directories
    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else output_dir / "checkpoints"
    log_dir = Path(args.log_dir) if args.log_dir else output_dir / "logs"
    
    for directory in [output_dir, checkpoint_dir, log_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Prepare dataset
    logger.info("Preparing datasets...")
    if args.dataset == "wikitext":
        train_dataset = WikiTextDataset(
            data_dir=args.data_path,
            split="train",
            tokenizer=tokenizer,
            seq_length=args.seq_length,
            version=args.wikitext_version
        )
        
        val_dataset = WikiTextDataset(
            data_dir=args.data_path,
            split="valid",
            tokenizer=tokenizer,
            seq_length=args.val_seq_length,
            version=args.wikitext_version
        )
    else:
        train_dataset, val_dataset, _ = prepare_datasets(
            data_path=args.data_path,
            tokenizer=tokenizer,
            train_seq_length=args.seq_length,
            val_seq_length=args.val_seq_length,
            val_split=args.val_split,
            test_split=0.0,
            seed=args.seed
        )
    
    logger.info(f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}")
    
    # Create dataloaders
    collator = CollatorForLanguageModeling(tokenizer)
    
    train_dataloader = get_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=config.training.num_workers,
        drop_last=True
    )
    
    if len(val_dataset) > 0:
        val_dataloader = get_dataloader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=config.training.num_workers,
            drop_last=False
        )
    else:
        val_dataloader = None
    
    # Create model
    logger.info("Creating model...")
    vocab_size = len(tokenizer)
    model = create_model(args, config, vocab_size)
    
    # Track memory usage
    log_memory_usage(logger, tag="After model creation")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        fp16=args.fp16,
        sliding_window=args.sliding_window,
        gradient_accumulation=args.gradient_accumulation,
        max_grad_norm=config.training.gradient_clip_val,
        config=config.training  # pass training config for correct LR schedule
    )
    
    # Load checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # Train the model
    logger.info("Starting training...")
    trainer.train(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        save_every=args.save_every
    )
    
    # Final evaluation
    logger.info("Training completed. Running final evaluation...")
    eval_metrics = trainer.evaluate()
    
    # Plot training metrics
    metrics_path = output_dir / "training_metrics.png"
    logger.info(f"Saving training metrics plot to {metrics_path}")
    plot_training_metrics(
        trainer.train_metrics.metrics,
        title="Training Metrics",
        model_name=args.model_type,  # Add model name to the plot title
        save_path=str(metrics_path),
        rolling_window=100
    )
    
    # Log final metrics
    logger.info("Final evaluation metrics:")
    for name, value in eval_metrics.items():
        logger.info(f"  {name}: {value:.4f}")
    
    return eval_metrics


def evaluate(args, config):
    """Evaluate a trained Titans model."""
    # Set random seed
    set_seed(args.seed)
    
    # Load tokenizer
    tokenizer = load_or_create_tokenizer(args.tokenizer)
    
    # Prepare dataset
    logger.info("Preparing evaluation dataset...")
    if args.dataset == "wikitext":
        eval_dataset = WikiTextDataset(
            data_dir=args.data_path,
            split="test",
            tokenizer=tokenizer,
            seq_length=args.val_seq_length,
            version=args.wikitext_version
        )
    else:
        _, _, eval_dataset = prepare_datasets(
            data_path=args.data_path,
            tokenizer=tokenizer,
            train_seq_length=args.seq_length,
            val_seq_length=args.val_seq_length,
            val_split=0.0,
            test_split=0.1,
            seed=args.seed
        )
        
        if eval_dataset is None:
            _, eval_dataset, _ = prepare_datasets(
                data_path=args.data_path,
                tokenizer=tokenizer,
                train_seq_length=args.seq_length,
                val_seq_length=args.val_seq_length,
                val_split=0.1,
                test_split=0.0,
                seed=args.seed
            )
    
    logger.info(f"Evaluation dataset size: {len(eval_dataset)}")
    
    # Create dataloader
    collator = CollatorForLanguageModeling(
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    )
    
    eval_dataloader = get_dataloader(
        eval_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=config.training.num_workers,
        drop_last=False
    )
    
    # Create model
    logger.info("Creating model...")
    vocab_size = len(tokenizer)
    model = create_model(args, config, vocab_size)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=None,
        val_dataloader=eval_dataloader,
        checkpoint_dir=args.checkpoint_dir,
    )
    
    # Load checkpoint
    if not args.checkpoint:
        logger.error("Checkpoint must be provided for evaluation")
        raise ValueError("Checkpoint must be provided for evaluation")
    
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    trainer.load_checkpoint(args.checkpoint)
    
    # Run evaluation
    logger.info("Running evaluation...")
    eval_metrics = trainer.evaluate()
    
    # Print results
    logger.info(f"Evaluation results:")
    for name, value in eval_metrics.items():
        logger.info(f"  {name}: {value:.4f}")
    
    return eval_metrics


def generate(args, config):
    """Generate text with a trained Titans model."""
    # Set random seed
    set_seed(args.seed)
    
    # Load tokenizer
    tokenizer = load_or_create_tokenizer(args.tokenizer)
    
    # Create model
    logger.info("Creating model...")
    vocab_size = len(tokenizer)
    model = create_model(args, config, vocab_size)
    
    # Create trainer for model loading
    trainer = Trainer(
        model=model,
        train_dataloader=None,
        val_dataloader=None,
    )
    
    # Load checkpoint
    if not args.checkpoint:
        logger.error("Checkpoint must be provided for text generation")
        raise ValueError("Checkpoint must be provided for text generation")
    
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    trainer.load_checkpoint(args.checkpoint)
    
    # Get prompt text
    if args.prompt:
        prompt = args.prompt
    else:
        prompt = input("Enter a prompt for text generation: ")
    
    # Tokenize prompt
    logger.info(f"Generating text from prompt: {prompt}")
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=args.seq_length  # use train_seq_length if not in args
    )
    input_ids = encoded["input_ids"]
    
    # Generate text
    logger.debug(f"Generation parameters: temperature={args.temperature}, "
                f"top_k={args.top_k}, top_p={args.top_p}, max_length={args.max_length}")
    
    output_ids = trainer.generate(
        input_ids=input_ids,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )
    
    # Decode generated text
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Print generated text
    logger.info("Generated text:")
    logger.info(generated_text)
    
    # Save to file
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "generated_text.txt"
    logger.info(f"Saving generated text to {output_file}")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(generated_text)
    
    return generated_text


def main():
    """Main entry point."""
    args = parse_args()
    
    # Set up logging
    global logger
    logger = setup_logging(args)
    
    # Create config
    if args.config:
        # TODO: Load config from file
        config = DefaultConfig()
    else:
        config = DefaultConfig()
        
        # Override config with command line arguments
        config.model.dim = args.model_dim
        config.model.depth = args.depth
        config.model.heads = args.heads
        config.model.dim_head = args.dim_head
        config.model.segment_len = args.segment_len
        
        config.training.batch_size = args.batch_size
        config.training.learning_rate = args.learning_rate
        config.training.warmup_steps = args.warmup_steps
        config.training.max_steps = args.max_steps
        config.training.eval_every = args.eval_every
        config.training.save_every = args.save_every
        config.training.fp16 = args.fp16
        
        config.output_dir = args.output_dir
        config.checkpoint_dir = args.checkpoint_dir or os.path.join(args.output_dir, "checkpoints")
        config.log_dir = args.log_dir or os.path.join(args.output_dir, "logs")
        config.seed = args.seed
    
    try:
        # Run selected mode
        if args.mode == "train":
            train(args, config)
        elif args.mode == "eval":
            evaluate(args, config)
        elif args.mode == "generate":
            generate(args, config)
        else:
            logger.error(f"Unknown mode: {args.mode}")
            raise ValueError(f"Unknown mode: {args.mode}")
    except Exception as e:
        logger.exception(f"Error during execution: {e}")
        raise


if __name__ == "__main__":
    main()