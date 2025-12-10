"""Configuration parameters for the Titans model and training."""

class ModelConfig:
    """Model configuration parameters."""
    
    def __init__(
        self,
        num_tokens=50257,          # Vocabulary size
        dim=256,                   # Model dimension
        depth=12,                  # Number of layers
        segment_len=512,           # Length of each segment for attention
        neural_memory_segment_len=None,  # Length of segments for neural memory
        neural_mem_gate_attn_output=False,  # Whether to gate attention with memory
        neural_memory_add_value_residual=False,  # Add value residual to memory
        num_longterm_mem_tokens=32,  # Number of long-term memory tokens
        num_persist_mem_tokens=0,    # Number of persistent memory tokens
        dim_head=64,               # Dimension of each attention head
        heads=8,                   # Number of attention heads
        ff_mult=4,                 # Feedforward multiplier
        num_residual_streams=4,    # Number of residual streams in hyper-connections
        use_flex_attn=False,       # Whether to use flex attention
        sliding_window_attn=True,  # Whether to use sliding window attention
        neural_mem_weight_residual=False  # Whether to use weight residual in neural memory
    ):
        self.num_tokens = num_tokens
        self.dim = dim
        self.depth = depth
        self.segment_len = segment_len
        self.neural_memory_segment_len = neural_memory_segment_len
        self.neural_mem_gate_attn_output = neural_mem_gate_attn_output
        self.neural_memory_add_value_residual = neural_memory_add_value_residual
        self.num_longterm_mem_tokens = num_longterm_mem_tokens
        self.num_persist_mem_tokens = num_persist_mem_tokens
        self.dim_head = dim_head
        self.heads = heads
        self.ff_mult = ff_mult
        self.num_residual_streams = num_residual_streams
        self.use_flex_attn = use_flex_attn
        self.sliding_window_attn = sliding_window_attn
        self.neural_mem_weight_residual = neural_mem_weight_residual

    def __repr__(self):
        attrs = ', '.join(f'{key}={value!r}' for key, value in self.__dict__.items())
        return f"{self.__class__.__name__}({attrs})"
    def to_dict(self):
        return self.__dict__

class TrainingConfig:
    """Training configuration parameters."""
    
    def __init__(
        self,
        batch_size=32,               # Batch size for training
        learning_rate=3e-4,          # Learning rate
        weight_decay=0.01,           # Weight decay for AdamW
        warmup_steps=10000,          # Linear warmup steps
        max_steps=100000,            # Maximum training steps
        eval_every=1000,             # Evaluate every N steps
        save_every=5000,             # Save checkpoint every N steps
        gradient_clip_val=1.0,       # Gradient clipping value
        fp16=True,                   # Whether to use mixed precision training
        num_workers=4,               # Number of workers for data loading
        train_sequence_length=1024,  # Training sequence length
        val_sequence_length=2048,    # Validation sequence length

        # Reasoning Trainer param (Optional)
        ## reasoning annotated data config
        coconut_reasoning_start = None,  # Starting index for coconut reasoning  
        coconut_reasoning_end = None,    # Ending index for coconut reasoning
        bot_token_id = 0,              # Begining of Thought Token ID
        eot_token_id = 1,              # End of Thought Token ID
        continuous_steps = 3,          # Number of continious thought steps 
        ## non annotated data config
        use_annotated_spans = False,            # Train with provided annotations if True
        hybrid_random_ratio = 0.2,     # 20% random spans, 80% tail heuristic
        random_span_length = None,    # If set, overrides continuous_steps for random spans

    ):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.eval_every = eval_every
        self.save_every = save_every
        self.gradient_clip_val = gradient_clip_val
        self.fp16 = fp16
        self.num_workers = num_workers
        self.train_sequence_length = train_sequence_length
        self.val_sequence_length = val_sequence_length

        self.coconut_reasoning_start = coconut_reasoning_start
        self.coconut_reasoning_end = coconut_reasoning_end
        self.bot_token_id = bot_token_id
        self.eot_token_id = eot_token_id
        self.continuous_steps = continuous_steps

        self.use_annotated_spans = use_annotated_spans
        self.hybrid_random_ratio = hybrid_random_ratio
        self.random_span_length = random_span_length
    def to_dict(self):
        return self.__dict__

class DefaultConfig:
    """Default configuration combining model and training parameters."""
    
    def __init__(self):
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.seed = 42 # Cause 42 is the answer to everything
        self.output_dir = "./outputs"
        self.data_dir = "./data"
        self.checkpoint_dir = "./checkpoints"
        self.log_dir = "./logs"