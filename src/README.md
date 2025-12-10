# Titans

A neural network implementation focusing on memory-augmented transformers with segmented attention mechanisms.

## Project Structure

```
titans/
├── __init__.py
├── config.py              # Configuration parameters
├── README.md
├── requirements.txt
├── models/
│   ├── __init__.py
│   ├── attention.py       # Segmented attention and related components
│   ├── factory.py         # Factory functions for model creation
│   ├── memory.py          # Neural memory module implementation
│   ├── transformer.py     # MAC transformer implementation
│   ├── utils.py           # Helper functions for the models
│   └── vanilla_transformer.py  # Standard transformer implementation
├── data/
│   ├── __init__.py
│   └── dataloader.py      # Dataset and dataloader utilities
├── training/
│   ├── __init__.py
│   ├── comparative_trainer.py  # Training module for comparing models
│   ├── comparison_metrics.py   # Metrics for model comparison 
│   ├── memory_metrics.py       # Memory-specific evaluation metrics
│   ├── metrics.py              # General evaluation metrics
│   └── trainer.py              # Training loop and utilities
├── evaluation/
│   ├── comparison.py           # Model comparison tools
│   └── needle_in_a_haystack_benchmark.py  # Specific evaluation benchmark
├── utils/                      # Utility functions
├── visualizations/             # Visualization utilities
│   └── plots.py                # Visualization tools including model state
```

## Current Progress

- Implemented core memory-augmented transformer architecture
- Created segmented attention mechanisms with axial positional embeddings
- Developed neural memory module with momentum gates and soft clamping
- Added comparative analysis against vanilla transformer baseline
- Implemented state tracking for stateful execution

## Architecture Overview

### Titans Transformer

The Titans model is a memory-augmented transformer architecture that combines:

1. **Segmented Attention** - Processes input in segments for efficient computation with sliding window attention mechanisms
2. **Neural Memory** - External memory module that stores and retrieves information across long contexts
3. **Continuous Axial Positional Embeddings** - Advanced positional encoding for improved sequence representation

The architecture is defined by:

```
TitansTransformer
├── Token Embedding
├── Positional Embedding (Axial or Standard)
├── Transformer Blocks [1..depth]
│   ├── Segmented Attention
│   │   ├── Sliding Window Attention
│   │   └── Block Diagonal Masking
│   ├── Feed Forward Network
│   └── Neural Memory (optional)
│       ├── Query-Key-Value Memory Interface
│       ├── Momentum Gates
│       └── Soft Clamping Mechanisms
└── Output Projection
```

The implementation uses a `TitansTransformer` class that inherits core transformer functionality while adding memory augmentation capabilities. The model supports both stateless and stateful execution modes, making it suitable for both training and efficient inference scenarios.

### Neural Memory Module

The memory module features:

- Weight decay factors for stability during long training runs
- Momentum gate transitions for controlled updates to prevent catastrophic forgetting
- Gradient norm clamping to prevent exploding gradients during backpropagation
- Soft clamping mechanisms for smooth operation and stable learning dynamics
- Query-key-value architecture for memory interaction with residual connections

The memory layer implementation in `memory.py` maintains state through the `NeuralMemoryState` class, which tracks:
- Current memory state tensors
- Memory updates for visualization and analysis
- Memory efficiency metrics to evaluate usage patterns

Memory updates are controlled through configurable parameters including weight decay factors and neural_mem_weight_residual settings, allowing fine-grained control over how information is stored and retrieved.

### State Management

The model maintains state through the `TransformerState` dataclass:
- Memory states for each layer, tracking information flow through the network
- KV caches for efficient inference by reusing previously computed key-value pairs
- Value residuals for enhanced memory operation and gradient flow
- Memory efficiency metrics including relevance scores for analytical purposes

State visualization is supported through the `visualize_model_state` function in `visualizations/plots.py`, which generates comprehensive visualizations including:
- Memory state heatmaps across all layers
- Key norm plots for analyzing KV cache patterns
- Update magnitude analysis for memory operations
- Memory relevance scores to measure information usage efficiency

This state management approach enables both improved training dynamics and efficient inference, with the model capable of maintaining contextual information across longer sequences than traditional transformers.

## Code Explanation

### Models
- **attention.py**: Implements segmented attention mechanisms that divide input into segments for efficient processing
- **memory.py**: Neural memory module that augments the transformer with external memory capabilities
- **transformer.py**: The Memory-Augmented Computing (MAC) transformer implementation
- **vanilla_transformer.py**: Standard transformer implementation for comparison
- **factory.py**: Factory functions for creating consistent model configurations
- **utils.py**: Helper functions for model initialization and tensor operations

### Data
- **dataloader.py**: Handles dataset loading, preprocessing, and batch creation

### Training
- **trainer.py**: Manages training loops, checkpoint saving/loading, and optimization
- **metrics.py**: Implements evaluation metrics for model performance assessment
- **comparative_trainer.py**: Training module for multiple models
- **comparison_metrics.py**: Metrics for comparing multiple models
- **memory_metrics.py**: Specialized metrics for memory evaluation

### Evaluation
- **comparison.py**: Evaluation utilities for model comparison
- **needle_in_a_haystack_benchmark.py**: Specialized benchmark for memory retention

### Visualization
- **plots.py**: Tools for visualizing attention weights, memory usage, training progress, and model state

## Usage Examples

### Training and Comparing Models

```bash
python -m titans.examples.compare_models \
  --dim 512 \
  --depth 6 \
  --heads 8 \
  --dim-head 64 \
  --segment-len 512 \
  --batch-size 16 \
  --lr 5e-5 \
  --steps 10000 \
  --warmup-steps 500 \
  --eval-steps 50 \
  --eval-interval 1000 \
  --log-interval 100 \
  --dataset wikitext \
  --seq-len 1024 \
  --data-dir ./data \
  --output-dir ./outputs/comparison \
  --save-interval 1000
```

### Model Configuration Options

- **Core architecture**
  - `--dim`: Model dimension (default: 512)
  - `--depth`: Number of transformer layers (default: 6)
  - `--heads`: Number of attention heads (default: 8)
  - `--dim-head`: Dimension of each attention head (default: 64)
  - `--segment-len`: Length of segments for Titans (default: 512)
  - `--ff-mult`: Feed-forward multiplier (default: 4)
  - `--use-memory`: Enable neural memory module
  - `--use-flex-attn`: Enable flexible attention
  - `--memory-dim`: Dimension of memory (default: same as model dim)

- **Training configuration**
  - `--batch-size`: Batch size for training (default: 16)
  - `--eval-batch-size`: Batch size for evaluation (default: 16)
  - `--lr`: Learning rate (default: 5e-5)
  - `--steps`: Number of training steps (default: 10000)
  - `--warmup-steps`: Learning rate warmup steps (default: 1000)
  - `--max-grad-norm`: Maximum gradient norm for clipping (default: 0.5)
  - `--label-smoothing`: Label smoothing factor (default: 0.1)

- **Output configuration**
  - `--output-dir`: Output directory for results and visualizations (default: ./outputs)
  - `--save-interval`: Steps between saving checkpoints (default: 1000)
  - `--log-interval`: Steps between logging (default: 100)
  - `--eval-interval`: Steps between evaluations (default: 500)

## TODO

- [x] Built data loading utilities
- [x] Write training and evaluation scripts
- [x] Add visualization capabilities for model analysis
- [x] Compare the model with standard transformer models
- [ ] Implement memory-augmented transformer on a standard benchmark dataset
- [ ] Compare the model with other state-of-the-art models

## Example Run

```bash
python3 main.py --mode train --dataset wikitext --wikitext_version 103 \
  --data_path ./titans/data/dataloader --model_dim 128 --depth 6 --heads 3 \
  --seq_length 1024 --batch_size 1 --val_batch_size 1 \
  --output_dir ./outputs/wikitext_model --learning_rate 0.03 --use_memory \
  --fp16 --max_epochs 50
```