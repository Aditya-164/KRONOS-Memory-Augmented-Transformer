# NaturalStupidity: Memory-Augmented Transformers

Implementation of "Titans" memory-augmented transformer architecture with segmented attention mechanisms for efficient long-sequence processing.

## Project Overview

NaturalStupidity implements a transformer architecture with neural memory components designed to efficiently process long sequences through segmented attention and memory mechanisms. It's based on the Titans architecture with extensions for improved memory handling and attention.

### Key Features

- **Segmented Attention**: Processes long sequences by dividing them into manageable segments
- **Neural Memory**: Long-term memory mechanism for information retention across sequences
- **Sliding Window Attention**: Efficient attention that focuses on local context
- **Flexible Architecture**: Configurable model dimensions, memory size, and attention mechanisms
- **Attention Visualization**: Rich visualization tools for understanding model internals
- **Comprehensive Evaluation**: Dedicated tools for model evaluation and analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/NaturalStupidity.git
cd NaturalStupidity

# Install requirements
pip install -r titans/requirements.txt
```

## Quick Start

### Training a Model

```bash
python main.py --mode train \
  --dataset wikitext \
  --wikitext_version 103 \
  --data_path ./data \
  --model_dim 512 \
  --depth 12 \
  --heads 8 \
  --seq_length 1024 \
  --batch_size 4 \
  --val_batch_size 2 \
  --gradient_accumulation 4 \
  --fp16 \
  --output_dir ./outputs/wikitext_model \
  --learning_rate 3e-4 \
  --use_memory \
  --sliding_window
```
step
+ **OOM mitigation**: If you encounter CUDA out-of-memory errors, try:
  - Reducing `--batch_size`
  - Enabling mixed precision (`--fp16`)
  - Increasing `--gradient_accumulation`
  - Splitting sequences or using a smaller `--seq_length`

### Evaluating a Model

#### Basic Evaluation

```bash
python main.py --mode eval \
  --dataset wikitext \
  --wikitext_version 103 \
  --data_path ./data \
  --checkpoint ./outputs/wikitext_model/checkpoints/best_model_10000.pt \
  --val_batch_size 2
```

#### Comprehensive Evaluation with Visualizations

```bash
python3 evaluate_model_with_visualizations.py \
  --checkpoint ./outputs/wikitext_model/checkpoints/best_model_10000.pt \
  --model_dim 512 \
  --depth 12 \
  --heads 8 \
  --use_memory \
  --vis_all \
  --dataset wikitext \
  --data_path ./data \
  --wikitext_version 103 \
  --num_gen_samples 5 \
  --temperature 0.7
```

This will generate:
- Complete evaluation metrics (perplexity, loss)
- Text generation samples
- Attention pattern visualizations
- Memory usage visualizations (if using memory)
- Generation animations showing how attention flows during text generation
- A comprehensive evaluation report in both JSON and human-readable formats

### Generating Text

```bash
python main.py --mode generate \
  --checkpoint ./outputs/wikitext_model/checkpoints/best_model_10000.pt \
  --prompt "Once upon a time" \
  --max_length 100 \
  --temperature 0.8 \
  --top_k 50 \
  --top_p 0.9
```

### Comparing Models

Compare Titans with vanilla transformer models to analyze performance differences:

```bash
python compare_models.py \
  --dim 512 \
  --depth 6 \
  --heads 8 \
  --segment-len 512 \
  --batch-size 16 \
  --steps 5000 \
  --eval-interval 500 \
  --dataset wikitext \
  --output-dir ./outputs/model_comparison
```

This will train both models in parallel and generate comprehensive comparative visualizations including:
- Training metrics comparison (loss, perplexity, accuracy)
- Inference performance (speed and memory usage)
- Memory efficiency metrics (for Titans model)
- Generation quality comparison

### Simple Needle in a Haystack

```bash
python3 simple_needle_test.py \
  --checkpoint ./outputs/checkpoints/final_562.pt \
  --tokenizer distilgpt2 \
  --model_dim 256 \
  --depth 6 \
  --heads 8 \
  --dim_head 64 \
  --segment_len 512 \
  --max_context_len 512 \
  --needle "Donald Trump is Blonde" \
  --haystack_dir ./haystack \
  --retrieval_question "How does Donald Trump look" \
  --output_dir ./simple_test_results \
  --temperature 0.7 \
  --top_p 0.9 \
  --top_k 50 \
  --debug
```

This evaluates the model on the "Needle in a Haystack" task, where it retrieves relevant information from a large corpus based on a given query.

## Model Architecture

The Titans architecture consists of:

1. **Transformer Base**: Standard transformer with multi-head attention and feedforward layers
2. **Segmented Attention**: Divides input sequences into segments for efficient processing
3. **Neural Memory**: Maintains and updates a memory state across sequence fragments
4. **Continuous Axial Positional Embeddings**: Position embeddings that handle long sequences

### Memory Mechanism

The neural memory component:
- Updates based on input sequences using a surprise signal
- Retrieves information relevant to the current context
- Provides persistent memory across processing steps
- Uses momentum and forgetting mechanisms for stable updates

### Attention Visualization

The project includes rich visualization capabilities:
- **Attention Patterns**: Visualize how attention flows between tokens
- **Head Comparison**: Compare behaviors of different attention heads
- **Layer Analysis**: Analyze how different layers attend to information
- **Memory Activation**: Visualize memory usage and update patterns
- **Generation Animation**: Create animations of attention during text generation

## Project Structure
```bash
titans/
├── __init__.py         # Package initialization
├── config.py           # Configuration parameters
├── models/             # Model implementations
│   ├── __init__.py
│   ├── attention.py    # Segmented attention mechanisms
│   ├── memory.py       # Neural memory module
│   ├── transformer.py  # Titans transformer implementation
│   └── utils.py        # Helper functions
├── data/               # Data handling utilities
│   ├── __init__.py
│   └── dataloader.py   # Dataset and dataloader classes
├── training/           # Training utilities
│   ├── __init__.py
│   ├── trainer.py      # Training loop implementation
│   ├── utils.py        # Training utilities including schedulers
│   └── metrics.py      # Evaluation metrics
├── visualizations/     # Visualization tools
│   ├── __init__.py
│   ├── plots.py        # Basic plotting functions
│   ├── attention_viz.py # Attention visualization module
│   └── comparison_plots.py # Plots for model comparison
├── evaluation/         # Evaluation tools
└── requirements.txt    # Project dependencies
```

## Scripts and Tools

The project includes several key scripts:

- `main.py` - Main training and basic evaluation script
- `evaluate_model_with_visualizations.py` - Comprehensive evaluation with visualizations
- `compare_models.py` - Comparative analysis between model architectures
- `needle_in_a_haystack_evaluate.py` - Specialized task for information retrieval evaluation

## Configuration Options
Models can be configured with various parameters:

- **Model Architecture**
  - `model_dim`: Dimensionality of the model
  - `depth`: Number of transformer layers
  - `heads`: Number of attention heads
  - `segment_len`: Length of attention segments
  - `use_memory`: Whether to use neural memory

- **Training Parameters**
  - `batch_size`: Batch size for training
  - `learning_rate`: Learning rate for optimization
  - `warmup_steps`: Steps for learning rate warmup
  - `fp16`: Use mixed-precision training
  - `label_smoothing`: Apply label smoothing to loss function

See `config.py` for all available configuration options.

## Dataset Support
- **WikiText**: Support for WikiText-2 and WikiText-103 datasets
- **Custom Text**: Support for custom text datasets

## Model Comparison

The project includes tools for comparative analysis between Titans and vanilla transformer models:

- **Performance Comparison**: Training speed, memory usage, and inference latency
- **Quality Metrics**: Perplexity, accuracy, and generation quality
- **Memory Analysis**: Visualization of memory update patterns and utilization
- **Attention Pattern Comparison**: Compare attention patterns between model architectures

Visualizations are automatically generated and saved to the specified output directory.

## TODO:
- Decide a good name for this project
- Make a basic implementation of the titans-paper
- Run the titans-paper on a simple dataset
- Implement the titans-paper on a more standard benchmark dataset
- Implement novel ideas to improve the titans-paper
>- Implement a conceptually buckketed memory
>- Implement a latent space reasoning module within transformer layers
>- Implement a masked adversial resistenat training module with random masking attention
- Run the improved model on a standard benchmark dataset
- Compare the results with the original titans-paper, and other state-of-the-art models
- Write a paper on the results
- Submit the paper to a conference
- Be happy
