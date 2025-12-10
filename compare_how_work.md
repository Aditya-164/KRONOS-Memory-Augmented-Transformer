The `compare_models.py` script performs a comparative analysis between a **Titans transformer** (a specialized variant) and a **vanilla transformer**. Here's a breakdown of its key components and workflow:

---

### **1. Core Purpose**
- Compare performance (training/inference), memory efficiency, and generation quality between Titans and vanilla transformers.
- Track metrics like loss, perplexity, accuracy, memory usage, and inference speed.
- Visualize results for analysis.

---

### **2. Key Components**
#### **a. Model Setup**
- **`create_model_pair()`**: Instantiates both models with shared hyperparameters (`dim`, `depth`, `heads`, etc.).
- **Titans-specific features**: Enabled via flags like `use_memory` (memory mechanisms) and `use_flex_attn` (flexible attention).

#### **b. Data Handling**
- **Datasets**: Supports `wikitext` (via `WikiTextDataset`) or custom datasets.
- **Tokenization**: Uses GPT-2's tokenizer (handles padding via EOS token).
- **Dataloaders**: Batches data with `CollatorForLanguageModeling` (masking for language modeling).

#### **c. Training Loop**
- **ComparativeTrainer**: Manages parallel training of both models:
  - **Mixed Precision**: Uses `GradScaler` for FP16 training.
  - **Gradient Handling**: Clipping (`max_grad_norm`) and accumulation.
  - **LR Scheduling**: Cosine annealing with warmup.
  - **Metrics Tracking**: Logs loss, perplexity, accuracy, and inference metrics.

#### **d. Evaluation & Metrics**
- **ComparativeMetrics**: Tracks per-model metrics (loss, perplexity, accuracy) and inference stats.
- **MemoryEfficiencyMetrics**: Monitors GPU memory usage, memory utilization, and retrieval quality (synthetic or Titans-specific).
- **Inference Comparison**: Measures time-per-token and memory during text generation.

#### **e. Visualization**
- Generates plots for:
  - Training curves (`plot_training_comparison`).
  - Inference performance (`plot_inference_metrics`).
  - Memory efficiency (`plot_memory_efficiency`).
  - Model state visualization (`visualize_model_state`).

#### **f. Text Generation**
- Generates sample text from both models to qualitatively compare outputs.

---

### **3. Workflow**
1. **Initialization**: Parse args, set up tokenizer/dataset, create models and optimizers.
2. **Training**:
   - Train both models in parallel with shared data batches.
   - Log metrics at intervals (`--log-interval`).
   - Evaluate periodically (`--eval-interval`).
3. **Post-Training**:
   - Collect memory metrics (real or synthetic).
   - Run inference benchmarks.
   - Generate and save sample text.
4. **Visualization**: Save plots to `--output-dir`.

---

### **4. Key Features**
- **Parallel Training**: Both models share the same data pipeline but train independently.
- **Memory Analysis**: Explicit tracking of Titans' memory mechanisms vs. vanilla transformer.
- **Synthetic Metrics**: Fallback metrics if Titans lacks memory instrumentation.
- **Reproducibility**: Configurable via CLI arguments (batch size, LR, steps, etc.).

---

### **5. Usage Example**
```bash
python compare_models.py \
  --dim 512 --depth 6 --heads 8 \
  --batch-size 32 --steps 10000 \
  --dataset wikitext --output-dir ./results
```

This would train both models on WikiText-103, save results to `./results`, and generate comparison plots.

