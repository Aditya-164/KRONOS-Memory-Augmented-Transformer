# Compare Models
python3 compare_models.py  --dim 256   --depth 4   --heads 4   --dim-head 64 --segment-len 256   --batch-size 2   --eval-batch-size 2   --lr 1e-4   --steps 3000   --warmup-steps 100   --eval-steps 50   --eval-interval 500   --log-interval 50   --dataset paul_gram.txt  --seq-len 512  --data-dir ./data --output-dir ./outputs --save-interval 500

# Train Titans Transformer
```bash
python main.py --mode train \
  --model_type titans \
  --data_path ./data/brown_corpus.txt \
  --tokenizer bert-base-uncased \
  --model_dim 128 --depth 3 --heads 3 \
  --seq_length 512 --batch_size 2 --val_batch_size 2 \
  --output_dir ./outputs/brown_model_titans \
  --log_file ./outputs/brown_model_titans/training_log.txt \
  --learning_rate 1e-5 \
  --use_memory \
  --max_epochs 5 \
  --warmup_steps 500 \
  --save_every 5000 \
  --eval_every 500 \
  --debug
```

# Train Kronos Transformer
```bash
python main.py --mode train \
  --model_type kronos \
  --data_path ./data/brown_corpus.txt \
  --tokenizer bert-base-uncased \
  --model_dim 128 --depth 3 --heads 3 \
  --seq_length 512 --batch_size 2 --val_batch_size 2 \
  --output_dir ./outputs/brown_model_kronos \
  --log_file ./outputs/brown_model_kronos/training_log.txt \
  --learning_rate 1e-5 \
  --use_memory \
  --max_epochs 5 \
  --warmup_steps 500 \
  --save_every 5000 \
  --eval_every 500 \
  --debug
```

# Train Vanilla Transformer
```bash
python main.py --mode train \
  --model_type vanilla \
  --data_path ./data/brown_corpus.txt \
  --tokenizer bert-base-uncased \
  --model_dim 128 --depth 3 --heads 3 \
  --seq_length 512 --batch_size 2 --val_batch_size 2 \
  --output_dir ./outputs/brown_model_vanilla \
  --log_file ./outputs/brown_model_vanilla/training_log.txt \
  --learning_rate 1e-5 \
  --max_epochs 5 \
  --warmup_steps 500 \
  --save_every 5000 \
  --eval_every 500 \
  --debug
```

# Needle in a HayStack
python needle_in_a_haystack_evaluate.py \
  --checkpoint ./outputs/brown_model_final/checkpoints/final_576.pt \
  --tokenizer bert-base-uncased \
  --model_dim 256 --depth 4 --heads 4 \
  --use_memory \
  --needle "The brown fox jumped over the lazy dog." \
  --haystack_dir ./test_haystack \
  --retrieval_question "What did the fox do?" \
  --evaluator_type local \
  --context_length_min 200 \
  --context_length_max 1000 \
  --num_intervals 3 \
  --output_dir ./needle_results \
  --save_results \
  --save_contexts \
  --log_file ./needle_results/needle_eval_local.log