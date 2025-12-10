import re
import os
import sys
from pathlib import Path
import json
import torch
from transformers import AutoTokenizer
import datasets
import pandas as pd
from tqdm.auto import tqdm

from src.models import TitansTransformer, TitansLM
from src.evaluation.prompts import DEFAULT_PROMPTS, get_formatted_input
from src.config import ModelConfig

TEMPLATE = '{instruction}\n{examples}\n{post_prompt}\nContext: {context}\n\nQuestion: {question}'

# Evaluation loop
def evaluate(prompt_cfg, prompt_name, model, tokenizer, device, benchmark_name, tasks, splits, gen_kwargs):
    for split in splits:
        print(f"\n--- Evaluating split: {split} ---")
        data = datasets.load_dataset(benchmark_name, name=split)

        for task in tasks:
            task_data = data[task]
            out_dir = Path('evals') / "TitansLM" / f"{task}_{split}_{prompt_name}"
            out_dir.mkdir(parents=True, exist_ok=True)
            csv_path = out_dir / 'results.csv'
            df = pd.DataFrame(columns=['target', 'output', 'question'])

            for sample in tqdm(task_data, desc=f"{task}-{split}"):
                target = sample['target']
                context = sample['input']
                question = sample['question']
                input_text = get_formatted_input(
                    context=context,
                    question=question,
                    examples=prompt_cfg['examples'],
                    instruction=prompt_cfg['instruction'],
                    post_prompt=prompt_cfg['post_prompt'],
                    template=TEMPLATE
                )
                inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=1000).to(device)
                with torch.no_grad():
                    gen_ids, _ = model.generate(inputs['input_ids'], **gen_kwargs)
                output = tokenizer.decode(gen_ids[inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

                df.loc[len(df)] = {'target': target, 'output': output, 'question': question}

            df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    config = ModelConfig() # use deafult modify in place to laod differing models

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    checkpoint = torch.load(r"..\checkpoints\final_562.pt", map_location='cpu')

    transformer = TitansTransformer(
        dim=config.model.dim,
        depth=config.model.depth,
        vocab_size=tokenizer.vocab_size,
        seq_len=512,
        dim_head=config.model.dim_head,
        heads=config.model.heads,
        segment_len=config.model.segment_len
    )
    model = TitansLM(transformer)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != 'cuda':
        raise RuntimeError("CUDA is required for this model.")
    else:
        print(f"Using device: {device}")

    model.eval()
    model.to(device)

    gen_kwargs = {
        'max_length': 1024,
        'temperature': 1.0,
        'top_k': None,
        'top_p': None,
        'return_logits': False
    }

    benchmark_name = 'booydar/babilong-1k-samples'
    tasks = ['qa1', 'qa2', 'qa3', 'qa4', 'qa5']
    splits = ['0k', '1k']

    # ---- Zero-shot ----
    i_cfg = {'instruction': '', 'examples': '', 'post_prompt': ''}
    prompt_name = '_'.join([f"{k}_{'no' if not i_cfg[k] else 'yes'}" for k in i_cfg])
    evaluate(i_cfg, prompt_name, model, tokenizer, device, benchmark_name, tasks, splits, gen_kwargs)

    # ---- Few-shot ----
    for task in tasks:
        prompt_cfg = DEFAULT_PROMPTS[task]
        prompt_name = f"fewshot_{task}"
        evaluate(prompt_cfg, prompt_name, model, tokenizer, device, benchmark_name, tasks, splits, gen_kwargs)
