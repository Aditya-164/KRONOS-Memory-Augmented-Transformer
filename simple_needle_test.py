import os
import random 
import numpy as np
import platform

from transformers import AutoTokenizer

import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from src.evaluation.local_evaluator import LocalNeedleEvaluator

# Reuse loader and tokenizer from needle evaluation script
from src.training.trainer import Trainer as GenerationTrainer
from src.evaluation.create_test_haystack  import create_test_haystack
from src import (
    TitansTransformer,
    TitansLM,
    DefaultConfig,
    get_logger,
    init_weights
)

def load_model_from_checkpoint(args, vocab_size):
    """Load a Titans model from checkpoint."""
    logger.info(f"Creating model with {args.depth} layers, {args.model_dim} dimensions")
    
    # Create a default config
    config = DefaultConfig()
    config.model.dim = args.model_dim
    config.model.depth = args.depth
    config.model.heads = args.heads
    config.model.dim_head = args.dim_head
    config.model.segment_len = args.segment_len
    
    # Create base transformer model
    model = TitansTransformer(
        dim=config.model.dim,
        depth=config.model.depth,
        vocab_size=vocab_size,
        seq_len=args.context_length_max,  # Use the max context length
        dim_head=config.model.dim_head,
        heads=config.model.heads,
        segment_len=config.model.segment_len,
        use_memory=args.use_memory,
        sliding=args.sliding_window,
        num_longterm_mem_tokens=config.model.num_longterm_mem_tokens,
        num_persist_mem_tokens=config.model.num_persist_mem_tokens,
        use_flex_attn=config.model.use_flex_attn,
        accept_value_residual=config.model.neural_memory_add_value_residual
    )
    
    # Initialize weights
    model.apply(lambda m: init_weights(m))
    
    # Wrap in language model for generation capabilities
    language_model = TitansLM(
        transformer=model,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )
    
    # Load the checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Check if the checkpoint contains model state_dict directly or has it under 'model_state_dict' key
    if 'model_state_dict' in checkpoint:
        missing, unexpected = language_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        missing, unexpected = language_model.load_state_dict(checkpoint, strict=False)
    
    if missing:
        logger.warning(f"Missing keys when loading model: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys when loading model: {unexpected}")
    
    return language_model


def load_tokenizer(tokenizer_name):
    """Load tokenizer."""
    logger.info(f"Loading tokenizer: {tokenizer_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        # Ensure the tokenizer has the necessary tokens
        special_tokens = {}
        if tokenizer.pad_token is None:
            special_tokens["pad_token"] = "<pad>"
        if tokenizer.bos_token is None:
            special_tokens["bos_token"] = "<bos>"
        if tokenizer.eos_token is None:
            special_tokens["eos_token"] = "<eos>"
            
        if special_tokens:
            logger.debug(f"Adding special tokens: {special_tokens}")
            tokenizer.add_special_tokens(special_tokens)
            
        logger.debug(f"Vocabulary size: {len(tokenizer)}")
        return tokenizer
    except Exception as e:
        logger.error(f"Error loading tokenizer {tokenizer_name}: {e}")
        logger.info("Using default GPT-2 tokenizer")
        return AutoTokenizer.from_pretrained("gpt2")


def setup_logging(args):
    """Set up logging based on command-line arguments."""
    global logger
    
    # Set environment variables for global logging configuration
    os.environ["TITANS_LOG_LEVEL"] = "DEBUG" if args.debug else "INFO"
    os.environ["TITANS_LOG_FORMAT"] = "detailed"
    
    if args.log_file:
        log_file = args.log_file
    else:
        log_file = None
        
    if log_file:
        os.environ["TITANS_LOG_FILE"] = str(log_file)
    
    # Create main logger
    level = "DEBUG" if args.debug else "INFO"
    logger = get_logger("titans.needle", level=level, log_file=log_file, format_type="detailed")
    
    # Log startup information
    logger.info(f"Starting Titans Needle in a Haystack test")
    logger.debug(f"Command-line arguments: {args}")
    
    if args.debug:
        logger.debug("Debug mode enabled")
    
    # Log system information
    system_info = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda if torch.cuda.is_available() else "Not available",
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "platform": platform.platform(),
    }
    
    logger.info(f"System: Python {system_info['python']}, PyTorch {system_info['pytorch']}, "
                f"Device: {system_info['device']}")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            logger.debug(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    return logger

def parse_args():
    parser = argparse.ArgumentParser(description="Simple Needle in Haystack Test")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, default="bert-base-uncased", help="Tokenizer to use")
    parser.add_argument("--model_dim", type=int, default=256, help="Model dimension")
    parser.add_argument("--depth", type=int, default=4, help="Number of layers")
    parser.add_argument("--heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--dim_head", type=int, default=64, help="Dimension per attention head")
    parser.add_argument("--segment_len", type=int, default=512, help="Length of attention segments")
    parser.add_argument("--sliding_window", action="store_true", help="Use sliding window attention")
    parser.add_argument("--use_memory", action="store_true", help="Use neural memory in model")
    parser.add_argument("--needle", type=str, required=True, help="Needle text to find")
    parser.add_argument("--haystack_dir", type=str, required=True, help="Directory with documents")
    parser.add_argument("--max_context_len", type=int, default=1000, help="Max context length")
    # Sampling parameters for generation
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--output_dir", type=str, default="./simple_test_results", help="Output directory")
    parser.add_argument("--retrieval_question", type=str, default=None, help="Question to ask about the needle")
    # Logging options
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for detailed logs")
    parser.add_argument("--log_file", type=str, default=None, help="Path to log file")
    return parser.parse_args()

def load_model(args):
    print(f"Loading model from {args.checkpoint}...")
    # Load tokenizer to get vocab size
    tokenizer = load_tokenizer(args.tokenizer)
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the model using the reused function with args and vocab size
    model = load_model_from_checkpoint(args, tokenizer.vocab_size)
    model = model.to(device)

    model.eval()
    print(f"Model loaded and moved to {device}")
    return model, device

def load_documents(haystack_dir):
    doc_paths = list(Path(haystack_dir).rglob("*.txt"))  # Assuming documents are in .txt format
    documents = []
    
    print(f"Loading {len(doc_paths)} documents from {haystack_dir}...")
    
    for path in tqdm(doc_paths):
        with open(path, "r") as f:
            documents.append(f.read())
    
    return documents

def run_simple_test(args):
    # align argument names for downstream evaluator
    args.context_length_max = args.max_context_len
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer using the reused function
    tokenizer = load_tokenizer(args.tokenizer)
    
    # Load model
    model, device = load_model(args)
    # Wrap in Trainer for generation 
    generation_trainer = GenerationTrainer(model=model, train_dataloader=None, val_dataloader=None)
    
    # Load documents
    documents = load_documents(args.haystack_dir)
    
    # Initialize evaluator
    evaluator = LocalNeedleEvaluator()
    
    # Build question and input prompt
    # Use user-provided retrieval_question if available
    if args.retrieval_question:
        question = args.retrieval_question
    else:
        question = f"Find this information: {args.needle}"
    # Full prompt includes question and needle reference
    prompt_template = f"{question}: {args.needle}\n\n"
    
    results = []
    print(f"Running test with {len(documents)} documents...")
    
    # Process each document with timeout protection
    for i, document in enumerate(tqdm(documents)):
        result = {
            "document_id": i,
            "document_length": len(document),
            "contains_needle": args.needle in document,
            "retrieval_results": {}
        }
        
        # Truncate to max context length
        if len(document) > args.max_context_len:
            context = document[:args.max_context_len]
        else:
            context = document
        
        # Encode the question & needle plus context
        inputs = tokenizer(prompt_template + context, return_tensors="pt").to(device)
        
        try:
            # Generate response with a timeout
            with torch.no_grad():
                # Generate using Trainer.generate (consistent with main.py)
                generated_ids, _ = generation_trainer.generate(
                    input_ids=inputs.input_ids,
                    max_length=200,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p
                )
            # Ensure batch dimension (TitansLM.generate squeezes batch dim)
            if generated_ids.dim() == 1:
                generated_ids = generated_ids.unsqueeze(0)
            # Only decode newly generated tokens (exclude prompt)
            prompt_len = inputs.input_ids.size(1)
            gen_ids = generated_ids[:, prompt_len:]
            response = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
            
            # Evaluate the response
            evaluation = evaluator.evaluate_retrieval(
                needle=args.needle,
                retrieved_text=context,
                question=question,
                response=response
            )
            
            result["response"] = response
            result["evaluation"] = evaluation
            
        except Exception as e:
            result["error"] = str(e)
        
        results.append(result)
        
        # Save progress after each document
        with open(f"{args.output_dir}/results.txt", "a") as f:
            f.write(f"Document {i+1}/{len(documents)}\n")
            f.write(f"Contains needle: {result.get('contains_needle')}\n")
            f.write(f"Response: {result.get('response', 'ERROR')}\n")
            f.write(f"Evaluation: {result.get('evaluation', {})}\n")
            f.write("-" * 50 + "\n")
    
    # Save final results
    import json
    with open(f"{args.output_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Test completed. Results saved to {args.output_dir}")
    return results

def main():
    args = parse_args()
    # Initialize shared logger
    setup_logging(args)

    # check if test haystack dierctory exists else create 
    if not os.path.exists(args.haystack_dir):
        create_test_haystack()
        logger.info(f"Created haystack directory at {args.haystack_dir}")
    else:
        logger.info(f"Using existing haystack directory at {args.haystack_dir}")

    run_simple_test(args)

if __name__ == "__main__":
    main()