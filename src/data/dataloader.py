import os
import json
import time
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Optional, Tuple, List, Dict, Any, Union, Callable
from pathlib import Path
import logging
import random
from transformers import PreTrainedTokenizer, AutoTokenizer
import requests
import tarfile
from tqdm import tqdm
import zipfile

"""Dataset and dataloader utilities for the Titans model."""

try:
    from transformers import PreTrainedTokenizer, AutoTokenizer
except ImportError:
    logging.warning("Transformers library not found. Some tokenization features will be unavailable.")

class TextDataset(Dataset):
    """Dataset for text data.
    
    This dataset loads text data from files or pre-tokenized data and
    provides sequences for training and evaluation.
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: Optional[Union[PreTrainedTokenizer, str]] = None,
        seq_length: int = 1024,
        stride: Optional[int] = None,
        return_tensors: bool = True,
        add_bos_token: bool = True,
        add_eos_token: bool = True,
        overwrite_cache: bool = False,
        cache_dir: Optional[str] = None,
        use_memory_map: bool = True,
        is_pre_tokenized: bool = False,
    ):
        """Initialize the text dataset.
        
        Args:
            data_path: Path to the data file or directory
            tokenizer: Tokenizer or name of pretrained tokenizer
            seq_length: Maximum sequence length
            stride: Stride for overlapping sequences (if None, stride = seq_length)
            return_tensors: Whether to return torch tensors (vs. lists)
            add_bos_token: Whether to add beginning-of-sequence token
            add_eos_token: Whether to add end-of-sequence token
            overwrite_cache: Whether to overwrite existing cache files
            cache_dir: Directory for cache files
            use_memory_map: Whether to use memory-mapped arrays for large datasets
            is_pre_tokenized: Whether the input data is already tokenized
        """
        super().__init__()
        self.data_path = Path(data_path)
        self.seq_length = seq_length
        self.stride = stride if stride is not None else seq_length
        self.return_tensors = return_tensors
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.overwrite_cache = overwrite_cache
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_memory_map = use_memory_map
        self.is_pre_tokenized = is_pre_tokenized
        
        # Initialize tokenizer
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer
        
        # Create cache directory if needed
        if self.cache_dir and not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True)
        
        # Load or create tokenized data
        self.token_ids = self.load_and_tokenize_data()
        
        # Create examples
        self.examples = self.create_examples()
        
    def load_and_tokenize_data(self) -> Union[np.ndarray, List[int]]:
        """Load data from file and tokenize it.
        
        Returns:
            List of token IDs or numpy array
        """
        # Check if cached file exists
        cache_path = None
        if self.cache_dir:
            cache_path = self.cache_dir / f"{self.data_path.stem}_tokenized.npy"
            if cache_path.exists() and not self.overwrite_cache:
                logging.info(f"Loading tokenized data from cache {cache_path}")
                if self.use_memory_map:
                    return np.memmap(cache_path, dtype=np.int32, mode='r')
                else:
                    return np.load(cache_path)
        
        token_ids = None

        if self.is_pre_tokenized:
            # Handle pre-tokenized data
            if self.data_path.suffix == '.npy':
                token_ids = np.load(self.data_path)
            elif self.data_path.suffix == '.pt':
                token_ids = torch.load(self.data_path).numpy()
            elif self.data_path.suffix == '.json':
                # Expect the JSON to contain a key "token_ids" with a list of ints.
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                token_ids = data.get("token_ids")
                if token_ids is None:
                    raise ValueError("Expected pre-tokenized JSON file with a 'token_ids' key.")
            else:
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    token_ids = [int(x) for x in text.split()]
        else:
            # For raw text data:
            if self.data_path.suffix == '.json':
                # Load the JSON file and extract text from each row
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                texts = []
                for row in data.get("rows", []):
                    # Extract the "text" field from each row (adjust keys if needed)
                    text = row.get("row", {}).get("text", "")
                    texts.append(text)
                full_text = "\n".join(texts)
            else:
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    full_text = f.read()
            
            if self.tokenizer is None:
                raise ValueError("Tokenizer is required for raw text data")
            
            # Tokenize the full text
            token_ids = self.tokenizer.encode(full_text)
        
        # Convert token_ids to a numpy array if they aren't already
        if not isinstance(token_ids, np.ndarray):
            token_ids = np.array(token_ids, dtype=np.int32)
        
        # Save to cache if specified
        if cache_path and self.overwrite_cache:
            logging.info(f"Saving tokenized data to cache {cache_path}")
            np.save(cache_path, token_ids)
        
        return token_ids
    
    def create_examples(self) -> List[Dict[str, Any]]:
        """Create examples from tokenized data.
        
        Returns:
            List of examples, where each example is a dict with 'input_ids' and 'labels'
        """
        examples = []
        # Handle small files: pad token_ids if shorter than required sequence length
        effective_seq_len = self.seq_length
        if self.add_bos_token:
            effective_seq_len -= 1
        if self.add_eos_token:
            effective_seq_len -= 1
        if len(self.token_ids) < effective_seq_len:
            pad_token_id = getattr(self.tokenizer, 'pad_token_id', 0)
            pad_length = effective_seq_len - len(self.token_ids)
            self.token_ids = list(self.token_ids) + [pad_token_id] * pad_length
        
        # Create examples with stride
        for i in range(0, len(self.token_ids) - effective_seq_len + 1, self.stride):
            input_ids = self.token_ids[i:i + effective_seq_len].tolist()
            
            # Add special tokens if needed
            if self.add_bos_token:
                if self.tokenizer:
                    bos_token_id = self.tokenizer.bos_token_id
                    if bos_token_id is None:
                        self.tokenizer.add_special_tokens({"bos_token":"<BOS>"})
                        bos_token_id = self.tokenizer.bos_token_id 
                    input_ids = [bos_token_id] + input_ids
                else:
                    input_ids = [0] + input_ids  # Using 0 as BOS if no tokenizer
                    
            if self.add_eos_token:
                if self.tokenizer:
                    eos_token_id = self.tokenizer.eos_token_id
                    if eos_token_id is None:
                        self.tokenizer.add_special_tokens({"eos_token":"<EOS>"})
                        eos_token_id = self.tokenizer.eos_token_id 
                    input_ids = [eos_token_id] + input_ids
                else:
                    input_ids = input_ids + [1]  # Using 1 as EOS if no tokenizer


            examples.append({
                "input_ids": torch.tensor(input_ids) if self.return_tensors else input_ids,
                "labels": torch.tensor(input_ids) if self.return_tensors else input_ids,
            })
        
        return examples
    
    def __len__(self) -> int:
        """Get the number of examples."""
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get an example by index."""
        return self.examples[idx]


class WikiTextDataset(TextDataset):
    """WikiText dataset for language modeling."""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        tokenizer: Optional[Union[PreTrainedTokenizer, str]] = None,
        seq_length: int = 1024,
        stride: Optional[int] = None,
        return_tensors: bool = True,
        version: str = "103",
        **kwargs
    ):
        """Initialize the WikiText dataset.
        
        Args:
            data_dir: Directory containing WikiText data
            split: Data split ('train', 'valid', or 'test')
            tokenizer: Tokenizer or name of pretrained tokenizer
            seq_length: Maximum sequence length
            stride: Stride for overlapping sequences (if None, stride = seq_length)
            return_tensors: Whether to return torch tensors (vs. lists)
            version: WikiText version ('2' or '103')
        """
        assert split in ["train", "valid", "test"], "Split must be 'train', 'valid', or 'test'"
        assert version in ["2", "103"], "Version must be '2' or '103'"
        
        data_dir = Path(data_dir)
        file_name = f"wiki.{split}.tokens" if version == "2" else f"wikitext-{version}-rows-0-100.json"
        data_path = data_dir / file_name
        
        # Download data if not available
        if not data_path.exists():
            self.download_wikitext(data_dir, version)
        
        super().__init__(
            data_path=data_path,
            tokenizer=tokenizer,
            seq_length=seq_length,
            stride=stride,
            return_tensors=return_tensors,
            **kwargs
        )
    
    @staticmethod
    def download_wikitext(data_dir: Path, version: str, offset: int = 0, length: int = 100, max_retries: int = 3):
        """
        Download WikiText rows using the Hugging Face API and save the JSON response.
        
        Args:
            data_dir: Directory to save the dataset.
            version: WikiText version ('2' or '103'). (Currently the API call is set for '103'.)
            offset: Starting row offset.
            length: Number of rows to fetch.
            max_retries: Maximum number of retries in case of a server error.
        """
        data_dir.mkdir(exist_ok=True, parents=True)
        
        # Construct the API URL
        url = (
            "https://datasets-server.huggingface.co/rows?"
            "dataset=Salesforce%2Fwikitext&config=wikitext-103-raw-v1&split=train"
            f"&offset={offset}&length={length}"
        )
        
        attempt = 0
        while attempt < max_retries:
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                break  # if successful, exit the retry loop
            except requests.exceptions.HTTPError as e:
                if response.status_code == 500:
                    attempt += 1
                    wait_time = 2 ** attempt  # Exponential backoff: 2, 4, 8 seconds...
                    logging.error(f"Attempt {attempt}/{max_retries}: 500 Internal Server Error received. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logging.error(f"HTTP Error: {e}")
                    raise
        else:
            # If we reach here, all retry attempts failed
            error_msg = f"Failed to download WikiText-{version} after {max_retries} attempts due to repeated 500 errors."
            logging.error(error_msg)
            raise requests.exceptions.HTTPError(error_msg)
        
        # Download the response content with progress bar
        total_size = int(response.headers.get('content-length', 0))
        content_chunks = []
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading rows") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    content_chunks.append(chunk)
                    pbar.update(len(chunk))
        content = b"".join(content_chunks)
        
        # Parse JSON response
        rows_data = json.loads(content.decode('utf-8'))
        
        # Save the JSON response to a file
        file_path = data_dir / f"wikitext-103-rows-{offset}-{length}.json"
        with open(file_path, 'w') as f:
            json.dump(rows_data, f, indent=2)
        
        logging.info(f"Downloaded {len(rows_data.get('rows', []))} rows from WikiText-103 API.")
        return rows_data


class CollatorForLanguageModeling:
    """Collator: pad token ID sequences, shift for labels, mask padding."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def __call__(self, examples):
        # Collect raw input_ids from each example
        seqs = [ex['input_ids'] if isinstance(ex['input_ids'], torch.Tensor) else torch.tensor(ex['input_ids']) for ex in examples]
        pad_id = getattr(self.tokenizer, 'pad_token_id', None) or getattr(self.tokenizer, 'eos_token_id', 0)
        # Pad to max length
        full = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=pad_id)
        # Prepare inputs and next-token labels
        input_ids = full[:, :-1]
        labels = full[:, 1:].clone()
        # Mask pad tokens
        labels[labels == pad_id] = -100
        # Attention mask: 1 for real tokens
        attention_mask = (full != pad_id).long()[:, :-1]
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


def get_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    collate_fn: Optional[Callable] = None,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """Create a dataloader from a dataset.
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        collate_fn: Function to collate examples into batches
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory for faster transfer to GPU
        drop_last: Whether to drop the last incomplete batch
        
    Returns:
        PyTorch dataloader
    """
    batch_size = min(batch_size, len(dataset))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def prepare_datasets(
    data_path: Union[str, Path],
    tokenizer: Optional[Union[PreTrainedTokenizer, str]] = None,
    train_seq_length: int = 1024,
    val_seq_length: int = 1024,
    train_stride: Optional[int] = None,
    val_stride: Optional[int] = None,
    val_split: float = 0.1,
    test_split: float = 0.0,
    seed: int = 42,
    **dataset_kwargs
) -> Tuple[Dataset, Dataset, Optional[Dataset]]:
    """Prepare training, validation, and optional test datasets.
    
    Args:
        data_path: Path to data file or directory
        tokenizer: Tokenizer or name of pretrained tokenizer
        train_seq_length: Sequence length for training
        val_seq_length: Sequence length for validation
        train_stride: Stride for training sequences
        val_stride: Stride for validation sequences
        val_split: Fraction of data to use for validation
        test_split: Fraction of data to use for testing
        seed: Random seed for data splitting
        **dataset_kwargs: Additional arguments for dataset creation
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
        test_dataset is None if test_split = 0
    """
    # Set random seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Load full dataset
    full_dataset = TextDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        seq_length=train_seq_length,
        stride=train_stride,
        **dataset_kwargs
    )
    
    # Split dataset
    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    test_size = int(test_split * dataset_size)
    train_size = dataset_size - val_size - test_size
    
    if test_size > 0:
        train_subset, val_subset, test_subset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )
        train_indices = train_subset.indices
        val_indices = val_subset.indices
        test_indices = test_subset.indices
        # Create separate datasets with appropriate sequence lengths
        val_dataset = TextDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            seq_length=val_seq_length,
            stride=val_stride,
            **dataset_kwargs
        )
        val_dataset.examples = [full_dataset.examples[i] for i in val_indices]
        
        test_dataset = TextDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            seq_length=val_seq_length,
            stride=val_stride,
            **dataset_kwargs
        )
        test_dataset.examples = [full_dataset.examples[i] for i in test_indices]
        
        # Update train dataset
        full_dataset.examples = [full_dataset.examples[i] for i in train_indices]
        train_dataset = full_dataset
    else:
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )
        
        # Create a separate validation dataset with appropriate sequence length
        val_dataset_custom = TextDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            seq_length=val_seq_length,
            stride=val_stride,
            **dataset_kwargs
        )
        val_dataset_custom.examples = [full_dataset.examples[i] for i in val_dataset.indices]
        val_dataset = val_dataset_custom
        
        # Update train dataset
        full_dataset.examples = [full_dataset.examples[i] for i in train_dataset.indices]
        train_dataset = full_dataset
        test_dataset = None
    
    return train_dataset, val_dataset, test_dataset