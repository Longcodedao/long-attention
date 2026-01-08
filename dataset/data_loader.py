import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset, IterableDataset
from datasets import load_from_disk

def get_tokenizer():
    """Centralized tokenizer loading to ensure consistency."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_data_source(name, split):
    """Helper to load various datasets."""
    # 1. The Small 6B Version -> DOWNLOAD IT (streaming=False)
    # This allows global shuffling and faster training after the initial download.
    if name == "slimpajama_6b":
        return load_dataset("DKYoon/SlimPajama-6B", split=split, streaming=True)
    
    # 2. The Full 627B Version (Massive - requires streaming)
    elif name == "slimpajama_627b":
        return load_dataset("cerebras/SlimPajama-627B", split=split, streaming=True)
        
    # 3. Local Custom Files (JSONL/TXT)
    elif os.path.exists(name):
        # Local files are usually small enough to not stream
        return load_dataset("json", data_files=name, split=split, streaming=False)
        
    else:
        raise ValueError(f"Unknown dataset or path: {name}")
        

def get_dataloader(console, accelerator, dataset_name, batch_size, 
                   seq_len=2048, split="train", num_workers=4, prefetch_factor=2):
    
    if accelerator.is_main_process:
        console.print(f"[Dataset] Loading '{dataset_name}' (Split: {split})...")

    tokenizer = get_tokenizer()
    raw_dataset = load_data_source(dataset_name, split)
    
    # 1. SHUFFLE (Training Only)
    # Essential for streaming datasets to break correlation
    if split == "train":
        # Buffer size depends on RAM. 10,000 is a safe starting point.
        raw_dataset = raw_dataset.shuffle(seed=42, buffer_size=10_000)

    # 2. SHARDING (Training Only)
    if isinstance(raw_dataset, IterableDataset) and accelerator.num_processes > 1:
        if split == "train":
            raw_dataset = raw_dataset.shard(
                num_shards=accelerator.num_processes,
                index=accelerator.process_index
            )
    
    # 3. DYNAMIC COLUMN DETECTION
    # Peek at the first item to know exactly what columns to remove
    try:
        sample = next(iter(raw_dataset))
        # Detect text column if not standard
        text_column = "text" if "text" in sample else list(sample.keys())[0]
        # remove ALL columns that existed in the raw data
        remove_cols = list(sample.keys()) 
    except StopIteration:
        # Handle empty dataset edge case
        text_column = "text"
        remove_cols = ["text", "meta", "red_pajama_subset"]

    # 4. TOKENIZATION
    def tokenization_fn(examples):
        return tokenizer(
            examples[text_column],
            truncation=True,
            max_length=seq_len,
            padding="max_length",
        )

    # 5. BATCHED MAPPING (Critical for Speed)
    mapped_ds = raw_dataset.map(
        tokenization_fn, 
        batched=True,           # <--- ENABLES FAST RUST TOKENIZATION
        batch_size=1000,        # Process 1k texts at a time
        remove_columns=remove_cols
    )
    mapped_ds = mapped_ds.with_format("torch")

    return DataLoader(
        mapped_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True
    ), tokenizer

# def get_dataloader(console, accelerator, dataset_name, batch_size, 
#                    seq_len=2048, split="train", num_workers=4, prefetch_factor=2):
#     """
#     Optimized DataLoader that loads pre-tokenized data from disk.
#     Requires running 'prepare_data.py' first.
#     """
    
#     # 1. Define the cache path (must match what you used in prepare_data.py)
#     CACHE_PATH = "./cached_slimpajama_6b" 
    
#     if accelerator.is_main_process:
#         console.print(f"[Dataset] Loading '{split}' split from disk cache: {CACHE_PATH}...")

#     # 2. Check if cache exists
#     if not os.path.exists(CACHE_PATH):
#         error_msg = f"Cache directory '{CACHE_PATH}' not found! Please run 'prepare_data.py' first."
#         if accelerator.is_main_process:
#             console.print(f"[red]{error_msg}[/red]")
#         raise FileNotFoundError(error_msg)

#     # 3. Load from Disk (Instant)
#     # load_from_disk uses memory mapping, so it is extremely fast and RAM efficient.
#     try:
#         tokenized_ds = load_from_disk(CACHE_PATH)
        
#         # Select the correct split
#         if split not in tokenized_ds:
#              if accelerator.is_main_process:
#                 console.print(f"[yellow]Warning: Split '{split}' not found. Available: {list(tokenized_ds.keys())}[/yellow]")
#              # Fail gracefully or default to train if acceptable
#              raise KeyError(f"Split '{split}' not found in cached dataset.")
#         else:
#             dataset = tokenized_ds[split]

#     except Exception as e:
#         raise RuntimeError(f"Failed to load dataset from disk: {e}")

#     # 4. Format for PyTorch
#     # The data is already tokenized; we just need to ensure it outputs tensors.
#     dataset = dataset.with_format("torch")

#     # 5. Create DataLoader
#     # pin_memory=True speeds up transfer to GPU
#     return DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=(split == "train"), # Only shuffle training data
#         num_workers=num_workers,
#         pin_memory=True,
#         prefetch_factor=prefetch_factor
#     ), None # Tokenizer is not needed for the loader anymore