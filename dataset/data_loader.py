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
        return load_dataset("DKYoon/SlimPajama-6B", split=split, streaming=False)
    
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
    """
    The MAIN entry point. Optimized for distributed training stability.
    """
    
    if accelerator.is_main_process:
        console.print(f"[Dataset] Loading '{dataset_name}' (Split: {split})...")

    tokenizer = get_tokenizer()
    raw_dataset = load_data_source(dataset_name, split)
    
    # --- Tokenization Logic ---
    def tokenization_fn(examples):
        return tokenizer(
            examples["text"], # SlimPajama-6B uses "text" column
            truncation=True,
            max_length=seq_len,
            padding="max_length",
        )

    # --- Pre-Processing & Tokenization ---
    if not isinstance(raw_dataset, IterableDataset):
        # === CRITICAL FIX: Prevent Deadlocks ===
        # We use accelerator.main_process_first() to force Rank 0 to do the heavy lifting
        # (downloading & tokenizing & caching) while other ranks wait.
        # Once Rank 0 is done, the data is cached on disk.
        # Ranks 1+ then wake up and instantly load the cache (0s duration).
        with accelerator.main_process_first():
            
            # Print only on main process to keep logs clean
            if accelerator.is_main_process:
                console.print(f"[yellow]Tokenizing {split} dataset... (This runs once & caches)[/yellow]")
            
            # Map with full CPU parallelism
            # NOTE: We use os.cpu_count() because inside this block, 
            # only ONE process is running, so we can hog all the cores safely.
            mapped_ds = raw_dataset.map(
                tokenization_fn,
                batched=True,
                num_proc=os.cpu_count(), 
                remove_columns=raw_dataset.column_names,
                desc=f"Tokenizing {split}",
                load_from_cache_file=True # Uses disk cache if available
            )
            
        mapped_ds = mapped_ds.with_format("torch")
        
        # True Global Shuffle for downloaded datasets
        should_shuffle = (split == "train")
        
    else:
        # Fallback for Streaming Datasets (like 627B)
        # We cannot use main_process_first here because streaming happens on the fly.
        mapped_ds = raw_dataset.map(tokenization_fn, remove_columns=["text", "meta", "red_pajama_subset"])
        mapped_ds = mapped_ds.with_format("torch")
        should_shuffle = False # Streaming datasets shuffle via buffer, not here

    # --- Create DataLoader ---
    return DataLoader(
        mapped_ds,
        batch_size=batch_size,
        shuffle=should_shuffle, 
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor
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