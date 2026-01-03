import os
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset, IterableDataset

def get_tokenizer():
    """Centralized tokenizer loading to ensure consistency."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_data_source(name, split):
    """Helper to load various datasets."""
    # 1. The Small 6B Version (Good for testing/small GPUs)
    if name == "slimpajama_6b":
        return load_dataset("DKYoon/SlimPajama-6B", split=split, streaming=True)
    
    # 2. The Full 627B Version (Massive - requires streaming)
    elif name == "slimpajama_627b":
        return load_dataset("cerebras/SlimPajama-627B", split=split, streaming=True)
        
    # 3. Local Custom Files (JSONL/TXT)
    elif os.path.exists(name):
        return load_dataset("json", data_files=name, split=split, streaming=True)
        
    else:
        raise ValueError(f"Unknown dataset or path: {name}")

def get_dataloader(console, accelerator, dataset_name, batch_size, 
                   seq_len=2048, split="train", num_workers = 4, prefetch_factor = 2):
    """
    The MAIN entry point. Dispatches to the correct loader based on 'dataset_name'.
    """
    # Safe Print (Main Process Only)
    if accelerator.is_main_process:
        console.print(f"[Dataset] Loading '{dataset_name}' (Split: {split})...")

    tokenizer = get_tokenizer()
    raw_dataset = load_data_source(dataset_name, split)
    
    # --- CRITICAL FIX: Only shard the TRAIN split ---
    # We purposefully do NOT shard the 'test' or 'validation' splits.
    # This forces every GPU to process the exact same test data.
    # This guarantees that all GPUs run the EXACT same number of batches, 
    # preventing the loop-exit deadlock.
    if isinstance(raw_dataset, IterableDataset) and accelerator.num_processes > 1:
        if split == "train":
            try:
                raw_dataset = raw_dataset.shard(
                    num_shards=accelerator.num_processes,
                    index=accelerator.process_index
                )
                if accelerator.is_main_process:
                    console.print(f"[dim]Successfully sharded '{split}' split.[/dim]")
            except IndexError:
                if accelerator.is_main_process:
                    console.print(f"[yellow]Warning: Could not shard '{split}'. Falling back to replication.[/yellow]")
        else:
            # For Test/Validation
            if accelerator.is_main_process:
                console.print(f"[dim]Skipping sharding for '{split}' to ensure lockstep evaluation.[/dim]")                
    # --- Auto-detect Column Name ---
    # SlimPajama uses "text", but custom datasets might use "content"
    try:
        sample = next(iter(raw_dataset))
        text_column = "text" if "text" in sample else list(sample.keys())[0]
    except StopIteration:
        text_column = "text"

    # --- Tokenization ---
    def tokenization_fn(examples):
        return tokenizer(
            examples[text_column],
            truncation=True,
            max_length=seq_len,
            padding="max_length",
        )

    # --- Map & Format ---
    # Smart column removal to keep PyTorch tensors clean
    if hasattr(raw_dataset, "column_names"):
        remove_cols = raw_dataset.column_names
    else:
        # Fallback if column_names is missing (common in some streaming modes)
        remove_cols = [text_column, "meta", "red_pajama_subset"]

    mapped_ds = raw_dataset.map(tokenization_fn, remove_columns=remove_cols)
    mapped_ds = mapped_ds.with_format("torch")

    return DataLoader(
        mapped_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True
    ), tokenizer