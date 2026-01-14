import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset, IterableDataset
from datasets import load_from_disk


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
        

def get_dataloader(console, accelerator, tokenizer, dataset_name, batch_size, 
                   seq_len=2048, split="train", num_workers=4, prefetch_factor=2):
    
    if accelerator.is_main_process:
        console.print(f"[Dataset] Loading '{dataset_name}' (Split: {split})...")

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
    )

def create_pg19_dataloader(tokenizer, seq_len, batch_size, split="train"):
    """
    Streams PG19 books, tokenizes them, and packs them into contiguous chunks of seq_len.
    This ensures no padding is used, maximizing efficient training.
    """
    # Load PG19 in streaming mode to avoid downloading 11GB+ immediately
    dataset = datasets.load_dataset("pg19", split=split, streaming=True)

    # Get the EOS Token ID 
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        # Fallback if tokenizer doesn't have an EOS defined (rare, but happens)
        print("Warning: Tokenizer has no eos_token_id. Using 0.")
        eos_token_id = 0
        
    def data_generator():
        buffer = []
        for sample in dataset:
            text = sample['text']
            # Skip short texts
            if len(text) < 1000: continue
            
            tokens = tokenizer(text, add_special_tokens = False).input_ids
            
            buffer.extend(tokens)
            buffer.append(eos_token_id)
            
            # Yield chunks of seq_len
            while len(buffer) >= seq_len:
                chunk = buffer[:seq_len]
                buffer = buffer[seq_len:]
                
                # Convert to tensor
                input_ids = torch.tensor(chunk, dtype=torch.long)
                # Labels are same as input (causal LM)
                yield {"input_ids": input_ids, "labels": input_ids}
                
    # Create iterable dataset
    iterable_dataset = datasets.IterableDataset.from_generator(data_generator)
    
    # Simple collator
    def collate_fn(examples):
        return {
            "input_ids": torch.stack([e["input_ids"] for e in examples]),
            "labels": torch.stack([e["labels"] for e in examples])
        }

    return torch.utils.data.DataLoader(
        iterable_dataset, 
        batch_size=batch_size, 
        collate_fn=collate_fn
    )