import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset, IterableDataset, load_from_disk

def load_data_source(name, split):
    """Helper to load various datasets."""
    # 1. SlimPajama 6B
    if name == "slimpajama_6b":
        return load_dataset("DKYoon/SlimPajama-6B", split=split, streaming=True)
    
    # 2. SlimPajama 627B
    elif name == "slimpajama_627b":
        return load_dataset("cerebras/SlimPajama-627B", split=split, streaming=True)
        
    # 3. Local Custom Files
    elif os.path.exists(name):
        return load_dataset("json", data_files=name, split=split, streaming=True) # Force streaming for packing
        
    else:
        raise ValueError(f"Unknown dataset or path: {name}")


def get_dataloader(console, accelerator, tokenizer, dataset_name, batch_size, 
                   seq_len=2048, split="train", num_workers=0, prefetch_factor=None):
    
    if accelerator.is_main_process:
        console.print(f"[Dataset] Loading '{dataset_name}' (Split: {split}) with PACKING...")

    # 1. Load the raw streaming dataset
    raw_dataset = load_data_source(dataset_name, split)

    # 2. Shuffle (Buffer Shuffling) - Only for training
    if split == "train":
        # Buffer size of 10k is a good balance for RAM vs Randomness
        raw_dataset = raw_dataset.shuffle(seed=42, buffer_size=10_000)

    # 3. Handle Sharding for Multi-GPU (Accelerate)
    # Even with streaming, we need to ensure each GPU gets different data
    if accelerator.num_processes > 1:
        # HuggingFace IterableDataset supports sharding at the file/stream level
        raw_dataset = raw_dataset.shard(
            num_shards=accelerator.num_processes,
            index=accelerator.process_index
        )

    # 4. Define the Packing Generator
    # This replaces the .map() function. It tokenizes and concatenates streams.
    def packed_generator():
        # Get EOS token (Critical for separating documents)
        eos_id = tokenizer.eos_token_id
        if eos_id is None:
            eos_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
            if eos_id is None: eos_id = 0 # Final fallback
        
        buffer = []
        
        # Determine text column name dynamically
        iterator = iter(raw_dataset)
        try:
            first_item = next(iterator)
            text_col = "text" if "text" in first_item else list(first_item.keys())[0]
            
            # Put the first item back (conceptually) or process it
            # Since we can't put back in iterator, we process it first
            item_list = [first_item]
        except StopIteration:
            return

        # Chain the first item with the rest of the iterator
        import itertools
        full_iterator = itertools.chain(item_list, iterator)

        for sample in full_iterator:
            text = sample[text_col]
            
            # Basic filtering for empty lines
            if not text or len(text) < 10: 
                continue

            # Tokenize FAST (No padding, no truncation yet)
            tokens = tokenizer.encode(text, add_special_tokens=False)
            
            # Add to buffer with EOS separator
            buffer.extend(tokens)
            buffer.append(eos_id)

            # Yield full chunks
            while len(buffer) >= seq_len:
                chunk = buffer[:seq_len]
                buffer = buffer[seq_len:] # Keep remainder
                
                # Yield tensor
                yield {
                    "input_ids": torch.tensor(chunk, dtype=torch.long),
                    "labels": torch.tensor(chunk, dtype=torch.long)
                }

    # 5. Create the Iterable Dataset
    packed_dataset = IterableDataset.from_generator(packed_generator)

    # 6. Create DataLoader
    # Note: num_workers must be 0 for simple IterableDatasets usually, 
    # unless using specific torchdata pipes, to avoid duplication issues.
    return DataLoader(
        packed_dataset,
        batch_size=batch_size,
        num_workers=0, # Keep 0 for safety with generators
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