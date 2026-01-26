import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset, IterableDataset, load_from_disk
import itertools

def load_data_source(name, split):
    """Helper to load various datasets."""
    # 1. SlimPajama 6B
    if name == "slimpajama_6b":
        return load_dataset("DKYoon/SlimPajama-6B", split=split, streaming=True)
    
    # 2. SlimPajama 627B
    elif name == "slimpajama_627b":
        return load_dataset("cerebras/SlimPajama-627B", split=split, streaming=True)

    # 3. WikiText-103 
    elif name == "wikitext":
        return load_dataset("wikitext", "wikitext-103-v1", split=split, streaming=True)
    
    # 4, PG19 (Project Gutenberg)
    elif name == "pg19":
        return load_dataet("google-deepmind/pg19", split = split, streaming = True)
                           
    # 3. Local Custom Files
    elif os.path.exists(name):
        return load_dataset("json", data_files=name, split=split, streaming=True) # Force streaming for packing
        
    else:
        raise ValueError(f"Unknown dataset or path: {name}")


def get_dataloader(console, accelerator, tokenizer, dataset_name, batch_size, 
                   seq_len=2048, split="train", num_workers=0):
    
    if accelerator.is_main_process:
        console.print(f"[Dataset] Loading '{dataset_name}' (Split: {split}) with PACKING...")

    # 1. Load the raw streaming dataset
    raw_dataset = load_data_source(dataset_name, split)

    # 2. Shuffle (Buffer Shuffling) - Only for training
    if split == "train":
        # Buffer size of 10k balances RAM usage vs Randomness
        raw_dataset = raw_dataset.shuffle(seed=42, buffer_size=10_000)

    # 3. Handle Sharding for Multi-GPU (Accelerate)
    # If the number of files is less than GPUs, It will cause error (Disable)
    # if accelerator.num_processes > 1:
    #     raw_dataset = raw_dataset.shard(
    #         num_shards=accelerator.num_processes,
    #         index=accelerator.process_index
    #     )

    # 4. Define the Packing Generator
    # 3. Define the Packing Generator with Manual Sharding
    def packed_generator():
        # Get EOS token
        eos_id = tokenizer.eos_token_id
        if eos_id is None:
            if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
                eos_id = tokenizer.pad_token_id
            else:
                eos_id = 0 
        
        buffer = []
        
        # Iterator setup
        iterator = iter(raw_dataset)
        try:
            first_item = next(iterator)
            text_col = "text" if "text" in first_item else list(first_item.keys())[0]
            full_iterator = itertools.chain([first_item], iterator)
        except StopIteration:
            return

        # ### FIX: MANUAL INTERLEAVING ###
        # We manually skip samples that don't belong to this GPU.
        # This works for ANY dataset, even if it's just 1 file.
        process_rank = accelerator.process_index
        num_processes = accelerator.num_processes

        for i, sample in enumerate(full_iterator):
            # If we have 2 GPUs:
            # GPU 0 processes indices: 0, 2, 4...
            # GPU 1 processes indices: 1, 3, 5...
            if i % num_processes != process_rank:
                continue

            text = sample[text_col]
            
            if not text or len(text.strip()) < 2: 
                continue

            tokens = tokenizer.encode(text, add_special_tokens=False)
            
            buffer.extend(tokens)
            buffer.append(eos_id)

            while len(buffer) >= seq_len:
                chunk = buffer[:seq_len]
                buffer = buffer[seq_len:] 
                
                input_ids = torch.tensor(chunk, dtype=torch.long)
                
                yield {
                    "input_ids": input_ids,
                    "labels": input_ids.clone() 
                }
    # 5. Create the Iterable Dataset
    packed_dataset = IterableDataset.from_generator(packed_generator)

    # 6. Create DataLoader
    return DataLoader(
        packed_dataset,
        batch_size=batch_size,
        num_workers=num_workers, 
        pin_memory=True
    )
    
# def create_pg19_dataloader(tokenizer, seq_len, batch_size, split="train"):
#     """
#     Streams PG19 books, tokenizes them, and packs them into contiguous chunks of seq_len.
#     This ensures no padding is used, maximizing efficient training.
#     """
#     # Load PG19 in streaming mode to avoid downloading 11GB+ immediately
#     dataset = datasets.load_dataset("pg19", split=split, streaming=True)

#     # Get the EOS Token ID 
#     eos_token_id = tokenizer.eos_token_id
#     if eos_token_id is None:
#         # Fallback if tokenizer doesn't have an EOS defined (rare, but happens)
#         print("Warning: Tokenizer has no eos_token_id. Using 0.")
#         eos_token_id = 0
        
#     def data_generator():
#         buffer = []
#         for sample in dataset:
#             text = sample['text']
#             # Skip short texts
#             if len(text) < 1000: continue
            
#             tokens = tokenizer(text, add_special_tokens = False).input_ids
            
#             buffer.extend(tokens)
#             buffer.append(eos_token_id)
            
#             # Yield chunks of seq_len
#             while len(buffer) >= seq_len:
#                 chunk = buffer[:seq_len]
#                 buffer = buffer[seq_len:]
                
#                 # Convert to tensor
#                 input_ids = torch.tensor(chunk, dtype=torch.long)
#                 # Labels are same as input (causal LM)
#                 yield {"input_ids": input_ids, "labels": input_ids}
                
#     # Create iterable dataset
#     iterable_dataset = datasets.IterableDataset.from_generator(data_generator)
    
#     # Simple collator
#     def collate_fn(examples):
#         return {
#             "input_ids": torch.stack([e["input_ids"] for e in examples]),
#             "labels": torch.stack([e["labels"] for e in examples])
#         }

#     return torch.utils.data.DataLoader(
#         iterable_dataset, 
#         batch_size=batch_size, 
#         collate_fn=collate_fn
#     )