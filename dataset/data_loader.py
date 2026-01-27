import os
import torch
import itertools
from torch.utils.data import DataLoader, IterableDataset as TorchIterableDataset
from datasets import load_dataset

# Define the wrapper class at module level to ensure it's pickleable if needed
class PyTorchPackedDataset(TorchIterableDataset):
    def __init__(self, generator, generator_kwargs):
        self.generator = generator
        self.generator_kwargs = generator_kwargs
        
    def __iter__(self):
        return self.generator(**self.generator_kwargs)

def load_data_source(name, split):
    """Helper to load various datasets in streaming mode."""
    if name == "slimpajama_6b":
        return load_dataset("DKYoon/SlimPajama-6B", split=split, streaming=True)
    elif name == "slimpajama_627b":
        return load_dataset("cerebras/SlimPajama-627B", split=split, streaming=True)
    elif name == "wikitext":
        return load_dataset("Salesforce/wikitext", "wikitext-103-v1", split=split, streaming=True)
    elif name == "wikitext-2":
        # We use the 'raw' version to avoid pre-tokenization issues, allowing your tokenizer to handle it
        return load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split, streaming=True)
    elif name == "pg19":
        return load_dataset("emozilla/pg19", split=split, streaming=True)
    elif os.path.exists(name):
        return load_dataset("json", data_files=name, split=split, streaming=True)
    else:
        raise ValueError(f"Unknown dataset or path: {name}")

# --- MOVED GENERATOR LOGIC OUTSIDE TO AVOID CLOSURE ISSUES ---
def _packed_generator_func(raw_dataset, tokenizer_encode_func, eos_id, seq_len, 
                           batch_size, fast_skip_batches, dataset_name, 
                           num_processes, process_index, is_sharded, is_main_process):
    
    buffer = []
    skipped_counter = 0 
    is_massive_doc_dataset = (dataset_name == "pg19")
    
    iterator = iter(raw_dataset)

    # Manual process sharding if file-level sharding failed
    if not is_sharded and num_processes > 1:
        iterator = itertools.islice(iterator, process_index, None, num_processes)

    while True:
        try:
            sample = next(iterator)
        except StopIteration:
            break

        text = sample.get("text", "") or sample.get("content", "")
        
        if not text or len(text.strip()) < 2: 
            continue

        # --- STRATEGY SELECTION ---
        if is_massive_doc_dataset:
            # STRATEGY A: Chunking (For PG19 / Books)
            chunk_size = 100_000 
            
            for i in range(0, len(text), chunk_size):
                text_chunk = text[i : i + chunk_size]
                tokens = tokenizer_encode_func(text_chunk, add_special_tokens=False)
                buffer.extend(tokens)
                
                while len(buffer) >= seq_len:
                    # --- FAST SKIP LOGIC ---
                    if skipped_counter < fast_skip_batches:
                        buffer = buffer[seq_len:]
                        skipped_counter += 1
                        if skipped_counter % 1000 == 0 and is_main_process:
                            print(f"[Resuming] Skipped {skipped_counter}/{fast_skip_batches} batches...", end="\r", flush = True)
                        continue
                    # -----------------------

                    # Yield batch dictionary
                    chunk = buffer[:seq_len]
                    input_ids = torch.tensor(chunk, dtype=torch.long)
                    yield {
                        "input_ids": input_ids,
                        "labels": input_ids.clone() 
                    }
                    buffer = buffer[seq_len:]
                    
        else:
            # STRATEGY B: Standard
            tokens = tokenizer_encode_func(text, add_special_tokens=False)
            buffer.extend(tokens)
            
            while len(buffer) >= seq_len:
                # --- FAST SKIP LOGIC ---
                if skipped_counter < fast_skip_batches:
                    buffer = buffer[seq_len:]
                    skipped_counter += 1
                    if skipped_counter % 1000 == 0 and is_main_process:
                            print(f"[Resuming] Skipped {skipped_counter}/{fast_skip_batches} batches...", end="\r")
                    continue
                # -----------------------

                chunk = buffer[:seq_len]
                input_ids = torch.tensor(chunk, dtype=torch.long)
                yield {
                    "input_ids": input_ids,
                    "labels": input_ids.clone() 
                }
                buffer = buffer[seq_len:]

        # Append EOS
        buffer.append(eos_id)


def get_dataloader(console, accelerator, tokenizer, dataset_name, batch_size, 
                   seq_len=2048, split="train", num_workers=0, fast_skip_batches=0):
    
    if accelerator.is_main_process:
        console.print(f"[Dataset] Loading '{dataset_name}' (Split: {split}) with PACKING...")
        if fast_skip_batches > 0:
            console.print(f"[Dataset] Fast-forwarding enabled: Skipping first {fast_skip_batches} batches internally.")

    # 1. Load the raw streaming dataset
    raw_dataset = load_data_source(dataset_name, split)

    # 2. Efficient Sharding Logic
    is_sharded = False
    try:
        if accelerator.num_processes > 1:
            raw_dataset = raw_dataset.shard(
                num_shards=accelerator.num_processes,
                index=accelerator.process_index
            )
            is_sharded = True
    except Exception as e:
        if accelerator.is_main_process:
            console.print(f"[Warning] Could not shard at file level: {e}. Using manual skipping fallback.")
        is_sharded = False

    # 3. Buffer Shuffling (Only for training)
    if split == "train":
        buf_size = 10_000 
        raw_dataset = raw_dataset.shuffle(seed=42, buffer_size=buf_size)

    # 4. Prepare Generator Arguments
    # We extract strictly simple types here to pass to the generator
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
            eos_id = tokenizer.pad_token_id
        else:
            eos_id = 0 
            
    gen_kwargs = {
        "raw_dataset": raw_dataset,
        "tokenizer_encode_func": tokenizer.encode,
        "eos_id": eos_id,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "fast_skip_batches": fast_skip_batches,
        "dataset_name": dataset_name,
        "num_processes": accelerator.num_processes,
        "process_index": accelerator.process_index,
        "is_sharded": is_sharded,
        "is_main_process": accelerator.is_main_process
    }

    # 5. Create the Iterable Dataset
    # We use the custom PyTorch wrapper DIRECTLY to avoid HF's pickling/hashing of the accelerator
    packed_dataset = PyTorchPackedDataset(
        generator=_packed_generator_func, 
        generator_kwargs=gen_kwargs
    )

    # 6. Create DataLoader
    return DataLoader(
        packed_dataset,
        batch_size=batch_size,
        num_workers=num_workers, 
        pin_memory=True
    )