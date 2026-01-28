import os
import torch
from torch.utils.data import DataLoader, IterableDataset as TorchIterableDataset
from datasets import load_dataset

class PyTorchPackedDataset(TorchIterableDataset):
    def __init__(self, generator, generator_kwargs):
        self.generator = generator
        self.generator_kwargs = generator_kwargs
        
    def __iter__(self):
        return self.generator(**self.generator_kwargs)

def load_data_source(name, split, num_shards=1, process_index=0):
    ds = None
    if name == "fineweb-edu":
        # 1. Load base dataset
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        
        # 2. Shard FIRST
        if num_shards > 1:
            try:
                ds = ds.shard(num_shards=num_shards, index=process_index)
            except Exception as e:
                print(f"[Warning] Failed to shard dataset natively: {e}")

        # 3. Apply Splits
        global_val_size = 5000
        local_val_size = global_val_size // num_shards
        if split == "validation":
            return ds.take(local_val_size)
        else:
            return ds.skip(local_val_size)
    
    elif os.path.exists(name):
        ds = load_dataset("json", data_files=name, split=split, streaming=True)
    else:
        # Fallback for others
        ds = load_dataset(name, split=split, streaming=True)

    if num_shards > 1 and name != "fineweb-edu":
         ds = ds.shard(num_shards=num_shards, index=process_index)
         
    return ds

def _packed_generator_func(raw_dataset, tokenizer_encode_func, eos_id, seq_len, 
                           batch_size, fast_skip_sequences, dataset_name, 
                           num_processes, process_index, is_main_process):
    
    buffer = []
    skipped_counter = 0 
    iterator = iter(raw_dataset)

    # Note: We must tokenize even when skipping because we need to know the token length
    # to find the sequence boundaries.
    
    for sample in iterator:
        text = sample.get("text", "") or sample.get("content", "")
        if not text or len(text.strip()) < 2: 
            continue

        tokens = tokenizer_encode_func(text, add_special_tokens=False)
        tokens.append(eos_id)
        buffer.extend(tokens)
        
        while len(buffer) >= seq_len:
            # FAST SKIP LOGIC
            if skipped_counter < fast_skip_sequences:
                # Discard buffer slice without creating tensors
                del buffer[:seq_len] 
                skipped_counter += 1
                
                # UI Feedback for long resumes
                if skipped_counter % 500 == 0 and is_main_process:
                    print(f"[Resuming] Skipped {skipped_counter}/{fast_skip_sequences} sequences...", end="\r", flush=True)
                continue

            chunk = buffer[:seq_len]
            # Use 'del' to manage memory efficiently
            del buffer[:seq_len]
            
            input_ids = torch.tensor(chunk, dtype=torch.long)
            yield {"input_ids": input_ids, "labels": input_ids.clone()}

def get_dataloader(console, accelerator, tokenizer, dataset_name, batch_size, 
                   seq_len=2048, split="train", num_workers=0, fast_skip_sequences=0):
    
    if accelerator.is_main_process:
        console.print(f"[Dataset] Loading '{dataset_name}' (Split: {split})...")
        if fast_skip_sequences > 0:
            console.print(f"[Dataset] Resuming: Skipping first {fast_skip_sequences} sequences (approx {fast_skip_sequences/batch_size:.0f} batches).")

    raw_dataset = load_data_source(
        dataset_name, 
        split, 
        num_shards=accelerator.num_processes, 
        process_index=accelerator.process_index
    )

    if split == "train":
        raw_dataset = raw_dataset.shuffle(seed=42, buffer_size=10_000)

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = getattr(tokenizer, "pad_token_id", 0)
            
    gen_kwargs = {
        "raw_dataset": raw_dataset,
        "tokenizer_encode_func": tokenizer.encode,
        "eos_id": eos_id,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "fast_skip_sequences": fast_skip_sequences,
        "dataset_name": dataset_name,
        "num_processes": accelerator.num_processes,
        "process_index": accelerator.process_index,
        "is_main_process": accelerator.is_main_process
    }

    packed_dataset = PyTorchPackedDataset(
        generator=_packed_generator_func, 
        generator_kwargs=gen_kwargs
    )

    return DataLoader(
        packed_dataset,
        batch_size=batch_size,
        num_workers=num_workers, 
        pin_memory=True
    )