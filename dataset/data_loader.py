import os
import torch
import itertools
from torch.utils.data import DataLoader, IterableDataset as TorchIterableDataset
from datasets import load_dataset

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
        return load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split, streaming=True)
    elif name == "pg19":
        return load_dataset("emozilla/pg19", split=split, streaming=True)
    elif name == "tinystories":
        return load_dataset("roneneldan/TinyStories", split=split, streaming=True)
        
    elif name == "fineweb-edu":
        # ... (Same as before) ...
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        val_size = 5000 
        if split == "validation":
            return ds.take(val_size)
        else:
            return ds.skip(val_size)
            
    elif os.path.exists(name):
        return load_dataset("json", data_files=name, split=split, streaming=True)
    else:
        raise ValueError(f"Unknown dataset or path: {name}")

# --- REMOVED fast_skip_batches from arguments ---
def _packed_generator_func(raw_dataset, tokenizer_encode_func, eos_id, seq_len, 
                           batch_size, dataset_name, 
                           num_processes, process_index, is_sharded, is_main_process):
    
    buffer = []
    # Removed: skipped_counter = 0 
    is_massive_doc_dataset = (dataset_name == "pg19")
    
    iterator = iter(raw_dataset)

    if not is_sharded and num_processes > 1:
        iterator = itertools.islice(iterator, process_index, None, num_processes)

    for sample in iterator:
        text = sample.get("text", "") or sample.get("content", "")
        if not text or len(text.strip()) < 2: 
            continue

        # --- TOKENIZATION ---
        if is_massive_doc_dataset:
            chunk_char_size = 100_000 
            for i in range(0, len(text), chunk_char_size):
                text_chunk = text[i : i + chunk_char_size]
                tokens = tokenizer_encode_func(text_chunk, add_special_tokens=False)
                
                if i + chunk_char_size >= len(text):
                    tokens.append(eos_id)
                
                buffer.extend(tokens)
                
                while len(buffer) >= seq_len:
                    # --- REMOVED MANUAL SKIPPING LOGIC HERE ---
                    chunk = buffer[:seq_len]
                    input_ids = torch.tensor(chunk, dtype=torch.long)
                    yield {"input_ids": input_ids, "labels": input_ids.clone()}
                    buffer = buffer[seq_len:]
        else:
            tokens = tokenizer_encode_func(text, add_special_tokens=False)
            tokens.append(eos_id)
            buffer.extend(tokens)
            
            while len(buffer) >= seq_len:
                # --- REMOVED MANUAL SKIPPING LOGIC HERE ---
                chunk = buffer[:seq_len]
                input_ids = torch.tensor(chunk, dtype=torch.long)
                yield {"input_ids": input_ids, "labels": input_ids.clone()}
                buffer = buffer[seq_len:]



# --- REMOVED fast_skip_batches arg ---
def get_dataloader(console, accelerator, tokenizer, dataset_name, batch_size, 
                   seq_len=2048, split="train", num_workers=0):
    
    if accelerator.is_main_process:
        console.print(f"[Dataset] Loading '{dataset_name}' (Split: {split}) with PACKING...")

    raw_dataset = load_data_source(dataset_name, split)

    is_sharded = False
    try:
        if accelerator.num_processes > 1:
            raw_dataset = raw_dataset.shard(num_shards=accelerator.num_processes, index=accelerator.process_index)
            is_sharded = True
    except Exception as e:
        if accelerator.is_main_process:
            console.print(f"[Warning] Sharding failed ({e}). Using manual fallback.")
        is_sharded = False

    if split == "train":
        raw_dataset = raw_dataset.shuffle(seed=42, buffer_size=10_000)

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = getattr(tokenizer, "pad_token_id", 0)
            
    # Removed fast_skip_batches from kwargs
    gen_kwargs = {
        "raw_dataset": raw_dataset,
        "tokenizer_encode_func": tokenizer.encode,
        "eos_id": eos_id,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "dataset_name": dataset_name,
        "num_processes": accelerator.num_processes,
        "process_index": accelerator.process_index,
        "is_sharded": is_sharded,
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