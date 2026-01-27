import os
import torch
import itertools
from torch.utils.data import DataLoader, IterableDataset as TorchIterableDataset
from datasets import load_dataset, IterableDataset as HFIterableDataset

def load_data_source(name, split):
    """Helper to load various datasets in streaming mode."""
    if name == "slimpajama_6b":
        return load_dataset("DKYoon/SlimPajama-6B", split=split, streaming=True)
    elif name == "slimpajama_627b":
        return load_dataset("cerebras/SlimPajama-627B", split=split, streaming=True)
    elif name == "wikitext":
        return load_dataset("Salesforce/wikitext", "wikitext-103-v1", split=split, streaming=True)
    elif name == "pg19":
        return load_dataset("emozilla/pg19", split=split, streaming=True)
    elif os.path.exists(name):
        return load_dataset("json", data_files=name, split=split, streaming=True)
    else:
        raise ValueError(f"Unknown dataset or path: {name}")


def get_dataloader(console, accelerator, tokenizer, dataset_name, batch_size, 
                   seq_len=2048, split="train", num_workers=0):
    
    if accelerator.is_main_process:
        console.print(f"[Dataset] Loading '{dataset_name}' (Split: {split}) with PACKING...")

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
        # PG19 is too big for large buffer shuffling, keep it smaller if needed
        buf_size = 10_000 
        raw_dataset = raw_dataset.shuffle(seed=42, buffer_size=buf_size)

    # 4. Define the Packing Generator
    def packed_generator():
        eos_id = tokenizer.eos_token_id
        if eos_id is None:
            if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
                eos_id = tokenizer.pad_token_id
            else:
                eos_id = 0 
        
        buffer = []
        
        # Flag to switch strategies
        is_massive_doc_dataset = (dataset_name == "pg19")

        while True:
            iterator = iter(raw_dataset)

            if not is_sharded and accelerator.num_processes > 1:
                iterator = itertools.islice(iterator, 
                                            accelerator.process_index,
                                            None, 
                                            accelerator.num_processes)

            for sample in iterator:
                text = sample.get("text", "") or sample.get("content", "")
                
                if not text or len(text.strip()) < 2: 
                    continue

                # --- STRATEGY SELECTION ---
                if is_massive_doc_dataset:
                    # STRATEGY A: Chunking (For PG19 / Books)
                    # Slices massive strings to avoid tokenizer hang/OOM
                    chunk_size = 100_000 
                    
                    for i in range(0, len(text), chunk_size):
                        text_chunk = text[i : i + chunk_size]
                        tokens = tokenizer.encode(text_chunk, add_special_tokens=False)
                        buffer.extend(tokens)
                        
                        # Yield immediately to keep memory low
                        while len(buffer) >= seq_len:
                            yield _pack_batch(buffer, seq_len)
                            buffer = buffer[seq_len:]
                            
                else:
                    # STRATEGY B: Standard (For Wikitext / SlimPajama)
                    # Tokenize the whole document at once (Faster for normal docs)
                    tokens = tokenizer.encode(text, add_special_tokens=False)
                    buffer.extend(tokens)
                    
                    # Yield
                    while len(buffer) >= seq_len:
                        yield _pack_batch(buffer, seq_len)
                        buffer = buffer[seq_len:]

                # Append EOS at the end of the document
                buffer.append(eos_id)

    # Helper function to create the batch dictionary
    def _pack_batch(buffer, seq_len):
        chunk = buffer[:seq_len]
        input_ids = torch.tensor(chunk, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone() 
        }

    # 5. Create the Iterable Dataset
    if hasattr(HFIterableDataset, "from_generator"):
        packed_dataset = HFIterableDataset.from_generator(packed_generator)
    else:
        class PyTorchPackedDataset(TorchIterableDataset):
            def __init__(self, generator):
                self.generator = generator
            def __iter__(self):
                return self.generator()
        packed_dataset = PyTorchPackedDataset(packed_generator)

    # 6. Create DataLoader
    return DataLoader(
        packed_dataset,
        batch_size=batch_size,
        num_workers=num_workers, 
        pin_memory=True
    )