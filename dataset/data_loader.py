import os
import torch
import itertools
from torch.utils.data import DataLoader, IterableDataset as TorchIterableDataset
from datasets import load_dataset, IterableDataset as HFIterableDataset

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
    
    # 4. PG19 (Project Gutenberg)
    elif name == "pg19":
        return load_dataset("emozilla/pg19", split=split, streaming=True)
                            
    # 5. Local Custom Files
    elif os.path.exists(name):
        # Force streaming so we can use the same logic for local JSON/Text files
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
    # We attempt to use HF's efficient .shard(). If the dataset has fewer files 
    # than GPUs (rare for big datasets), we fall back to manual skipping.
    is_sharded = False
    try:
        if accelerator.num_processes > 1:
            # This assigns specific files to specific GPUs (Zero redundancy)
            raw_dataset = raw_dataset.shard(
                num_shards=accelerator.num_processes,
                index=accelerator.process_index
            )
            is_sharded = True
    except Exception as e:
        # This usually happens if the dataset doesn't support file-level sharding
        if accelerator.is_main_process:
            console.print(f"[Warning] Could not shard at file level: {e}. Using manual skipping fallback.")
        is_sharded = False

    # 3. Buffer Shuffling (Only for training)
    if split == "train":
        # Buffer size balances randomness vs RAM usage. 10k is usually safe.
        raw_dataset = raw_dataset.shuffle(seed=42, buffer_size=10_000)

    # 4. Define the Packing Generator
    def packed_generator():
        # Auto-detect EOS token, fallback to 0 if missing
        eos_id = tokenizer.eos_token_id
        if eos_id is None:
            if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
                eos_id = tokenizer.pad_token_id
            else:
                eos_id = 0 
        
        buffer = []
        
        # --- INFINITE LOOP FOR TRAINING ---
        # We loop forever if split="train". This prevents deadlocks where one GPU 
        # finishes 1 batch earlier than others and waits forever.
        while True:
            iterator = iter(raw_dataset)

            # Manual skipping fallback (only if .shard() failed above)
            if not is_sharded and accelerator.num_processes > 1:
                # This is less efficient but guarantees correctness
                iterator = itertools.islice(iterator, 
                                            accelerator.process_index,
                                            None, 
                                            accelerator.num_processes)

            for sample in iterator:
                # Handle different column names (text vs content)
                text = sample.get("text", "") or sample.get("content", "")
                
                if not text or len(text.strip()) < 2: 
                    continue

                tokens = tokenizer.encode(text, add_special_tokens=False)
                
                buffer.extend(tokens)
                buffer.append(eos_id)

                # Yield chunks of exactly seq_len
                while len(buffer) >= seq_len:
                    chunk = buffer[:seq_len]
                    buffer = buffer[seq_len:] 
                    
                    input_ids = torch.tensor(chunk, dtype=torch.long)
                    
                    yield {
                        "input_ids": input_ids,
                        "labels": input_ids.clone() 
                    }
            
            # If VALIDATION, stop here (we don't want infinite validation loops)
            if split != "train":
                break
            # If TRAINING, the loop restarts automatically (Infinite)

    # 5. Create the Iterable Dataset (Compatibility Mode)
    # Checks if the user has the new HF method. If not, uses a standard PyTorch wrapper.
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