import torch
import os
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# --- CONFIGURATION ---
DATASET_NAME = "DKYoon/SlimPajama-6B"
OUTPUT_FILE = "./frozen_data/slimpajama_test_packed.pt" # Single file is fine for test set (small)
SEQ_LEN = 2048
SPLIT = "test"  # Benchmarking on test split

def pack_and_freeze():
    # 1. Setup
    if not os.path.exists(os.path.dirname(OUTPUT_FILE)):
        os.makedirs(os.path.dirname(OUTPUT_FILE))
        
    print(f"⚡ Downloading and Packing {DATASET_NAME} [{SPLIT}]...")
    
    # Load Tokenizer (Use GPT-2 as standard for comparison)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = 50256
    
    EOS_ID = tokenizer.eos_token_id

    # 2. Load Dataset (Streaming=False to ensure we get everything first, it's small)
    # The Test split of SlimPajama-6B is only ~9k rows (~5MB), so we can download it fully.
    dataset = load_dataset(DATASET_NAME, split=SPLIT, streaming=False)
    
    buffer = []
    packed_samples = []
    
    print(f"🔄 Processing {len(dataset)} documents...")
    
    for item in tqdm(dataset):
        text = item['text']
        
        # Tokenize without padding
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        # Add to buffer with EOS separator
        buffer.extend(tokens)
        buffer.append(EOS_ID)
        
        # Drain buffer into 2048 chunks
        while len(buffer) >= SEQ_LEN:
            chunk = buffer[:SEQ_LEN]
            buffer = buffer[SEQ_LEN:] # Keep remainder
            
            packed_samples.append(torch.tensor(chunk, dtype=torch.long))

    # 3. Save as a single Tensor [N, SEQ_LEN]
    if packed_samples:
        all_data = torch.stack(packed_samples)
        print(f"✅ Saving {all_data.shape[0]} sequences to {OUTPUT_FILE}...")
        torch.save(all_data, OUTPUT_FILE)
        print(f"Total Tokens: {all_data.numel()}")
    else:
        print("⚠️ Warning: Dataset was too small to create even one sequence!")

if __name__ == "__main__":
    pack_and_freeze()