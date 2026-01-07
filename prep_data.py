import os
import multiprocessing
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm.auto import tqdm  # <--- Import tqdm

# Config
DATASET_NAME = "DKYoon/SlimPajama-6B"
SEQ_LEN = 2048
CACHE_DIR = "./cached_slimpajama_6b"  # Where we save the processed data

def main():
    # Use tqdm.write instead of print so it doesn't mess up the progress bars
    tqdm.write(f"--- Starting Data Prep for {DATASET_NAME} ---")
    tqdm.write(f"Target Cache Directory: {CACHE_DIR}")
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # 1. Load the raw dataset
    tqdm.write("Loading dataset...")
    # 'load_dataset' has its own internal progress bar for downloading
    raw_datasets = load_dataset(DATASET_NAME, streaming=False)
    
    def tokenization_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=SEQ_LEN,
            padding="max_length",
        )

    # 2. Tokenize with MAX parallelism
    num_cores = os.cpu_count()
    tqdm.write(f"Tokenizing with {num_cores} CPU cores...")

    # The .map() function AUTOMATICALLY uses tqdm if you provide 'desc'
    tokenized_datasets = raw_datasets.map(
        tokenization_fn,
        batched=True,
        num_proc=num_cores,
        remove_columns=["text", "meta"],
        desc="Tokenizing",  # <--- This string appears next to the progress bar
    )

    # 3. Save to Disk
    tqdm.write("Saving tokenized dataset to disk... (This may take a while)")
    tokenized_datasets.save_to_disk(CACHE_DIR)
    tqdm.write(f"Done! Data saved to {CACHE_DIR}")

if __name__ == "__main__":
    main()