import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import json

# --- UI IMPORTS ---
from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn, MofNCompleteColumn
)

# --- MODEL IMPORTS ---
try:
    from model.holo import HoloConfig, HoloForCausalLM
except ImportError:
    print("Warning: Holo model not found.")

try:
    from mamba_ssm import MambaConfig, MambaForCausalLM
    MAMBA_LIB = "official"
except ImportError:
    try:
        from transformers import MambaConfig, MambaForCausalLM
        MAMBA_LIB = "transformers"
    except ImportError:
        MAMBA_LIB = None
        print("Warning: Mamba model not found.")

# ==========================================
# 1. NEEDLE DATASET
# ==========================================

class NeedleHaystackDataset(Dataset):
    def __init__(self, size=5000, context_len=256, vocab_size=16):
        """
        Synthethic Needle In A Haystack.
        vocab_size: Number of 'noise' tokens.
        Special Token (vocab_size): The 'Needle Key'.
        """
        self.size = size
        self.context_len = context_len
        self.vocab_size = vocab_size
        self.needle_key_token = vocab_size  # The "Key" (e.g., "Passkey:")
        
    def __len__(self): return self.size

    def __getitem__(self, idx):
        # 1. Generate the Target (The value we want to retrieve)
        target_token = torch.randint(0, self.vocab_size, (1,)).item()
        
        # 2. Prepare the Needle Sequence: [KEY] [VALUE]
        # We assume the needle takes up 2 positions.
        needle = torch.tensor([self.needle_key_token, target_token])
        
        # 3. Generate Haystack (Noise)
        # Length = context - needle_len (2) - query_len (1)
        haystack_len = self.context_len - 4
        haystack = torch.randint(0, self.vocab_size, (haystack_len,))
        
        # 4. Insert Needle at Random Depth
        # Random insertion point between 0 and end of haystack
        insert_idx = torch.randint(0, haystack_len + 1, (1,)).item()
        
        context_part1 = haystack[:insert_idx]
        context_part2 = haystack[insert_idx:]
        
        # 5. Construct Full Sequence
        # [Haystack_1] [Needle_Key] [Target] [Haystack_2] [Needle_Key] -> Predict [Target]
        pad_token = 0
        full_seq = torch.cat([
            context_part1,
            needle,
            context_part2,
            torch.tensor([self.needle_key_token, pad_token]) # The Query
        ])
        
        # 6. Create Labels
        labels = torch.full((self.context_len,), -100)
        # The label is the LAST token (Target) which comes after the final Key
        labels[-1] = target_token
        
        return full_seq, labels
        

def calculate_accuracy(logits, labels):
    # We want the prediction coming FROM the 'Key' token.
    # Since Input is [ ... Key, Target], the Key is at index -2.
    
    # Check prediction at position -2 (The Key)
    key_logit = logits[..., -2, :] 
    pred_token = torch.argmax(key_logit, dim=-1)
    
    # Check against the actual Target (which is at -1 in the input/labels)
    target_token = labels[..., -1]
    
    return (pred_token == target_token).float().mean().item()
    
# ==========================================
# 2. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="holo", choices=["holo", "mamba"])
    parser.add_argument("--use_version", type=int, default=1) 
    parser.add_argument("--epochs", type=int, default=20) 
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--vocab_size", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_len", type=int, default=256)
    args = parser.parse_args()

    console = Console()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    os.makedirs("plots", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # --- INITIALIZE MODEL ---
    # Note: vocab_size + 1 for the regular tokens + 1 for the special Key token
    model_vocab_size = args.vocab_size + 1 
    
    if args.model == "holo":
        config = HoloConfig(d_model=args.d_model, num_hidden_layers=args.n_layers, 
                            vocab_size=model_vocab_size, resid_dropout=0.0, dropout=0.0,
                            use_version=args.use_version)
        model = HoloForCausalLM(config).to(device)
    else:
        config_kwargs = {"vocab_size": model_vocab_size, "ssm_cfg": {"dropout": 0.0}}
        if MAMBA_LIB == "official":
            config = MambaConfig(d_model=args.d_model, n_layer=args.n_layers, **config_kwargs)
        else:
            config = MambaConfig(hidden_size=args.d_model, num_hidden_layers=args.n_layers, **config_kwargs)
        model = MambaForCausalLM(config).to(device)

    # --- TRAINING LOOP ---
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
    
    # Train on Short Contexts (Random Depths are inherent in the dataset class)
    train_ds = NeedleHaystackDataset(size=5000, context_len=args.train_len, vocab_size=args.vocab_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), MofNCompleteColumn(),
        TextColumn("[bold magenta]{task.fields[stats]}"), TimeRemainingColumn(),
    ) as progress:
        
        train_task = progress.add_task(f"Training NIAH ({args.model.upper()})", total=args.epochs, stats="Init")
        
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(input_ids=x, labels=y)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
            
            progress.update(train_task, advance=1, stats=f"Loss: {total_loss/len(train_loader):.4f}")

    # --- FINAL EXTRAPOLATION BENCHMARK (NIAH SCALING) ---
    console.print(f"\n[bold cyan]Running Needle-In-Haystack Scaling Benchmark...[/bold cyan]")
    
    # Testing lengths: 256 up to 512K
    test_lengths = [2**i for i in range(8, 20)] # Starts at 256
    accuracies = []
    
    model.eval()
    with torch.no_grad():
        for l in test_lengths:
            batch_accs = []
            # Test 50 samples per length (Random depths averaged)
            for _ in range(50):
                # Ensure batch size 1 to avoid padding issues with variable lengths if we were mixing
                bench_ds = NeedleHaystackDataset(size=1, context_len=l, vocab_size=args.vocab_size)
                bx, by = next(iter(DataLoader(bench_ds, batch_size=1)))
                try:
                    logits = model(input_ids=bx.to(device)).logits
                    acc = calculate_accuracy(logits, by.to(device))
                    batch_accs.append(acc)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        batch_accs.append(0.0) 
                        torch.cuda.empty_cache()
                        break 
                    else:
                        raise e
            
            avg_acc = np.mean(batch_accs) if batch_accs else 0.0
            accuracies.append(avg_acc)
            console.print(f"Ctx Len {l:5d}: {avg_acc:.2%}")

    # --- PLOTTING ---
    plt.figure(figsize=(10, 6))
    plt.plot(test_lengths, accuracies, marker='o', linewidth=2, 
             color='#2ecc71' if args.model=='mamba' else '#3498db', label=args.model.upper())
    
    plt.xscale('log', base=2)
    plt.axvline(x=args.train_len, color='gray', linestyle='--', alpha=0.5, label=f"Train Len ({args.train_len})")
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Haystack Length (Log2)")
    plt.ylabel("Retrieval Accuracy")
    plt.title(f"{args.model.upper()} Needle In A Haystack (PassKey)")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    if args.model == "holo":
        plot_path = f"plots/{args.model}_v{args.use_version}_niah.png"
    else:
        plot_path = f"plots/{args.model}_niah.png"
        
    plt.savefig(plot_path)
    console.print(f"[bold green]Plot saved to {plot_path}[/bold green]")

    # Save Results
    data_path = f"results/{args.model}_niah_results.json"
    with open(data_path, 'w') as f:
        json.dump({"lengths": test_lengths, "accuracies": accuracies, "model": args.model}, f)
   
''' Ver. 2: Comments on the new code (Long Dang - 16/01/2026
    There are 3 Critical Errors that will cause our model to fail silently (Loss decreases, Accuracy 0%).
    1. The "Token Collision" Bug (Fatal)
    The Code: haystack = torch.randint(0, self.vocab_size...) and pad_token = 0.
    The Problem: The token 0 is used for random noise inside the haystack, but it is also used as the query delimiter ([Needle Key] [Pad]).
    Result: The model cannot tell if 0 is just garbage noise or the signal to output the answer. It gets confused and guesses.
    Fix: Reserve a special ID (e.g., vocab_size - 1) only for prompts/padding and exclude it from noise generation.
    2. The "Wrong Logit" Bug (Evaluation)
    The Code: key_logit = logits[..., -2, :]
    The Problem: The student's input sequence is [... Needle_Key, Pad].
    The model predicts the next token.
    The logit at index -2 (Needle Key) predicts the token at -1 (Pad).
    The logit at index -1 (Pad) predicts the Target.
    Result: you checked if the model predicts the Pad token correctly, not the Answer.
    Fix: Check logits[..., -1, :] (the last token).
    3. Lack of Associative Symmetry
    The Code: [Needle Key] [Target] ... [Haystack] ... [Needle Key] -> [Target]
    The Problem: This is "Hard Mode". The model has to recall Target seeing Key.
    The Fix: "Easy Mode" (Mechanism Learning) requires [Prompt] [Key] ... [Prompt] [Key]. The model learns to bind Key to Prompt.
    '''
