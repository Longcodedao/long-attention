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
# Ensure these are in your python path or adjust imports
try:
    from model.holo import HoloConfig, HoloForCausalLM
except ImportError:
    print("Warning: Holo model not found. Ensure 'model.holo' is available.")

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
# 1. DATASET & UTILITIES
# ==========================================

class InductionDataset(Dataset):
    def __init__(self, size=5500, seq_len=256, vocab_size=16, prefix_length=20):
        self.size, self.seq_len, self.vocab_size = size, seq_len, vocab_size
        self.prefix_length = prefix_length
        self.special_token = vocab_size 

    def __len__(self): return self.size

    def __getitem__(self, idx):
        # Same logic as before: Prefix... [Special] [Mem] ... Gap ... [Special] -> [Mem]
        all_tokens = torch.arange(self.vocab_size)
        memory_token = torch.randint(0, self.vocab_size, (1,)).item()
        
        prefix_len = torch.randint(1, self.prefix_length + 1, (1,)).item()
        prefix = torch.randint(0, self.vocab_size, (prefix_len,))
        
        max_suffix = max(1, self.seq_len // 10)
        suffix_len = torch.randint(1, max_suffix + 1, (1,)).item()
        suffix = torch.randint(0, self.vocab_size, (suffix_len,))
        
        fixed_parts_len = prefix_len + 2 + 1 + suffix_len
        gap_len = self.seq_len - fixed_parts_len
        
        mask = all_tokens != memory_token
        other_tokens = all_tokens[mask]
        gap = other_tokens[torch.randint(0, len(other_tokens), (gap_len,))]
        
        full_seq = torch.cat([
            prefix, 
            torch.tensor([self.special_token, memory_token]), 
            gap, 
            torch.tensor([self.special_token]), 
            suffix
        ])
        
        labels = torch.full((self.seq_len,), -100)
        trigger_pos = prefix_len + 2 + gap_len 
        if trigger_pos + 1 < self.seq_len:
            labels[trigger_pos + 1] = memory_token
            
        return full_seq, labels

def calculate_accuracy(logits, labels):
    preds = torch.argmax(logits, dim=-1)
    shift_preds = preds[..., :-1]
    shift_labels = labels[..., 1:]
    mask = shift_labels != -100
    if mask.sum() == 0: return 0.0
    return (shift_preds[mask] == shift_labels[mask]).float().mean().item()

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
    
    # Ensure plots directory exists
    os.makedirs("plots", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # --- INITIALIZE MODEL ---
    if args.model == "holo":
        config = HoloConfig(d_model=args.d_model, num_hidden_layers=args.n_layers, 
                            vocab_size=args.vocab_size + 1, resid_dropout=0.0, dropout=0.0,
                           use_version = args.use_version) # Low dropout for synthetic
        model = HoloForCausalLM(config).to(device)
    else:
        config_kwargs = {"vocab_size": args.vocab_size + 1, "ssm_cfg": {"dropout": 0.0}}
        if MAMBA_LIB == "official":
            config = MambaConfig(d_model=args.d_model, n_layer=args.n_layers, **config_kwargs)
        else:
            config = MambaConfig(hidden_size=args.d_model, num_hidden_layers=args.n_layers, **config_kwargs)
        model = MambaForCausalLM(config).to(device)

    # --- TRAINING LOOP ---
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
    # Train on Short Sequences (e.g., 256)
    train_ds = InductionDataset(size=5000, seq_len=args.train_len, vocab_size=args.vocab_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), MofNCompleteColumn(),
        TextColumn("[bold magenta]{task.fields[stats]}"), TimeRemainingColumn(),
    ) as progress:
        
        train_task = progress.add_task(f"Training {args.model.upper()}", total=args.epochs, stats="Init")
        
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

    # --- FINAL EXTRAPOLATION BENCHMARK ---
    console.print(f"\n[bold cyan]Running Final Extrapolation Benchmark for {args.model.upper()}...[/bold cyan]")
    
    # Powers of 2 from 64 up to 16384 (or higher if GPU permits)
    test_lengths = [2**i for i in range(6, 21)] # 64, 128 ... 16384
    accuracies = []
    
    model.eval()
    with torch.no_grad():
        for l in test_lengths:
            batch_accs = []
            # Test 20 samples per length
            for _ in range(100):
                bench_ds = InductionDataset(size=1, seq_len=l, vocab_size=args.vocab_size)
                bx, by = next(iter(DataLoader(bench_ds, batch_size=1)))
                try:
                    logits = model(input_ids=bx.to(device)).logits
                    acc = calculate_accuracy(logits, by.to(device))
                    batch_accs.append(acc)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        batch_accs.append(0.0) # Fail gracefully on OOM
                        torch.cuda.empty_cache()
                        break # Stop trying this length
                    else:
                        raise e
            
            avg_acc = np.mean(batch_accs) if batch_accs else 0.0
            accuracies.append(avg_acc)
            console.print(f"Len {l}: {avg_acc:.2%}")

    # --- PLOTTING (INDIVIDUAL) ---
    plt.figure(figsize=(10, 6))
    plt.plot(test_lengths, accuracies, marker='o', linewidth=2, 
             color='#2ecc71' if args.model=='mamba' else '#e74c3c', label=args.model.upper())
    
    plt.xscale('log', base=2)
    plt.axvline(x=args.train_len, color='gray', linestyle='--', alpha=0.5, label=f"Train Len ({args.train_len})")
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Sequence Length (Log2)")
    plt.ylabel("Accuracy")
    plt.title(f"{args.model.upper()} Induction Head Extrapolation")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)

    
    if args.model == "holo":
        plot_path = f"plots/{args.model}_version{ model.config.use_version}_extrapolation.png"
    else:
        plot_path = f"plots/{args.model}_extrapolation.png"
        
    plt.savefig(plot_path)
    console.print(f"[bold green]Plot saved to {plot_path}[/bold green]")

    # Save data for comparison script
    os.makedirs("benchmarks/results", exist_ok = True)
    
    data_path = f"benchmarks/results/{args.model}_results.json"
    with open(data_path, 'w') as f:
        json.dump({"lengths": test_lengths, "accuracies": accuracies, "model": args.model}, f)

''' Comments by Long Dang 16/01/2026
The code provided is a decent start for a general model test, but it has several critical flaws that will likely lead to the same "0% Accuracy" or "Overfitting" failures.
Critical Analysis
1/ Architecture Mismatch (The Fatal Flaw):
The Code: It imports from model.holo import HoloForCausalLM.
The Problem: This uses to an older version of our model. The "Memory Wall" is only broken by the specific "Holy Grail" architecture we just derived (Conv1d + Static Head + Shared QK). If you run this script with a standard Holo model, it will fail the long-context test.
The Fix: You must inject the updated LongAttention class (the one with the conv1d and freqs[0]=0 logic) directly into this script or update your library.
2/ Fixed-Length Training (The "Cliff"):
The Code: train_ds = InductionDataset(..., seq_len=args.train_len) (defaults to 256).
The Problem: As we proved, models trained on fixed lengths overfit to positional embeddings. A model trained on 256 tokens will fail completely at 257 tokens because it has never seen those position indices.
The Fix: You need Curriculum Training. The training loop must feed batches of variable lengths (e.g., random between 32 and 256) to force the model to learn the mechanism (Induction) rather than the location.
3/ Vocabulary Size vs. Sequence Length:
The Code: vocab_size=16, test_lengths up to 16,384.
The Problem: With a vocab of 16 and a sequence of 16,000, the "Needle" (Memory Token) will statistically appear ~1,000 times in the noise gap purely by chance. This makes the task ambiguous ("Which 'A' should I recall?").
The Fix: Increase vocab_size significantly (e.g., to 100 or 1000) so the Needle is unique or rare, ensuring a clean signal.
'''
