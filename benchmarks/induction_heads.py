import torch 
import torch.nn as nn
from model.holo import HoloConfig, HoloForCausalLM
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.optim as optim
from itertools import cycle
import argparse
import sys
import math

# --- UI IMPORTS ---
from rich.console import Console
from rich.progress import (
    Progress, 
    SpinnerColumn, 
    TextColumn, 
    BarColumn, 
    TimeRemainingColumn, 
)

# --- LIBRARY HANDLING ---
try:
    from mamba_ssm import MambaConfig, MambaForCausalLM
    MAMBA_LIB = "official"
except ImportError:
    try:
        from transformers import MambaConfig, MambaForCausalLM
        MAMBA_LIB = "transformers"
    except ImportError:
        MAMBA_LIB = None

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="holo", choices=["holo", "mamba"])
    parser.add_argument("--d_model", type=int, default=128) 
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--vocab_size", type=int, default=16) 
    parser.add_argument("--steps", type=int, default=3000) 
    return parser.parse_args()

def calculate_accuracy(logits, labels):
    """
    Standard Causal LM accuracy calculation for the Induction task.
    Matches the logit at index t with the label at index t+1.
    """
    # [batch, seq_len, vocab] -> [batch, seq_len]
    preds = torch.argmax(logits, dim=-1)

    # Shift: Predict the next token
    shift_preds = preds[..., :-1].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Mask only the 'value' token (-100 is ignored)
    mask = shift_labels != -100
    
    if mask.sum() == 0: 
        return 0.0
        
    correct = (shift_preds == shift_labels) & mask
    return (correct.sum().float() / mask.sum().float()).item()

class InductionDataset(Dataset):
    """
    Exact Olsson et al. (2022) induction heads dataset.
    """

    def __init__(
        self,
        size: int = 10_000,
        seq_len: int = 256,
        vocab_size: int = 16,
        prefix_length: int = 10
    ):
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.prefix_length = prefix_length
        
        assert prefix_length < seq_len - 2, "Need bigger prefix length"

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # 1. Define the specific tokens
        # Note: special_token is usually vocab_size (if embedding size allows)
        all_tokens = torch.arange(self.vocab_size)
        memory_token = torch.randint(0, self.vocab_size, (1, ))
        special_token = self.vocab_size
        token_important = torch.tensor([special_token, memory_token.item()])

        # 2. Identify all tokens in vocab that ARE NOT the memory_token
        # This creates a "safe" pool for the gap noise
        mask = all_tokens != memory_token
        other_tokens = all_tokens[mask]

        # 3. Calculate gap length
        # Format: prefix + [special, memory] + gap + [special] -> label: memory
        gap_length = self.seq_len - self.prefix_length - 2 - 1

        # 4. Randomly pick tokens from the "other_tokens" pool
        # We pick random indices from 0 to len(other_tokens)
        random_indices = torch.randint(0, len(other_tokens), (gap_length, ))
        gap = other_tokens[random_indices]

        # 5. Assemble the sequence
        prefix_toks = torch.randint(0, self.vocab_size, (self.prefix_length, ))
        # Concat: [prefix] + [special, memory] + [gap] + [special]
        full_seq = torch.cat([
            prefix_toks, 
            token_important,
            gap, 
            torch.tensor([special_token])
        ])
        
        labels = torch.full((self.seq_len, ), - 100)
        labels[-1] = memory_token.item()
        
        return full_seq, labels

# --- MAIN ---
if __name__ == "__main__":
    args = get_args()
    console = Console()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    TRAIN_SEQ_LEN = 256

    # --- MODEL INIT ---
    if args.model == "holo":
        config = HoloConfig(
            num_heads=4, 
            d_model=args.d_model, 
            num_hidden_layers=args.n_layers,
            vocab_size=args.vocab_size, 
            holo_expansion_ratio=4
        )
        model = HoloForCausalLM(config).to(device)
    
    elif args.model == "mamba":
        if MAMBA_LIB is None:
            console.print("[bold red]Error:[/bold red] Mamba libraries not found.")
            sys.exit(1)
        
        # Initialize Mamba based on available library version
        config_kwargs = {"vocab_size": args.vocab_size}
        if MAMBA_LIB == "official":
            config = MambaConfig(d_model=args.d_model, n_layer=args.n_layers, d_state=16, expand=2, **config_kwargs)
        else:
            config = MambaConfig(hidden_size=args.d_model, num_hidden_layers=args.n_layers, state_size=16, expand=2, **config_kwargs)
        model = MambaForCausalLM(config).to(device)

    # --- OPTIMIZER ---
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0, betas=(0.9, 0.95))
    
    # Warmup + Cosine Decay
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, total_steps=args.steps, 
        pct_start=0.1, anneal_strategy='cos',
        div_factor=10, final_div_factor=100
    )

    ds = InductionDataset(size=10000, seq_len=TRAIN_SEQ_LEN, vocab_size=args.vocab_size)
    ds_loader = DataLoader(ds, batch_size=32, num_workers=0, pin_memory=True)
    iterator = cycle(ds_loader)

    model.train()

    # --- PHASE 1: TRAINING ---
    console.print(f"[bold]Training {args.model.upper()}...[/bold]")

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        TextColumn("[cyan]{task.completed}/{task.total}"),
        BarColumn(),
        TextColumn("[bold green]Induction Acc: {task.fields[acc]:.2%}"),
        TimeRemainingColumn(),
    )

    with progress:
        task_id = progress.add_task("Induction Task", total=args.steps, acc=0.0)

        for step in range(1, args.steps + 1):
            input_ids, labels = next(iterator)
            input_ids, labels = input_ids.to(device), labels.to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            progress.update(task_id, advance=1)

            if step % 50 == 0:
                acc = calculate_accuracy(outputs.logits, labels)
                progress.update(task_id, acc=acc)
                
                if acc > 0.995 and step > 500:
                    console.print(f"\n[bold green]Solved at step {step}![/bold green]")
                    break

    # --- PHASE 2: EXTRAPOLATION ---
    console.print(f"\n[bold cyan]Extrapolation Benchmark ({args.model.upper()})[/bold cyan]")
    test_lengths = [2**i for i in range(6, 21)] 
    TEST_BATCH_SIZE = 1
    
    model.eval()
    with torch.no_grad():
        for length in test_lengths:
            test_ds = InductionDataset(size=TEST_BATCH_SIZE, seq_len=length, vocab_size=args.vocab_size)
            test_loader = DataLoader(test_ds, batch_size=TEST_BATCH_SIZE)
            
            inputs, labels = next(iter(test_loader))
            inputs, labels = inputs.to(device), labels.to(device)
            try:
                outputs = model(input_ids=inputs)
                acc = calculate_accuracy(outputs.logits, labels)
                
                color = "green" if acc > 0.98 else "yellow" if acc > 0.5 else "red"
                console.print(f"Len {length:8d}: [{color}]{acc*100:6.2f}% Acc[/{color}]")
            
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    console.print(f"Len {length:8d}: [bold red]OOM[/bold red]")
                    torch.cuda.empty_cache()
                    break
                raise e