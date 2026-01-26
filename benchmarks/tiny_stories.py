import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.tensorboard import SummaryWriter

from transformers import AutoTokenizer
from datasets import load_dataset
import argparse
import os

# Rich for terminal beauty
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.console import Console

# Import your model classes here
from model.long import LongConfig, LongForCausalLM

console = Console()

class StreamDataset(IterableDataset):
    def __init__(self, hf_dataset, tokenizer, max_len):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __iter__(self):
        for item in self.hf_dataset:
            text = item['text']
            enc = self.tokenizer(
                text, 
                truncation=True, 
                max_length=self.max_len, 
                padding="max_length", 
                return_tensors="pt"
            )
            yield enc['input_ids'].squeeze(0)

@torch.no_grad()
def generate_sample(model, tokenizer, writer, step, max_new_tokens=80, temperature=0.8, top_k=40, repetition_penalty=1.2):
    model.eval()
    prompt = "Once upon a time,"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
    generated = input_ids.tolist()[0]
    
    past_key_values = None
    curr_in = input_ids

    for _ in range(max_new_tokens):
        outputs = model(curr_in, past_key_values=past_key_values)
        logits = outputs.logits[:, -1, :]
        
        # Apply Repetition Penalty
        for token in set(generated):
            if logits[0, token] > 0:
                logits[0, token] /= repetition_penalty
            else:
                logits[0, token] *= repetition_penalty

        logits = logits / temperature
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = -float('Inf')
        
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        generated.append(next_token.item())
        curr_in = next_token
        past_key_values = outputs.past_key_values

        if next_token.item() == tokenizer.eos_token_id:
            break
            
    text = tokenizer.decode(generated, skip_special_tokens=True)
    
    # Log to TensorBoard
    writer.add_text("Generated_Sample", text, step)
    
    console.print(f"\n[bold magenta]Step {step} Sample:[/bold magenta] {text}\n")
    model.train()

def train(args):
    # Setup TensorBoard
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # 1. Load Data & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    console.print("[green]Loading TinyStories dataset (Streaming)...[/green]")
    hf_dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    train_dataset = StreamDataset(hf_dataset, tokenizer, args.seq_len)

    # 2. Model Setup
    config = LongConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.layers,
        num_heads=args.heads,
        expansion_ratio=8/3,
        conv_kernel=4,
        hybrid_ratio=args.hybrid_ratio,
        max_position_embeddings=args.seq_len
    )
    model = LongForCausalLM(config).cuda()
    
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    console.print(f"[bold blue]Model Parameters:[/bold blue] {total_params:.2f}M")
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # LR Scheduler (Helpful for stability)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps)

    loader = DataLoader(train_dataset, batch_size=args.batch_size)
    
    model.train()
    step = 0

    # 3. Rich Progress Bar Setup
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description} [bold blue][{task.completed}/{task.total}][/bold blue]"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        train_task = progress.add_task("[cyan]Training Model...", total=args.max_steps)

        for batch in loader:
            if step >= args.max_steps: break
            
            batch = batch.cuda()
            outputs = model(batch, labels=batch)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # Logging
            writer.add_scalar("Loss/train", loss.item(), step)
            writer.add_scalar("LR", scheduler.get_last_lr()[0], step)
            
            step += 1
            progress.update(train_task, advance=1, description=f"Loss: {loss.item():.4f}")
            
            if step % args.save_every == 0:
                generate_sample(model, tokenizer, writer, step)
                # Save model logic can go here
                # torch.save(model.state_dict(), f"checkpoint_{step}.pt")

    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LongAttention Model")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--hybrid_ratio", type=int, default=0, help="0 for pure linear, 4 for every 4th layer softmax")
    parser.add_argument("--log_dir", type=str, default="runs/long_model_experiment")

    args = parser.parse_args()
    train(args)