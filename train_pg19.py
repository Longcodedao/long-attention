import sys
import os
import math
import torch
from transformers import get_cosine_schedule_with_warmup
import argparse
from accelerate import Accelerator
from accelerate.utils import set_seed
from torchmetrics import MeanMetric
from rich.live import Live
from rich.panel import Panel
from rich.console import Group
from rich.table import Table
from rich.box import DOUBLE_EDGE

# --- LOCAL IMPORTS ---
# We assume these still exist in your folder structure
from model import model_loader
from data import data_loader
import utils
import datasets
import transformers
import warnings

# Disable Hugging Face progress bars and set logging to error only
datasets.disable_progress_bar()
datasets.utils.logging.set_verbosity_error()
transformers.utils.logging.set_verbosity_error()

# Silence the specific Torch distributed warning
warnings.filterwarnings("ignore", message=".*barrier().*")

# ===============================
# 1. Argument Parsing (Tuned for LCFT)
# ===============================
def parse_args():
    parser = argparse.ArgumentParser(description="PG19 Long-Context Fine-Tuning Script")

    # Dataset & Model Paths
    parser.add_argument("--pretrained_path", type=str, required=True, 
                        help="Path to the generic 512-context checkpoint (e.g., './output/slimpajama_6b')")
    parser.add_argument("--dataset_name", type=str, default="pg19", help="HuggingFace dataset name")

    # Model Configuration 
    parser.add_argument("--model_type", type=str, default="mamba", choices=["holo", "gpt2", "mamba", "mamba2"])
    parser.add_argument("--model_size", type=str, default="small", help="Must match the pretrained model size")
    
    # Hyperparameters (Optimized for Long Context)
    parser.add_argument("--seq_len", type=int, default=16384, help="Target context length (16k)")
    parser.add_argument("--batch_size", type=int, default=1, help="Keep this at 1 for 16k context to avoid OOM")
    parser.add_argument("--grad_accum_steps", type=int, default=8, help="Higher accum to simulate larger batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Lower learning rate for fine-tuning")
    parser.add_argument("--max_steps", type=int, default=1000, help="1000 steps at 16k context is ~130M tokens (plenty for adaptation)")
    parser.add_argument("--warmup_steps", type=int, default=50, help="Short warmup to stabilize gradients")
    
    # Validation & Saving
    parser.add_argument("--save_steps", type=int, default=200, help="Save checkpoints frequently")
    parser.add_argument("--eval_steps", type=int, default=50, help="Eval is expensive at 16k, run less often")
    
    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/pg19_finetune", help="Directory to save checkpoints")
    parser.add_argument("--output_dir", type=str, default="./output/pg19_finetune", help="Directory to save the final model")
    parser.add_argument("--log_dir", type=str, default="./pg19_logs", help="Directory for TensorBoard logs")
    
    return parser.parse_args()

args = parse_args()

# ===============================
# 2. Setup Accelerator
# ===============================
accelerator = Accelerator(
    gradient_accumulation_steps=args.grad_accum_steps,
    log_with="tensorboard",
    project_dir="."
)
set_seed(42)
console = utils.get_console(accelerator)

if accelerator.is_main_process:
    utils.print_config_table(console, accelerator, args)
    console.print(f"[bold green]Starting Long-Context Fine-Tuning (Target: {args.seq_len} tokens)...[/bold green]")

# ===============================
# 3. Load Model & Tokenizer
# ===============================
# Initialize the architecture (Must match the pretrained config)
model, tokenizer = model_loader.get_model_and_tokenizer(
    model_type=args.model_type,
    model_size=args.model_size,
    seq_len=args.seq_len, # Initialize with NEW sequence length (16k)
    device=accelerator.device
)

# --- CRITICAL: LOAD PRETRAINED WEIGHTS ---
# We load the weights from the 512-context run, but apply them to the 16k-context architecture.
# Mamba adapts naturally; Transformers might need position embedding interpolation (RoPE handles this auto usually).
if accelerator.is_main_process:
    console.print(f"[bold yellow]Loading pretrained weights from: {args.pretrained_path}[/bold yellow]")

try:
    # Try loading as a folder (HF format)
    if os.path.isdir(args.pretrained_path):
        model = model.from_pretrained(args.pretrained_path).to(accelerator.device)
    else:
        # Try loading as a direct .bin/.pt file
        state_dict = torch.load(args.pretrained_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False) # strict=False allows resizing of pos embeddings if needed
except Exception as e:
    console.print(f"[bold red]Error loading weights: {e}[/bold red]")
    sys.exit(1)

# FORCE Gradient Checkpointing (Mandatory for 16k Context)
model.gradient_checkpointing_enable()
if hasattr(model.config, "use_cache"):
    model.config.use_cache = False

# ===============================
# 4. Custom PG19 Streaming Loader
# ===============================


# Create Loaders
train_loader = data_loader.create_pg19_dataloader(tokenizer, args.seq_len, args.batch_size, split="train")
val_loader = data_loader.create_pg19_dataloader(tokenizer, args.seq_len, args.batch_size, split="validation")

# ===============================
# 5. Optimization Setup
# ===============================
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps
)

# Prepare with Accelerator
model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
    model, optimizer, train_loader, val_loader, scheduler
)

loss_metric = MeanMetric().to(accelerator.device)

# ===============================
# 6. Training Loop (Standard)
# ===============================
global_step = 0

# UI State
val_message = "[dim]Waiting for first evaluation...[/dim]"
val_border_style = "dim" 
progress_bar = utils.create_progress_bar()

if accelerator.is_main_process:
    accelerator.init_trackers(args.log_dir, config=vars(args))    
    train_task_id = progress_bar.add_task("[green]Fine-Tuning (PG19)...", total=args.max_steps)
    live = Live(console=console, refresh_per_second=1, redirect_stdout=True, redirect_stderr=True)
    live.start()
    
    if not os.path.exists(args.checkpoint_dir): os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)

# Helper for Evaluation
def evaluate(model, loader, max_batches=20):
    model.eval()
    losses = []
    iterator = iter(loader)
    with torch.no_grad():
        for _ in range(max_batches):
            try:
                batch = next(iterator)
                outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
                losses.append(accelerator.gather(outputs.loss).mean().item())
            except StopIteration:
                break
    model.train()
    return sum(losses) / len(losses) if losses else float("inf")

model.train()
data_iter = iter(train_loader)

try:
    while global_step < args.max_steps:
        # --- A. Step ---
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        with accelerator.accumulate(model):
            outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
            loss = outputs.loss
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            gathered_loss = accelerator.gather(loss.detach())
            loss_metric.update(gathered_loss.mean())

        # --- B. Logging & UI ---
        if accelerator.sync_gradients:
            global_step += 1
            current_loss = loss_metric.compute().item()
            loss_metric.reset() # Reset metric after logging
            current_ppl = math.exp(current_loss) if current_loss < 20 else float("inf")
            current_lr = scheduler.get_last_lr()[0]

            if accelerator.is_main_process:
                progress_bar.update(train_task_id, completed=global_step)
                
                # Build UI
                metrics_panel = utils.create_metrics_table(global_step, args.max_steps, current_loss, current_ppl, current_lr)
                val_panel = Panel(val_message, title="Validation Status", border_style=val_border_style, width=60)
                live.update(Group(metrics_panel, val_panel, progress_bar))
                
            accelerator.log({"train_loss": current_loss, "train_ppl": current_ppl, "lr": current_lr}, step=global_step)

            # --- C. Evaluation ---
            if global_step % args.eval_steps == 0:
                if accelerator.is_main_process:
                    val_message = "[bold yellow]Running Evaluation on Long Sequences...[/bold yellow]"
                    val_border_style = "yellow"
                    live.update(Group(metrics_panel, Panel(val_message, title="Validation", border_style="yellow"), progress_bar))
                
                val_loss = evaluate(model, val_loader)
                val_ppl = math.exp(val_loss) if val_loss < 20 else float("inf")
                
                accelerator.log({"val_loss": val_loss, "val_ppl": val_ppl}, step=global_step)
                
                if accelerator.is_main_process:
                    val_message = f"[bold cyan]PG19 Eval (Step {global_step}):[/bold cyan]\nLoss: {val_loss:.4f} | PPL: {val_ppl:.2f}"
                    val_border_style = "cyan"

            # --- D. Checkpointing ---
            if global_step % args.save_steps == 0:
                accelerator.wait_for_everyone()
                accelerator.save_state(f"{args.checkpoint_dir}/step_{global_step}")

    # --- END OF TRAINING ---
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        live.stop()
        console.print("[bold blue]Saving final adapted model...[/bold blue]")
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save)
        tokenizer.save_pretrained(args.output_dir)
        console.print(f"[bold green]✅ Long-Context Model saved to: {args.output_dir}[/bold green]")

finally:
    accelerator.end_training()