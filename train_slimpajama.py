import sys
import os
import math
import time
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from datasets import load_dataset, IterableDataset
from model.configuration_holo import HoloConfig
from model.modeling_holo import HoloForCausalLM

# --- NEW IMPORTS ---
from accelerate import Accelerator 
from accelerate.utils import set_seed 
from torchmetrics import MeanMetric 
from torchmetrics.text import Perplexity
from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

# ===============================
# 1, Setup Accelerator & Device
# ===============================
# Accelerator handles device placement, gradient accumulation, and mixed precision automatically
accelerator = Accelerator(
    gradient_accumulation_steps = 8,   # Replaces your manual GRAD_ACCUM_STEPS
    mixed_precision = "bf16",
    log_with="tensorboard",
    project_dir = "."
)
set_seed(42)

# Rich Console for pretty printing (only on main process)
console = Console() if accelerator.is_main_process else None

# Configuration
BATCH_SIZE = 2             # Per-device batch size
LEARNING_RATE = 3e-4
MAX_STEPS = 1000
SEQ_LEN = 2048
WARMUP_STEPS = 100
SAVE_STEPS = 200

if accelerator.is_main_process:
    console.print(f"[bold green]Starting Training on {accelerator.num_processes} GPU(s)[/bold green]")


# ==========================================
# 2. Data Loading (DDP Compatible)
# ==========================================
# Data Loading
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Load Dataset (Streaming to avoid disk usage)
dataset = load_dataset("DKYoon/SlimPajama-6B", split="train", streaming=True)

if isinstance(dataset, IterableDataset) and accelerator.num_processes > 1:
    # This shards the stream so Rank 0 gets chunk 0, Rank 1 gets chunk 1, etc
    dataset = dataset.shard(
        num_shards = accelerator.num_processes, 
        index = accelerator.process_index
    )
    

def tokenization_fn(examples):
    return tokenizer(
        examples["text"], 
        truncation=True, 
        max_length=SEQ_LEN, 
        padding="max_length",
    )


# Create Iterator
def get_data_loader(dataset, batch_size):
    # We use map with batched=False for streaming usually, or custom collate
    # Here we stick to your logic but cleaned up for DataLoader
    mapped_ds = dataset.map(tokenization_fn, remove_columns = ["text", "meta"])
    mapped_ds = mapped_ds.with_format("torch")
    
    # CHANGE: Add num_workers and pin_memory
    return DataLoader(
        mapped_ds, 
        batch_size=batch_size, 
        num_workers = 4,
        prefetch_factor = 2,
        pin_memory=True,    # Faster transfer to CUDA
    )

train_loader = get_data_loader(dataset, BATCH_SIZE)


# ==========================================
# 3. Model Initialization
# ==========================================
config = HoloConfig.from_preset("small", use_version=2)
model = HoloForCausalLM(config)

# Optimizer 
optimizer = torch.optim.AdamW(model.parameters(), 
                              lr = LEARNING_RATE,
                              betas=(0.9, 0.95))
# Scheuler 
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps = WARMUP_STEPS, num_training_steps = MAX_STEPS
)


# ==========================================
# 4. Prepare with Accelerator
# ==========================================
# This wraps model (DDP), optimizer, and loader automatically
model, optimizer, train_loader, scheduler = accelerator.prepare(
    model, optimizer, train_loader, scheduler
)

# Initialize TorchMetrics
# We move metrics to the correct device
loss_metric = MeanMetric().to(accelerator.device)
perplexity_metric = Perplexity(ignore_index=tokenizer.pad_token_id).to(accelerator.device)


# ==========================================
# 5. Rich Dashboard Utilities
# ==========================================
def create_training_table(step, loss, ppl, lr):
    table = Table(show_header=True, header_style="bold magenta", expand=True)
    table.add_column("Global Step", style="dim", width=12)
    table.add_column("Loss", justify="right")
    table.add_column("Perplexity", justify="right")
    table.add_column("LR", justify="right")
    
    loss_str = f"{loss:.4f}" if loss is not None else "..."
    ppl_str = f"{ppl:.2f}" if ppl is not None else "..."
    
    table.add_row(f"{step}", loss_str, ppl_str, f"{lr:.2e}")
    return Panel(table, title="Holo Training Metrics", border_style="blue")


# ==========================================
# 6. Training Loop
# ==========================================
model.train()
global_step = 0
data_iter = iter(train_loader)

# Define Progress Bar Structure
progress_bar = Progress(
    SpinnerColumn(),
    TextColumn("[bold blue]{task.description}"),
    BarColumn(bar_width=None), # Flexible width
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    TimeRemainingColumn(),
    expand=True
)

if accelerator.is_main_process:
    # Intialize TensorBoard before the loop starts
    accelerator.init_trackers("holo_logs", config={"lr": LEARNING_RATE, "batch": BATCH_SIZE})
    # Add the main training task 
    train_task_id = progress_bar.add_task("[green]Training...", total = MAX_STEPS)

    # Create a Live display that renders a Group (Table + Progress Bar)
    # We initialize it with empty data first
    live = Live(console=console, refresh_per_second=4)
    live.start()

try:
    while global_step < MAX_STEPS:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        # Accelerator handles gradient accumulation automatically via this context
        with accelerator.accumulate(model):
            # print(batch["input_ids"].shape)
            outputs = model(input_ids=batch["input_ids"], labels=batch["input_ids"])
            loss = outputs.loss
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # --- METRICS UPDATE ---
            # Gather loss across GPUs for accurate logging
            gathered_loss = accelerator.gather(loss.detach())
            loss_metric.update(gathered_loss.mean())
            
            # For PPL, we technically need logits, but exp(loss) is standard approx
            # We update metric with the loss value for smoothing
            # (Strictly speaking PPL = exp(cross_entropy))
            
        # Only update global step if gradient sync happened (accumulation finished)
        if accelerator.sync_gradients:
            global_step += 1
            
            # Compute current metrics
            current_loss = loss_metric.compute().item()
            try:
                current_ppl = math.exp(current_loss)
            except OverflowError:
                current_ppl = float("inf")
            
            current_lr = scheduler.get_last_lr()[0]
            
            # Reset metric for the next window if you want instantaneous vs running avg
            # loss_metric.reset() 

            # Accelerator handles the "is_main_process" check for logging internally,
            # but putting it here ensures we log exactly when we update the UI.
            accelerator.log({
                "train_loss": current_loss,
                "perplexity": current_ppl,
                "learning_rate": current_lr
            }, step=global_step)
            
            # Update Terminal Output (Rank 0 only)
            if accelerator.is_main_process:
                # 1. Update the Progress Bar Object 
                progress_bar.update(train_task_id, completed = global_step)

                # 2. Create the Metrics Table
                metrics_panel = create_training_table(global_step, current_loss, current_ppl, current_lr)

                # 3. Group them together (Table on top, Bar on bottom)
                display_group = Group(
                    metrics_panel,
                    progress_bar
                )
                
                # 4. Push to Live
                live.update(display_group)
            
            # Save Checkpoint
            if global_step % SAVE_STEPS == 0:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                save_path = f"./holo_checkpoints/step_{global_step}"
                if accelerator.is_main_process:
                    unwrapped_model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
                    # We print above the Live display to avoid messing up the UI
                    live.console.print(f"[dim]Saved checkpoint to {save_path}[/dim]")

finally:
    if accelerator.is_main_process:
        live.stop()

    # Close the TensorBoard connection safely
    accelerator.end_training()

    if accelerator.is_main_process:
        console.print("[bold green]Training Finished![/bold green]")