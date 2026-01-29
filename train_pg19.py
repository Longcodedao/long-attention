import sys
import os
import math
import torch
import random
import argparse
import datasets
import transformers
import warnings
import torch.nn.functional as F
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import set_seed
from torchmetrics import MeanMetric
from rich.live import Live
from rich.panel import Panel
from rich.console import Group
from rich.table import Table
from transformers import get_cosine_schedule_with_warmup

# --- LOCAL IMPORTS ---
from model import model_loader
from dataset import data_loader
import utils
import torch.distributed as dist
# Disable standard progress bars to let Rich handle the UI
datasets.disable_progress_bar()
datasets.utils.logging.set_verbosity_error()
transformers.utils.logging.set_verbosity_error()
warnings.filterwarnings("ignore", message=".*barrier().*")

# ===============================
# 0. Monitoring Strategy: Evaluation Prompts
# ===============================
EVAL_PROMPTS = [
    "The mysterious book began with",
    "It was the best of times, it was",
    "The history of the empire is",
    "She walked into the dark room and",
    "In the early 19th century, the philosophy",
    "The captain looked out at the sea and",
    "To understand the nature of reality, one must"
]

def log_sample_generation(model, tokenizer, accelerator, global_step):
    # Manual wrapping 
    unwrapped_model = model
    while hasattr(unwrapped_model, "module"):
        unwrapped_model = unwrapped_model.module
        
    # Unwrap torch.compile wrapper (safe check)
    if hasattr(unwrapped_model, "_orig_mod"):
        unwrapped_model = unwrapped_model._orig_mod
    
    prompt = random.choice(EVAL_PROMPTS)
    generated_text = "" 

    # ONLY GPU 0 does the risky work
    if accelerator.is_main_process:
        try:
            with torch.no_grad():
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(accelerator.device)
                output_ids = unwrapped_model.generate(
                    input_ids, 
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.8,
                    pad_token_id=tokenizer.eos_token_id
                )
                generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                
                # Log to TensorBoard immediately while on main process
                tracker = accelerator.get_tracker("tensorboard")
                if tracker:
                    tracker.writer.add_text("Samples/PG19", 
                                            f"**Step {global_step}**\n\n**Prompt:** {prompt}\n\n{generated_text}", global_step)
        except Exception as e:
            generated_text = f"Generation failed: {e}"
            print(f"Gen Error: {e}")

    # Return to the main loop (GPU 1 will return "", GPU 0 will return the text)
    return prompt, generated_text
    
# ===============================
# 1. Argument Parsing
# ===============================
def parse_args():
    parser = argparse.ArgumentParser(description="LLM Training Script - PG19")
    parser.add_argument("--dataset", type=str, default="pg19", help="dataset name")
    parser.add_argument("--val_dataset", type=str, default="pg19", help="validation dataset name")
    parser.add_argument("--model_type", type=str, default="long", choices=["gpt2", "mamba", "mamba2", "long"], help="Model architecture")
    parser.add_argument("--model_size", type=str, default="small", help="Model size")
    
    # PG19 specific hyperparams
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device batch size (Reduced for longer seq_len)")
    parser.add_argument("--grad_accum_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=50000, help="Total training steps")
    parser.add_argument("--seq_len", type=int, default=2048, help="PG19 standard is 2048+")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="Warmup steps")
    parser.add_argument("--save_steps", type=int, default=2500, help="Save checkpoint every X steps")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluate every X steps")

    # System
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Save VRAM")      
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint or 'latest'")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/pg19_long", help="Save directory")
    parser.add_argument("--output_dir", type=str, default="./output/pg19_long", help="Final model directory")
    parser.add_argument("--log_dir", type=str, default="./logs/pg19", help="TensorBoard log directory")
    
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
    

# ===============================
# 3. Handle Resume Logic (Determine Step)
# ===============================
resume_step = 0
if args.resume_from_checkpoint:
    if args.resume_from_checkpoint == "latest":
        if os.path.exists(args.checkpoint_dir):
            folders = [f for f in os.listdir(args.checkpoint_dir) if "step_" in f]
            if folders:
                folders.sort(key=lambda x: int(x.split("_")[1]))
                args.resume_from_checkpoint = os.path.join(args.checkpoint_dir, folders[-1])
                resume_step = int(folders[-1].split("_")[1])
    else:
        try:
            # defined as the last part of the path (e.g. "step_5000")
            step_str = os.path.basename(os.path.normpath(args.resume_from_checkpoint))
            resume_step = int(step_str.split("_")[-1])
        except:
            resume_step = 0 

# ===============================
# 4. Load Model & Tokenizer
# ===============================
model, tokenizer = model_loader.get_model_and_tokenizer(
    model_type=args.model_type,
    model_size=args.model_size,
    seq_len=args.seq_len,
    device=accelerator.device
)

if args.gradient_checkpointing:
    model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False



# ===============================
# 5. Load Data
# ===============================
# Note: Relies on Accelerator's native skipping, no fast_skip_batches arg
train_loader = data_loader.get_dataloader(
    console, accelerator, tokenizer, 
    args.dataset, args.batch_size, args.seq_len, 
    split="train"
)
val_loader = data_loader.get_dataloader(
    console, accelerator, tokenizer, 
    args.val_dataset, args.batch_size, args.seq_len, 
    split="validation",
)

accelerator.print("Data Loaders initialized. Preparing for distributed training...")

# ===============================
# 6. Optimizer & Scheduler
# ===============================
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps
)

# Prepare everything with Accelerator
model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
    model, optimizer, train_loader, val_loader, scheduler
)

# --- CONDITIONAL COMPILATION ---
# Only compile if model_type is 'long', otherwise run standard PyTorch
if args.model_type == "long":
    if accelerator.is_main_process:
        console.print(f"[bold green][Performance] Model Type is '{args.model_type}'. Applying torch.compile()...[/bold green]")

    unwrapped_model = accelerator.unwrap_model(model)
    compiled_model = torch.compile(unwrapped_model, mode="default")
    
    # Check if DeepSpeed/DDP has wrapped the model
    if hasattr(model, "module"):
        # IMPORTANT: Compile the INNER module to avoid RecursionError with DeepSpeed Engine
        model.module = compiled_model
    else:
        # Fallback for single-GPU/non-DeepSpeed runs
        model = compiled_model
else:
    if accelerator.is_main_process:
        console.print(f"[dim][Performance] Model Type is '{args.model_type}'. Skipping torch.compile.[/dim]")
        

# CRITICAL: Ensure all processes have finished 'preparing' before moving to state loading
accelerator.wait_for_everyone()
accelerator.print("Preparation complete. Moving to training loop.")



# --- LOAD STATE (Model + Optimizer + Scheduler) ---
if args.resume_from_checkpoint:
    if args.resume_from_checkpoint != "latest" or os.path.exists(args.checkpoint_dir):
        if accelerator.is_main_process:
            console.print(f"[bold yellow]Loading state from {args.resume_from_checkpoint}...[/bold yellow]")
        accelerator.load_state(args.resume_from_checkpoint)
        if accelerator.is_main_process:
            console.print("[green]State loaded successfully (Weights, Opt, Sched).[/green]")

loss_metric = MeanMetric().to(accelerator.device)

# ===============================
# 7. Training Loop with Rich UI
# ===============================
model.train()
# --- CORRECT DATA RESUMPTION LOGIC ---
# We calculate how many "micro-batches" to skip.
# Global Step = 1 Update.
# 1 Update = (Gradient Accumulation) Batches.
batches_to_skip = resume_step * args.grad_accum_steps

if resume_step > 0:
    if accelerator.is_main_process:
        console.print(f"[bold yellow]Resuming Data: Skipping {batches_to_skip} micro-batches to match Step {resume_step}...[/bold yellow]")
    
    # This acts as a "Fast Forward" for the data loader using the official method
    # It ensures the data iterator starts exactly where the model left off
    active_dataloader = accelerator.skip_first_batches(train_loader, batches_to_skip)
else:
    active_dataloader = train_loader

# Create the iterator from the fast-forwarded loader
data_iter = iter(active_dataloader)

# UI Variables
gen_text_display = "Waiting for first evaluation..."
gen_prompt_display = "None"
val_loss_display = "N/A"
grad_norm_display = "0.0"

progress_bar = utils.create_progress_bar()

if accelerator.is_main_process:
    accelerator.init_trackers(os.path.basename(args.log_dir), config=vars(args))
    train_task_id = progress_bar.add_task("[green]PG19 Training...", total=args.max_steps)
    
    # Check Scheduler State (Sanity Check)
    current_lr = scheduler.get_last_lr()[0]
    console.print(f"[bold cyan]Resumed Scheduler LR: {current_lr:.2e}[/bold cyan]")
    
    live = Live(console=console, refresh_per_second=4) 
    live.start()

global_step = resume_step

try:
    while global_step < args.max_steps:
        # --- Data Fetching ---
        try:
            batch = next(data_iter)
        except StopIteration:
            # If we run out of data, reset to the beginning (epoch done)
            if accelerator.is_main_process:
                # console.print("[dim]Epoch Complete. Restarting Data Loader...[/dim]") # Optional log
                pass
            data_iter = iter(train_loader)
            batch = next(data_iter)

        # --- Forward / Backward ---
        with accelerator.accumulate(model):
            outputs = model(input_ids=batch["input_ids"], labels=batch["input_ids"])
            loss = outputs.loss
            accelerator.backward(loss)
            
            # Clip Gradients & Log Norm (Crucial for stability)
            if accelerator.sync_gradients:
                # clip_grad_norm_ returns the norm before clipping
                total_norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)

                if isinstance(total_norm, torch.Tensor):
                    grad_norm_display = f"{total_norm.item():.2f}"
                else:
                    grad_norm_display = f"{total_norm:.2f}"

                if torch.isnan(total_norm) or torch.isinf(total_norm):
                    console.print(f"!!! Skipping Step {global_step} due to NaN/Inf Grad Norm !!!")
                    optimizer.zero_grad()
                else:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
            
            gathered_loss = accelerator.gather(loss.detach())
            loss_metric.update(gathered_loss.mean())

        # --- Step Updates ---
        if accelerator.sync_gradients:
            global_step += 1
            current_loss = loss_metric.compute().item()
            loss_metric.reset()
            current_ppl = math.exp(current_loss) if current_loss < 20 else 9999.0
            current_lr = scheduler.get_last_lr()[0]

            if accelerator.is_main_process:
                progress_bar.update(train_task_id, completed=global_step)
                
                # --- UI Construction ---
                metrics_table = Table(show_header=False, box=None)
                metrics_table.add_row("Loss", f"[bold red]{current_loss:.4f}[/bold red]")
                metrics_table.add_row("PPL", f"[yellow]{current_ppl:.2f}[/yellow]")
                metrics_table.add_row("LR", f"{current_lr:.2e}")
                metrics_table.add_row("Grad Norm", f"{grad_norm_display}")
                
                status_panel = Panel(
                    metrics_table, 
                    title=f"Step {global_step}/{args.max_steps}", 
                    border_style="green",
                    width=30
                )
                
                gen_panel = Panel(
                    f"[dim]Prompt:[/dim] [cyan]{gen_prompt_display}[/cyan]\n\n{gen_text_display}",
                    title=f"Last Generation (Val Loss: {val_loss_display})",
                    border_style="blue",
                    height=12
                )
                
                live.update(Group(
                    Panel(progress_bar, border_style="none"),
                    transformers.utils.logging.get_logger("transformers").level == 40 and "" or "", # spacer
                    status_panel,
                    gen_panel
                ))

            # Log to TensorBoard
            accelerator.log({
                "train_loss": current_loss, 
                "train_ppl": current_ppl, 
                "lr": current_lr,
                "grad_norm": float(grad_norm_display) 
            }, step=global_step)
            
            # --- EVALUATION ---
            if global_step % args.eval_steps == 0:
                # 1. Calculate Val Loss
                accelerator.wait_for_everyone()
    
                model.eval()
                val_losses = []
                # Limit eval to 30 batches for speed
                val_iter = iter(val_loader)
                for _ in range(30):
                    try:
                        vbatch = next(val_iter)
                    except StopIteration:
                        break
                    with torch.no_grad():
                        v_out = model(input_ids=vbatch["input_ids"], labels=vbatch["input_ids"])
                        val_losses.append(accelerator.gather(v_out.loss).mean().item())
                
                avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else float("inf")
                val_ppl = math.exp(avg_val_loss) if avg_val_loss < 20 else 9999.0
                val_loss_display = f"{avg_val_loss:.4f}"
                accelerator.log({"val_loss": avg_val_loss, "val_ppl": val_ppl}, step=global_step)

                # 2. Generate Sample (Main Process Only)
                prompt_used, sample_text = log_sample_generation(model, tokenizer, accelerator, global_step)

                if accelerator.is_main_process:
                    # Update UI variables
                    gen_prompt_display = prompt_used
                    gen_text_display = f"[white]{sample_text[:300]}...[/white]"

                # Sync 2: Crucial! GPU 1 waits for GPU 0's generation/logging to finish
                accelerator.wait_for_everyone()
                model.train()

            # --- SAVE ---
            if global_step % args.save_steps == 0:
                # Sync 3: Ensure all gradients/evals are finished before touching disk
                accelerator.wait_for_everyone()
                accelerator.save_state(f"{args.checkpoint_dir}/step_{global_step}")

                if accelerator.is_main_process:
                    console.print(f"[bold cyan]Checkpoint saved at step {global_step}[/bold cyan]")

    # --- FINALIZE ---
    if accelerator.is_main_process:
        live.stop()
        console.print("[bold green]Training Complete! Saving final model...[/bold green]")

    accelerator.wait_for_everyone()

    # === FIX: Manual Unwrapping to bypass Accelerate/torch.compile crash ===
    # This robust unwrap logic ensures we don't save the compiled container
    unwrapped_model = model
    
    # 1. Unwrap DeepSpeed/DDP (recursively get .module)
    while hasattr(unwrapped_model, "module"):
        unwrapped_model = unwrapped_model.module
        
    # 2. Unwrap torch.compile (look for _orig_mod)
    # Accelerate crashes here because it assumes _orig_mod is always in __dict__,
    # but a simple getattr/hasattr check is much safer.
    if hasattr(unwrapped_model, "_orig_mod"):
        unwrapped_model = unwrapped_model._orig_mod

    # 3. Save
    unwrapped_model.save_pretrained(
        args.output_dir, 
        save_function=accelerator.save, 
        safe_serialization=False
    )
    if tokenizer:
        tokenizer.save_pretrained(args.output_dir)
        
    if accelerator.is_main_process:
        console.print(f"[bold green]Model saved to {args.output_dir}[/bold green]")

except KeyboardInterrupt:
    if accelerator.is_main_process:
        live.stop()
        console.print("[bold red]Training Interrupted![/bold red]")

finally:
    accelerator.end_training()
    if dist.is_initialized():
        dist.destroy_process_group()