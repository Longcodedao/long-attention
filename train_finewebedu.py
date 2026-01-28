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


datasets.disable_progress_bar()
datasets.utils.logging.set_verbosity_error()
transformers.utils.logging.set_verbosity_error()
warnings.filterwarnings("ignore", message=".*barrier().*")

# ===============================
# 0. Monitoring Strategy: Evaluation Prompts
# ===============================
EVAL_PROMPTS = [
    "The three main states of matter are solid, liquid, and",
    "In a eukaryotic cell, the DNA is primarily located within the",
    "Photosynthesis is the process by which plants use sunlight to",
    "If a triangle has a 90-degree angle, it is classified as a",
    "To calculate the area of a rectangle, you multiply the length by the",
    "In the equation 3x = 12, the value of x is",
    "In Python, a 'for' loop is used to iterate over a",
    "HTML stands for HyperText Markup",
    "To print a message to the console in JavaScript, you use",
    "The Great Depression began in 1929 after the crash of the",
    "The primary cause of the seasons on Earth is the tilt of its",
    "Democracy is a form of government where power is held by the",
    "The antonym of 'expand' is",
    "In the sentence 'The cat sat on the mat', the verb is",
    "The plural form of the word 'criterion' is"
]

def log_sample_generation(model, tokenizer, accelerator, global_step):
    model.eval()
    prompt = random.choice(EVAL_PROMPTS)
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    generated_text = ""
    try:
        with torch.no_grad():
            output_ids = model.generate(
                input_ids, 
                max_new_tokens=50,       
                do_sample=True,          
                temperature=0.7,         
                top_k=40,                
                repetition_penalty=1.2,  
                pad_token_id=tokenizer.eos_token_id
            )
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    except Exception as e:
        generated_text = f"[Generation Error]: {e}"

    model.train()

    if accelerator.is_main_process:
        tracker = accelerator.get_tracker("tensorboard")
        if tracker:
            tracker.writer.add_text("Validation/Sample", f"**Prompt:** {prompt}\n\n**Output:** {generated_text}", global_step)
            
    return prompt, generated_text

# ===============================
# 1. Argument Parsing
# ===============================
def parse_args():
    parser = argparse.ArgumentParser(description="Mamba/Transformer Training Script")
    parser.add_argument("--dataset", type=str, default="fineweb-edu", help="dataset name")
    parser.add_argument("--val_dataset", type=str, default="fineweb-edu", help="validation dataset name")
    parser.add_argument("--model_type", type=str, default="long", choices=["gpt2", "mamba", "mamba2", "long"], help="Model architecture")
    parser.add_argument("--model_size", type=str, default="small", help="Model size (small, medium, etc.)")
    
    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=6e-4, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=6500, help="Total training steps")
    parser.add_argument("--seq_len", type=int, default=1024, help="Sequence length")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every X steps")
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluate every X steps")

    # System
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Save VRAM")      
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint or 'latest'")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Save directory")
    parser.add_argument("--output_dir", type=str, default="./output", help="Final model directory")
    parser.add_argument("--log_dir", type=str, default="./logs", help="TensorBoard log directory")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers (set >0 to avoid GPU starvation)")
    
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
# 3. Handle Resume Logic
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
            resume_step = int(os.path.basename(args.resume_from_checkpoint).split("_")[1])
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


# --- CONDITIONAL COMPILATION ---
# Only compile if model_type is 'long', otherwise run standard PyTorch
if args.model_type == "long":
    if accelerator.is_main_process:
        console.print(f"[bold green][Performance] Model Type is '{args.model_type}'. Applying torch.compile()...[/bold green]")
        
    # Check if DeepSpeed/DDP has wrapped the model
    if hasattr(model, "module"):
        # IMPORTANT: Compile the INNER module to avoid RecursionError with DeepSpeed Engine
        model.module = torch.compile(model.module, mode="default")
    else:
        # Fallback for single-GPU/non-DeepSpeed runs
        model = torch.compile(model, mode="default")
else:
    if accelerator.is_main_process:
        console.print(f"[dim][Performance] Model Type is '{args.model_type}'. Skipping torch.compile.[/dim]")


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

model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
    model, optimizer, train_loader, val_loader, scheduler
)

# CRITICAL: Ensure all processes have finished 'preparing' before moving to state loading
accelerator.wait_for_everyone()
accelerator.print("Preparation complete. Moving to training loop.")


if args.resume_from_checkpoint:
    if args.resume_from_checkpoint != "latest" or os.path.exists(args.checkpoint_dir):
        if accelerator.is_main_process:
            console.print(f"[bold yellow]Loading state from {args.resume_from_checkpoint}...[/bold yellow]")
        accelerator.load_state(args.resume_from_checkpoint)
        if accelerator.is_main_process:
            console.print("[green]State loaded successfully.[/green]")

loss_metric = MeanMetric().to(accelerator.device)

# ===============================
# 7. Training Loop
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


data_iter = iter(train_loader)

# UI State
gen_text_display = "Waiting for first evaluation..."
gen_prompt_display = "None"
val_loss_display = "N/A"
grad_norm_display = "0.0"

progress_bar = utils.create_progress_bar()

if accelerator.is_main_process:
    cleaned_log_name = os.path.basename(os.path.normpath(args.log_dir))
    accelerator.init_trackers(cleaned_log_name, config=vars(args))
    train_task_id = progress_bar.add_task("[green]Training...", total=args.max_steps)
    
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
            data_iter = iter(train_loader)
            batch = next(data_iter)

        # --- Forward / Backward ---
        with accelerator.accumulate(model):
            outputs = model(input_ids=batch["input_ids"], labels=batch["input_ids"])
            loss = outputs.loss
            accelerator.backward(loss)
            
            # Clip Gradients & Log Norm (Crucial for Mamba stability)
            if accelerator.sync_gradients:
                # clip_grad_norm_ returns the norm before clipping
                total_norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)
                if isinstance(total_norm, torch.Tensor):
                    grad_norm_display = f"{total_norm.item():.2f}"
                else:
                    grad_norm_display = f"{total_norm:.2f}"
            
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            gathered_loss = accelerator.gather(loss.detach())
            loss_metric.update(gathered_loss.mean())

            
        # --- Step Updates ---
        if accelerator.sync_gradients:
            global_step += 1
            
            # GATHER OUTSIDE ACCUMULATE
            # gathered_loss = accelerator.gather(loss.detach())
            # loss_metric.update(gathered_loss.mean())
            
            current_loss = loss_metric.compute().item()
            loss_metric.reset()
            current_ppl = math.exp(current_loss) if current_loss < 20 else 9999.0
            current_lr = scheduler.get_last_lr()[0]

            if accelerator.is_main_process:
                progress_bar.update(train_task_id, completed=global_step)
                
                # UI Construction
                metrics_table = Table(show_header=False, box=None)
                metrics_table.add_row("Loss", f"[bold red]{current_loss:.4f}[/bold red]")
                metrics_table.add_row("PPL", f"[yellow]{current_ppl:.2f}[/yellow]")
                metrics_table.add_row("LR", f"{current_lr:.2e}")
                metrics_table.add_row("Grad Norm", f"{grad_norm_display}")
                
                status_panel = Panel(metrics_table, title=f"Step {global_step}/{args.max_steps}", border_style="green", width=30)
                gen_panel = Panel(f"[dim]Prompt:[/dim] [cyan]{gen_prompt_display}[/cyan]\n\n{gen_text_display}", title=f"Last Generation (Val Loss: {val_loss_display})", border_style="blue", height=10)
                
                live.update(Group(Panel(progress_bar, border_style="none"), "", status_panel, gen_panel))

            accelerator.log({"train_loss": current_loss, "train_ppl": current_ppl, "lr": current_lr, "grad_norm": float(grad_norm_display)}, step=global_step)
            
            # --- EVALUATION ---
            if global_step % args.eval_steps == 0:
                model.eval()
                val_losses = []
                val_iter = iter(val_loader)
                for _ in range(20):
                    try:
                        vbatch = next(val_iter)
                    except StopIteration:
                        val_iter = iter(val_loader)
                        vbatch = next(val_iter)
                        
                    with torch.no_grad():
                        v_out = model(input_ids=vbatch["input_ids"].to(accelerator.device), labels=vbatch["input_ids"].to(accelerator.device))
                        val_losses.append(accelerator.gather(v_out.loss).mean().item())
                
                avg_val_loss = sum(val_losses) / len(val_losses)
                val_ppl = math.exp(avg_val_loss) if avg_val_loss < 20 else 9999.0
                val_loss_display = f"{avg_val_loss:.4f}"
                accelerator.log({"val_loss": avg_val_loss, "val_ppl": val_ppl}, step=global_step)
                model.train()

                if accelerator.is_main_process:
                    prompt_used, sample_text = log_sample_generation(model, tokenizer, accelerator, global_step)
                    gen_prompt_display = prompt_used
                    gen_text_display = f"[white]{sample_text}[/white]"

            # --- SAVE ---
            if global_step % args.save_steps == 0:
                accelerator.save_state(f"{args.checkpoint_dir}/step_{global_step}")

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