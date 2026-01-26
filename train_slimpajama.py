import sys
import os
import math
import torch
import argparse
import warnings
from accelerate import Accelerator
from accelerate.utils import set_seed
from torchmetrics import MeanMetric
from rich.live import Live
from rich.panel import Panel
from rich.console import Group
from transformers import get_cosine_schedule_with_warmup

# --- LOCAL IMPORTS ---
from model import model_loader
from dataset import data_loader
import utils
import datasets
import transformers

# Silence verbose logging
datasets.disable_progress_bar()
datasets.utils.logging.set_verbosity_error()
transformers.utils.logging.set_verbosity_error()
warnings.filterwarnings("ignore", message=".*barrier().*")

# ===============================
# 1. Generation Helper
# ===============================
def log_sample_generation(model, tokenizer, accelerator, global_step, prompt="The meaning of life is"):
    """Generates a sample and logs it to TensorBoard and Console."""
    model.eval()
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    try:
        with torch.no_grad():
            output_ids = model.generate(
                input_ids, 
                max_new_tokens=50,
                do_sample=True,
                temperature=0.8,
                top_k=40,
                pad_token_id=tokenizer.eos_token_id
            )
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    except Exception as e:
        generated_text = f"Generation failed: {e}"

    model.train()

    if accelerator.is_main_process:
        print(f"\n[Step {global_step}] Sample: {generated_text}\n")
        tracker = accelerator.get_tracker("tensorboard")
        if tracker:
            tracker.writer.add_text("Samples", f"**Step {global_step}:**\n\n{generated_text}", global_step)
    
    return generated_text

# ===============================
# 2. Argument Parsing
# ===============================
def parse_args():
    parser = argparse.ArgumentParser(description="SlimPajama Training Script")
    
    # Dataset
    parser.add_argument("--dataset", type=str, default="slimpajama_6b")
    parser.add_argument("--val_dataset", type=str, default="slimpajama_6b")

    # Model Configuration 
    parser.add_argument("--model_type", type=str, default="long", choices=["holo", "gpt2", "mamba", "mamba2", "long"])
    parser.add_argument("--model_size", type=str, default="187m")
    
    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=250)

    # Checkpointing
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/slimpajama_long")
    parser.add_argument("--output_dir", type=str, default="./output/slimpajama_long")
    parser.add_argument("--log_dir", type=str, default="./logs/slimpajama")
    
    return parser.parse_args()

args = parse_args()

# ===============================
# 3. Setup Accelerator
# ===============================
accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum_steps, log_with="tensorboard", project_dir=".")
set_seed(42)
console = utils.get_console(accelerator)

if accelerator.is_main_process:
    utils.print_config_table(console, accelerator, args)

# ===============================
# 4. Load Model & Data
# ===============================
model, tokenizer = model_loader.get_model_and_tokenizer(
    model_type=args.model_type,
    model_size=args.model_size,
    seq_len=args.seq_len,
    device=accelerator.device
)


train_loader = data_loader.get_dataloader(console, accelerator, tokenizer, args.dataset, args.batch_size, args.seq_len, split="train")
val_loader = data_loader.get_dataloader(console, accelerator, tokenizer, args.val_dataset, args.batch_size, args.seq_len, split="validation")

# ===============================
# 5. Optimizer & Scheduler
# ===============================
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps)

model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, val_loader, scheduler)

loss_metric = MeanMetric().to(accelerator.device)

# ===============================
# 6. Training Loop Logic
# ===============================
def evaluate(model, val_loader, max_eval_batches=50):
    model.eval()
    losses = []
    val_iter = iter(val_loader)
    with torch.no_grad():
        for _ in range(max_eval_batches):
            try: batch = next(val_iter)
            except StopIteration: break
            outputs = model(input_ids=batch["input_ids"], labels=batch["input_ids"])
            losses.append(accelerator.gather(outputs.loss).mean().item())
    model.train()
    return sum(losses) / len(losses) if losses else float("inf")

global_step = 0
val_message = "[dim]Waiting for first evaluation...[/dim]"
val_border_style = "dim"
system_message = None

if accelerator.is_main_process:
    accelerator.init_trackers(args.log_dir, config=vars(args))
    progress_bar = utils.create_progress_bar()
    train_task_id = progress_bar.add_task("[green]Training...", total=args.max_steps)
    live = Live(console=console, refresh_per_second=2, redirect_stdout=True, redirect_stderr=True)
    live.start()

# --- MAIN LOOP ---
data_iter = iter(train_loader)

try:
    while global_step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        with accelerator.accumulate(model):
            outputs = model(input_ids=batch["input_ids"], labels=batch["input_ids"])
            accelerator.backward(outputs.loss)
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            loss_metric.update(accelerator.gather(outputs.loss.detach()).mean())

        if accelerator.sync_gradients:
            global_step += 1
            
            # UI Update
            if accelerator.is_main_process:
                current_loss = loss_metric.compute().item()
                current_lr = scheduler.get_last_lr()[0]
                metrics_panel = utils.create_metrics_table(global_step, args.max_steps, current_loss, math.exp(current_loss), current_lr)
                
                ui_elements = [metrics_panel, Panel(val_message, title="Validation", border_style=val_border_style, width=60)]
                if system_message: ui_elements.append(Panel(system_message, border_style="yellow"))
                ui_elements.append(progress_bar)
                
                progress_bar.update(train_task_id, completed=global_step)
                live.update(Group(*ui_elements))
                accelerator.log({"train_loss": current_loss, "lr": current_lr}, step=global_step)

            # Evaluation
            if global_step % args.eval_steps == 0:
                val_loss = evaluate(model, val_loader)
                log_sample_generation(model, tokenizer, accelerator, global_step)
                
                if accelerator.is_main_process:
                    val_message = f"Step {global_step} | Loss: {val_loss:.4f} | PPL: {math.exp(val_loss):.2f}"
                    val_border_style = "cyan"
                    accelerator.log({"val_loss": val_loss}, step=global_step)

            # Save
            if global_step % args.save_steps == 0:
                accelerator.save_state(os.path.join(args.checkpoint_dir, f"step_{global_step}"))

finally:
    if accelerator.is_main_process:
        live.stop()
    accelerator.end_training()