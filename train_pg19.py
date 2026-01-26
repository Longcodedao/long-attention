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
import torch.nn.functional as F

# --- LOCAL IMPORTS ---
from model import model_loader
from dataset import data_loader
import utils
import datasets
import transformers
import warnings

# Optimization for PG19
datasets.disable_progress_bar()
warnings.filterwarnings("ignore", message=".*barrier().*")

def log_sample_generation(model, tokenizer, accelerator, global_step, prompt="The mysterious book began with"):
    model.eval()
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    try:
        with torch.no_grad():
            output_ids = model.generate(
                input_ids, 
                max_new_tokens=128, # Books need more tokens to show coherence
                do_sample=True,
                temperature=0.8,
                top_k=40,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    except Exception as e:
        generated_text = f"Generation failed: {e}"

    model.train()
    if accelerator.is_main_process:
        # Log to TensorBoard
        tracker = accelerator.get_tracker("tensorboard")
        if tracker:
            tracker.writer.add_text("Samples/PG19", f"**Step {global_step}**\n\n{generated_text}", global_step)
        # Print to console above the Live display
        print(f"\n[Step {global_step}] Sample: {generated_text[:200]}...\n")

def parse_args():
    parser = argparse.ArgumentParser(description="LLM Training Script - PG19")
    parser.add_argument("--dataset", type=str, default="pg19")
    parser.add_argument("--val_dataset", type=str, default="pg19")
    parser.add_argument("--model_type", type=str, default="long", choices=["holo", "gpt2", "mamba", "mamba2", "long"])
    parser.add_argument("--model_size", type=str, default="small")
    
    # PG19 specific hyperparams
    parser.add_argument("--batch_size", type=int, default=4) # Reduced for longer seq_len
    parser.add_argument("--grad_accum_steps", type=int, default=8) 
    parser.add_argument("--lr", type=float, default=2e-4) # Lower LR for stability on large data
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--seq_len", type=int, default=2048) # PG19 standard is 2048+
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--save_steps", type=int, default=2500)
    parser.add_argument("--eval_steps", type=int, default=500)

    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/pg19_long")
    parser.add_argument("--output_dir", type=str, default="./output/pg19_long")
    parser.add_argument("--log_dir", type=str, default="./logs/pg19")
    
    return parser.parse_args()

def evaluate(model, val_loader, max_eval_batches=30):
    model.eval()
    losses = []
    val_iter = iter(val_loader)
    with torch.no_grad():
        for _ in range(max_eval_batches):
            try:
                batch = next(val_iter)
            except StopIteration:
                break
            outputs = model(input_ids=batch["input_ids"], labels=batch["input_ids"])
            losses.append(accelerator.gather(outputs.loss).mean().item())
    model.train()
    return sum(losses) / len(losses) if losses else float("inf")

# --- MAIN EXECUTION ---
args = parse_args()
accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum_steps, log_with="tensorboard", project_dir=".")
set_seed(42)
console = utils.get_console(accelerator)

# Load Model
model, tokenizer = model_loader.get_model_and_tokenizer(
    model_type=args.model_type,
    model_size=args.model_size,
    seq_len=args.seq_len,
    device=accelerator.device
)

if args.gradient_checkpointing:
    model.gradient_checkpointing_enable()

# Load Data
train_loader = data_loader.get_dataloader(console,
                                          accelerator, 
                                          tokenizer,
                                          args.dataset, 
                                          args.batch_size, 
                                          args.seq_len, 
                                          split="train")
val_loader = data_loader.get_dataloader(console, accelerator, tokenizer, args.val_dataset, args.batch_size, args.seq_len, split="validation")

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps)

model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, val_loader, scheduler)
loss_metric = MeanMetric().to(accelerator.device)

# Resume logic (Omitted for brevity, keep your original logic here)
global_step = 0
data_iter = iter(train_loader)

# UI Setup
progress_bar = utils.create_progress_bar()
val_message = "[dim]Waiting for PG19 evaluation...[/dim]"
val_border_style = "dim"
last_eval_step = 0

if accelerator.is_main_process:
    accelerator.init_trackers(args.log_dir, config=vars(args))
    train_task_id = progress_bar.add_task("[cyan]PG19 Training", total=args.max_steps)
    live = Live(console=console, refresh_per_second=2, redirect_stdout=True)
    live.start()

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
            curr_loss = loss_metric.compute().item()
            curr_ppl = math.exp(curr_loss) if curr_loss < 20 else float("inf")
            
            if accelerator.is_main_process:
                progress_bar.update(train_task_id, completed=global_step)
                metrics_panel = utils.create_metrics_table(global_step, args.max_steps, curr_loss, curr_ppl, scheduler.get_last_lr()[0])
                val_panel = Panel(val_message, title="Eval Status", border_style=val_border_style, width=60)
                live.update(Group(metrics_panel, val_panel, progress_bar))

            accelerator.log({"train_loss": curr_loss, "train_ppl": curr_ppl}, step=global_step)

            if global_step % args.eval_steps == 0:
                val_loss = evaluate(model, val_loader)
                val_ppl = math.exp(val_loss) if val_loss < 20 else float("inf")
                last_eval_step = global_step
                
                if accelerator.is_main_process:
                    log_sample_generation(model, tokenizer, accelerator, global_step)
                    val_message = f"[bold green]Step {last_eval_step}[/bold green]\nLoss: {val_loss:.4f} | PPL: {val_ppl:.2f}"
                    val_border_style = "green"

                accelerator.log({"val_loss": val_loss, "val_ppl": val_ppl}, step=global_step)

            if global_step % args.save_steps == 0:
                accelerator.wait_for_everyone()
                accelerator.save_state(f"{args.checkpoint_dir}/step_{global_step}")

finally:
    if accelerator.is_main_process:
        live.stop()
    accelerator.end_training()