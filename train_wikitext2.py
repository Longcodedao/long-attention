import torch
import argparse
import datasets
import transformers
import warnings
import torch.nn.functional as F

from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from accelerate.utils import set_seed
from torchmetrics import MeanMetric
from rich.live import Live
from rich.panel import Panel
from rich.console import Group

# --- LOCAL IMPORTS ---
from model import model_loader
from dataset import data_loader
import utils
import sys
import math

# Suppress non-critical warnings
datasets.disable_progress_bar()
warnings.filterwarnings("ignore", message=".*barrier().*")

def log_sample_generation(model, tokenizer, accelerator, global_step, prompt="The Valkyrie"):
    """
    Generates text to visually verify the model is learning structures.
    WikiText-2 often contains encyclopedia headers, so we use a relevant prompt.
    """
    model.eval()
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    try:
        with torch.no_grad():
            output_ids = model.generate(
                input_ids, 
                max_new_tokens=64, 
                do_sample=True,
                temperature=0.7,
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
            tracker.writer.add_text("Samples/WikiText2", f"**Step {global_step}**\n\n{generated_text}", global_step)
        
        # Print clearly to console
        print(f"\n[Step {global_step}] Sample: {generated_text.replace(prompt, f'[bold]{prompt}[/bold]')}\n")


def perform_sanity_check(model, train_loader, accelerator, console):
    """
    Runs a single forward pass to check for shape errors, memory usage, and basic stability.
    """
    if accelerator.is_main_process:
        console.print("[bold yellow]Running Pre-training Sanity Check...[/bold yellow]")
    
    model.train()
    try:
        # Get a batch from the raw loader (currently on CPU)
        batch = next(iter(train_loader))
        
        # --- CRITICAL FIX: Manually move data to the accelerator's device ---
        # We have to do this because accelerator.prepare() hasn't been called on the loader yet
        input_ids = batch["input_ids"].to(accelerator.device)
        labels = batch["labels"].to(accelerator.device)
        # --------------------------------------------------------------------

        with accelerator.accumulate(model):
            # Pass the GPU tensors to the model
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            accelerator.backward(loss)
            
        mem_usage = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        if accelerator.is_main_process:
            console.print(f"[green]✓ Forward/Backward Pass Successful.[/green]")
            console.print(f"[green]✓ Initial Loss: {loss.item():.4f}[/green]")
            console.print(f"[green]✓ Peak Memory: {mem_usage:.2f} GB[/green]")
            console.print("-" * 40)
            
    except Exception as e:
        console.print(f"[bold red]CRITICAL: Sanity Check Failed![/bold red]")
        console.print(f"Error: {e}")
        # Print the traceback to see exactly where inside the model it failed
        import traceback
        traceback.print_exc()
        sys.exit(1)        


def parse_args():
    parser = argparse.ArgumentParser(description="LLM Training Script - WikiText-2")
    
    # Dataset config
    parser.add_argument("--dataset", type=str, default="wikitext-2", help="wikitext-2 or wikitext-103")
    parser.add_argument("--val_dataset", type=str, default="wikitext-2")
    
    # Model config
    parser.add_argument("--model_type", type=str, default="long", choices=["holo", "gpt2", "mamba", "mamba2", "long"])
    parser.add_argument("--model_size", type=str, default="small")
    
    # Training Hyperparams
    parser.add_argument("--batch_size", type=int, default=4) 
    parser.add_argument("--grad_accum_steps", type=int, default=4) 
    parser.add_argument("--lr", type=float, default=5e-4) # Slightly higher LR for smaller dataset/faster convergence
    parser.add_argument("--seq_len", type=int, default=2048) 
    
    # Steps (Scaled down for WikiText-2)
    parser.add_argument("--max_steps", type=int, default=2000) 
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=100) # Frequent eval for debugging

    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/wikitext2")
    parser.add_argument("--output_dir", type=str, default="./output/wikitext2")
    parser.add_argument("--log_dir", type=str, default="./logs/wikitext2")
    
    return parser.parse_args()


def evaluate(model, val_loader, accelerator, max_eval_batches=20):
    model.eval()
    losses = []
    val_iter = iter(val_loader)
    
    with torch.no_grad():
        for i in range(max_eval_batches):
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

# 1. Load Model
# Note: vocab_size=None lets the loader detect it from the tokenizer (usually ~50k for GPT2 tokenizer)
model, tokenizer = model_loader.get_model_and_tokenizer(
    model_type=args.model_type,
    model_size=args.model_size,
    seq_len=args.seq_len,
    device=accelerator.device
)

if args.gradient_checkpointing:
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    elif hasattr(model, "_set_gradient_checkpointing"):
        # Handle custom models like LongModel
        model._set_gradient_checkpointing(module=model.long_model, value=True)


# 2. Load Data
train_loader = data_loader.get_dataloader(
    console, accelerator, tokenizer, 
    args.dataset, args.batch_size, args.seq_len, split="train"
)
val_loader = data_loader.get_dataloader(
    console, accelerator, tokenizer, 
    args.val_dataset, args.batch_size, args.seq_len, split="validation"
)

# 3. Sanity Check (Before Optimizer init to fail fast)
perform_sanity_check(model, train_loader, accelerator, console)

# 4. Optimization
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps)

model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, val_loader, scheduler)
loss_metric = MeanMetric().to(accelerator.device)


# 5. Resume logic
global_step = 0
batches_to_skip = 0

if args.resume_from_checkpoint:
    if args.resume_from_checkpoint != "":
        accelerator.print(f"Resuming training from: {args.resume_from_checkpoint}")
        accelerator.load_state(args.resume_from_checkpoint)
        try:
            step_str = os.path.basename(os.path.normpath(args.resume_from_checkpoint))
            global_step = int(step_str.split("_")[-1])
        except ValueError:
            accelerator.print("Warning: Could not parse global step. Starting step count at 0.")

        batches_to_skip = global_step * args.grad_accum_steps
        accelerator.print(f"  -> Resuming at Global Step {global_step}")
        accelerator.print(f"  -> Fast-forwarding {batches_to_skip} batches...")

        # Reload train_loader with skip logic
        train_loader = data_loader.get_dataloader(
            console, accelerator, tokenizer, 
            args.dataset, args.batch_size, args.seq_len, 
            split="train", fast_skip_batches=batches_to_skip
        )
        train_loader = accelerator.prepare(train_loader)

data_iter = iter(train_loader)

# 6. UI Setup
progress_bar = utils.create_progress_bar()
val_message = "[dim]Waiting for validation...[/dim]"
val_border_style = "dim"

if accelerator.is_main_process:
    accelerator.init_trackers(args.log_dir, config=vars(args))
    train_task_id = progress_bar.add_task("[cyan]WikiText-2 Training", total=args.max_steps)
    live = Live(console=console, refresh_per_second=4, redirect_stdout=True) # Higher refresh for fast debug
    live.start()


# 7. Training Loop
try:
    while global_step < args.max_steps:
        # Data Loop Handling
        try:
            batch = next(data_iter)
        except StopIteration:
            # For WikiText-2, we hit the end quickly. We must restart the iterator.
            # This enables "Epoch" behavior within a "Step" based loop.
            if accelerator.is_main_process:
                console.print("[dim]Dataset exhausted. Restarting iterator...[/dim]")
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
                val_loss = evaluate(model, val_loader, accelerator)
                val_ppl = math.exp(val_loss) if val_loss < 20 else float("inf")
                
                if accelerator.is_main_process:
                    log_sample_generation(model, tokenizer, accelerator, global_step)
                    val_message = f"[bold green]Step {global_step}[/bold green]\nLoss: {val_loss:.4f} | PPL: {val_ppl:.2f}"
                    val_border_style = "green"

                accelerator.log({"val_loss": val_loss, "val_ppl": val_ppl}, step=global_step)

            if global_step % args.save_steps == 0:
                accelerator.wait_for_everyone()
                accelerator.save_state(f"{args.checkpoint_dir}/step_{global_step}")

    # --- SAVE FINAL MODEL ---
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        console.print(f"\n[bold cyan]Training Finished! Saving final model to {args.output_dir}...[/bold cyan]")
        
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, 
            is_main_process=accelerator.is_main_process, 
            save_function=accelerator.save
        )
        if tokenizer:
            tokenizer.save_pretrained(args.output_dir)
            
        console.print(f"[bold green]✓ Model saved successfully to {args.output_dir}[/bold green]")

finally:
    if accelerator.is_main_process:
        live.stop()
    accelerator.end_training()