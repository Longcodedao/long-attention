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
# Ensure these files (model_loader.py, dataset.py, utils.py) are in the same directory
from model import model_loader
from dataset import data_loader
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


def log_sample_generation(model, tokenizer, accelerator, global_step, prompt="The history of"):
    """
    Generates a sample and logs it to TensorBoard and Console.
    Uses model.generate for robustness.
    """
    model.eval()
    
    # 1. Prepare Input (Correctly handling devices for Accelerate)
    # We use the device of the model parameters to ensure we are on the right GPU
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # 2. Robust Generation
    # We use a try-catch block because some custom models (like raw Mamba) 
    # might not support .generate() out of the box.
    try:
        with torch.no_grad():
            output_ids = model.generate(
                input_ids, 
                max_new_tokens=100,      # Generate a bit more context
                do_sample=True,          # Add variety
                temperature=0.8,         # 1.0 is creative, 0.0 is deterministic
                top_k=40,                # Limit to top 40 tokens
                repetition_penalty=1.2,  # Prevent loops
                pad_token_id=tokenizer.eos_token_id
            )
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
    except Exception as e:
        # Fallback if the model doesn't support .generate()
        generated_text = f"Generation failed (Model missing .generate()): {e}"

    model.train()

    # 3. Log to Console (So you see it immediately)
    if accelerator.is_main_process:
        # We assume 'console' is available globally or passed in. 
        # If not, use standard print.
        print(f"\n[Step {global_step}] Generated: {generated_text}\n")

    # 4. Log to TensorBoard
    # We access the tracker safely via accelerator
    if accelerator.is_main_process:
        tracker = accelerator.get_tracker("tensorboard")
        if tracker:
            tracker.writer.add_text("Validation/Sample", f"**Step {global_step}:**\n\n{generated_text}", global_step)
            
    return generated_text

# ===============================
# 1. Argument Parsing
# ===============================
def parse_args():
    parser = argparse.ArgumentParser(description="LLM Training Script - WikiText-103")

    # Dataset defaults changed for WikiText
    parser.add_argument("--dataset", type=str, default="wikitext", 
                        help="Name of dataset (must match key in data_loader.py)")
    parser.add_argument("--val_dataset", type=str, default="wikitext", help="Dataset to use for validation")

    # Model Configuration 
    parser.add_argument("--model_type", type=str, default="long", 
                        choices=["holo", "gpt2", "mamba", "mamba2", "long"],
                        help="Type of model to train")
    parser.add_argument("--model_size", type=str, default="small", 
                        help="Model size preset (e.g. small, medium, 187m). 'small' is good for WikiText.")
    
    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=8, help="Per-device batch size (Higher is better for speed)")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=6e-4, help="Learning rate (WikiText often tolerates higher LR)")
    parser.add_argument("--max_steps", type=int, default=20000, help="Total training steps")
    parser.add_argument("--seq_len", type=int, default=1024, help="Sequence length (WikiText articles are rarely super long)")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps for scheduler")
    parser.add_argument("--save_steps", type=int, default=1000, help="Steps interval for saving checkpoints")
    parser.add_argument("--eval_steps", type=int, default=200, help="Run evaluation every X steps")

    # Optimization
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing to save memory")     

    # Checkpointing
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint (e.g. 'latest')")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/wikitext_long", help="Directory to save checkpoints")
    parser.add_argument("--output_dir", type=str, default="./output/wikitext_long", help="Directory to save the final model")
    
    parser.add_argument("--log_dir", type=str, default="./logs/wikitext", help="Directory for TensorBoard logs")
    
    return parser.parse_args()

args = parse_args()

# ===============================
# 2. Setup Accelerator & Logging
# ===============================
accelerator = Accelerator(
    gradient_accumulation_steps=args.grad_accum_steps,
    log_with="tensorboard",
    project_dir="."
)
set_seed(42)
console = utils.get_console(accelerator)

if args.model_type == "mamba" and accelerator.mixed_precision == "no":
    if accelerator.is_main_process:
        console.print("[bold red]WARNING: Training Mamba in FP32 is slow. Use --bf16 via 'accelerate config'.[/bold red]")

if accelerator.is_main_process:
    utils.print_config_table(console, accelerator, args)
    console.print(f"[bold green]Starting Training for {args.model_type.upper()} on WikiText-103...[/bold green]")


# ===============================
# 3. Load Model & Tokenizer
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
# 4. Load Data (WikiText Specific)
# ===============================
# The logic for 'wikitext' is handled inside data_loader.py (which we updated previously)
train_loader = data_loader.get_dataloader(
    console, accelerator, tokenizer, 
    args.dataset, args.batch_size, args.seq_len, split="train"
)

val_loader = data_loader.get_dataloader(
    console, accelerator, tokenizer, 
    args.val_dataset, args.batch_size, args.seq_len, split="validation"
)

# ===============================
# 5. Optimizer & Scheduler
# ===============================
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps
)

model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
    model, optimizer, train_loader, val_loader, scheduler
)

loss_metric = MeanMetric().to(accelerator.device)

# ===============================
# 6. Resume Logic
# ===============================
global_step = 0
resume_step = 0

if args.resume_from_checkpoint:
    path = args.resume_from_checkpoint
    if path == "latest":
        chk_dir = args.checkpoint_dir
        if os.path.exists(chk_dir):
            folders = [os.path.join(chk_dir, d) for d in os.listdir(chk_dir) if d.startswith("step_")]
            if folders:
                folders.sort(key=lambda x: int(x.split("_")[-1]))
                path = folders[-1]
            else:
                path = None 
        else:
            path = None

    if path and os.path.exists(path):
        accelerator.load_state(path)
        try:
            resume_step = int(os.path.basename(path).split("_")[-1])
            global_step = resume_step
        except ValueError:
            resume_step = 0

        if accelerator.is_main_process:
            console.print(f"[bold yellow]Resuming training from: {path} (Step: {resume_step})[/bold yellow]")

        active_dataloader = accelerator.skip_first_batches(train_loader, resume_step * args.grad_accum_steps)
    else:
        if accelerator.is_main_process:
            console.print(f"[bold red]Checkpoint '{path}' not found. Starting from scratch.[/bold red]")
        active_dataloader = train_loader
else:
    active_dataloader = train_loader

accelerator.wait_for_everyone()

# ===============================
# 7. The Evaluation Function
# ===============================
def evaluate(model, val_loader, max_eval_batches=50):
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
            gathered_loss = accelerator.gather(outputs.loss).mean()
            losses.append(gathered_loss.item())

    model.train() 
    return sum(losses) / len(losses) if losses else float("inf")

# ===============================
# 8. Training Loop
# ===============================
model.train()
data_iter = iter(active_dataloader)

# UI State
val_message = "[dim]Waiting for first evaluation...[/dim]"
val_border_style = "dim" 
last_eval_step = 0
system_message = None

progress_bar = utils.create_progress_bar()

if accelerator.is_main_process:
    accelerator.init_trackers(args.log_dir, config=vars(args))     
    train_task_id = progress_bar.add_task("[green]Training...", total=args.max_steps)
    live = Live(
        console=console, 
        refresh_per_second = 2, # Slightly faster refresh for small dataset
        redirect_stdout = True, 
        redirect_stderr = True   
    )
    live.start()
    
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    tokenizer.save_pretrained(os.path.join(args.checkpoint_dir, "tokenizer"))

try:
    while global_step < args.max_steps:
        # --- TRAINING STEP ---
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        with accelerator.accumulate(model):
            outputs = model(input_ids=batch["input_ids"], labels=batch["input_ids"])
            loss = outputs.loss
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            gathered_loss = accelerator.gather(loss.detach())
            loss_metric.update(gathered_loss.mean())

        # --- POST-STEP ---
        if accelerator.sync_gradients:
            global_step += 1
            
            current_loss = loss_metric.compute().item()
            try:
                current_ppl = math.exp(current_loss)
            except OverflowError:
                current_ppl = float("inf")
            current_lr = scheduler.get_last_lr()[0]

            if accelerator.is_main_process:
                progress_bar.update(train_task_id, completed=global_step)
                
                metrics_panel = utils.create_metrics_table(
                    global_step, args.max_steps, current_loss, current_ppl, current_lr
                )
                
                val_panel = Panel(
                    val_message, 
                    title="Validation Status", 
                    border_style=val_border_style,
                    width=60
                )

                ui_elements = [metrics_panel, val_panel]
                if system_message:
                    ui_elements.append(Panel(system_message, border_style="yellow", title="System"))
                ui_elements.append(progress_bar)
                
                live.update(Group(*ui_elements))

            accelerator.log({"train_loss": current_loss, "train_ppl": current_ppl, "lr": current_lr}, step=global_step)
            
            # --- EVALUATION ---
            if global_step % args.eval_steps == 0:
                if accelerator.is_main_process:
                    val_message = "[bold yellow]Running Evaluation...[/bold yellow]"
                    val_border_style = "yellow"
                    live.update(Group(metrics_panel, Panel(val_message, title="Validation Status", border_style=val_border_style, width=60), progress_bar))
                
                val_loss = evaluate(model, val_loader)
                val_ppl = math.exp(val_loss) if val_loss < 20 else float("inf")

                # 2. Generate & Log Sample (NEW)
                # Only run on main process to save time/compute
                if accelerator.is_main_process:
                   log_sample_generation(model, tokenizer, accelerator, global_step, prompt="The history of")
                # last_eval_step = global_step
                
                accelerator.log({"val_loss": val_loss, "val_ppl": val_ppl}, step=global_step)

                if accelerator.is_main_process:
                    val_message = f"[bold cyan]Last Eval (Step {last_eval_step}):[/bold cyan]\nLoss: {val_loss:.4f} | PPL: {val_ppl:.2f}"
                    val_border_style = "cyan"        
                    
            # --- SAVE ---
            if global_step % args.save_steps == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    system_message = f"[bold yellow]Saving checkpoint to step_{global_step}...[/bold yellow]"                      
                    live.update(Group(metrics_panel, val_panel, Panel(system_message, border_style="yellow"), progress_bar))
                    
                accelerator.save_state(f"{args.checkpoint_dir}/step_{global_step}")
                
                if accelerator.is_main_process:
                    system_message = None
                accelerator.wait_for_everyone()

    # --- FINALIZE ---
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        console.print("[bold yellow]Training Complete. Running Final Evaluation...[/bold yellow]")
        
    final_val_loss = evaluate(model, val_loader, max_eval_batches=100)
    final_val_ppl = math.exp(final_val_loss) if final_val_loss < 20 else float("inf")
    
    if accelerator.is_main_process:
        live.stop() 
        
        console.print("[bold blue]Saving final model...[/bold blue]")
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, 
            is_main_process=accelerator.is_main_process, 
            save_function=accelerator.save,
            safe_serialization=False
        )
        tokenizer.save_pretrained(args.output_dir)
        
        from rich.table import Table
        from rich.box import DOUBLE_EDGE
        summary_table = Table(title="[bold green]WikiText-103 Results[/bold green]", box = DOUBLE_EDGE)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="magenta")

        summary_table.add_row("Final Val Loss", f"{final_val_loss:.4f}")
        summary_table.add_row("Final Val PPL", f"{final_val_ppl:.2f}")

        console.print(summary_table)
        console.print(f"[bold green]✅ Saved to: {args.output_dir}[/bold green]")
        
finally:
    if accelerator.is_main_process:
        try:
            live.stop()
        except:
            pass
    accelerator.end_training()