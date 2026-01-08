import sys
import os
import math
import torch
from transformers import get_cosine_schedule_with_warmup
import argparse
from accelerate import Accelerator
from accelerate.utils import set_seed
from torchmetrics import MeanMetric
from torchmetrics.text import Perplexity
from rich.live import Live
from rich.panel import Panel
from rich.console import Group

# --- LOCAL IMPORTS ---
from model.configuration_holo import HoloConfig
from model.modeling_holo import HoloForCausalLM
from dataset import data_loader
import utils
import datasets
import transformers
import warnings

# Disable Hugging Face progress bars and set logging to error only
datasets.disable_progress_bar()
datasets.utils.logging.set_verbosity_error()
transformers.utils.logging.set_verbosity_error()

# Silence the specific Torch distributed warning seen in your logs
warnings.filterwarnings("ignore", message=".*barrier().*")

# ===============================
# 1. Argument Parsing
# ===============================
def parse_args():
    parser = argparse.ArgumentParser(description="Holo Training Script")

    # Dataset 
    parser.add_argument("--dataset", type=str, default="slimpajama_6b", 
                        help="Name of dataset ('slimpajama_6b') or path to local file ('./data/train.jsonl')")
    parser.add_argument("--val_dataset", type=str, default="slimpajama_6b", help="Dataset to use for validation")
    
    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=1000, help="Total training steps")
    parser.add_argument("--seq_len", type=int, default=2048, help="Sequence length (context window)")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps for scheduler")
    parser.add_argument("--save_steps", type=int, default=200, help="Steps interval for saving checkpoints")
    parser.add_argument("--eval_steps", type=int, default=10, help="Run evaluation every X steps")
    
    # Model Configuration
    parser.add_argument("--model_size", type=str, default="small", choices=["small", "small+", "medium", "medium+", "large", "large+"], help="Holo model preset")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing to save memory")    

    # Checkpointing
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint (e.g. 'latest' or './holo_checkpoints/step_500')")
    parser.add_argument("--output_dir", type=str, default="./holo_checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--log_dir", type=str, default="./holo_logs", help="Directory for TensorBoard logs")
    
    return parser.parse_args()

args = parse_args()

# ===============================
# 2. Setup Accelerator & Logging
# ===============================
accelerator = Accelerator(
    gradient_accumulation_steps=args.grad_accum_steps,
    # mixed_precision=args.mixed_precision,
    log_with="tensorboard",
    project_dir="."
)
set_seed(42)

console = utils.get_console(accelerator)

if accelerator.is_main_process:
    utils.print_config_table(console, accelerator, args)
    console.print(f"[bold green]Starting Training...[/bold green]")


# ===============================
# 3. Load Data & Model
# ===============================
train_loader, tokenizer = data_loader.get_dataloader(
    console,
    accelerator, 
    args.dataset,
    args.batch_size, 
    args.seq_len,
    split = "train"
)
val_loader, _ = data_loader.get_dataloader(
    console,
    accelerator, 
    args.val_dataset,
    args.batch_size, 
    args.seq_len,
    split = "validation"
)

config = HoloConfig.from_preset(args.model_size, use_version=2)
model = HoloForCausalLM(config)

# --- ADD THIS BLOCK ---
if args.gradient_checkpointing:
    # This method is standard in Hugging Face PreTrainedModel
    # It tells the model to not store intermediate activations
    model.gradient_checkpointing_enable()
    
    # If using a custom model that doesn't inherit from PreTrainedModel, 
    # you might need config.use_cache = False as well
    if hasattr(config, "use_cache"):
        config.use_cache = False

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps
)

# Prepare with Accelerator
model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
    model, optimizer, train_loader, val_loader, scheduler
)

# Metrics
loss_metric = MeanMetric().to(accelerator.device)

# ===============================
# 4. Resume Logic
# ===============================
global_step = 0
resume_step = 0

if args.resume_from_checkpoint:
    path = args.resume_from_checkpoint

    # Automatic 'latest' detection
    if path == "latest":
        chk_dir = args.output_dir 
        if os.path.exists(chk_dir):
            folders = [os.path.join(chk_dir, d) for d in os.listdir(chk_dir) if d.startswith("step_")]
            if folders:
                folders.sort(key=lambda x: int(x.split("_")[-1]))
                path = folders[-1]
            else:
                path = None 
        else:
            path = None

    # Sync path across all processes to prevent one GPU starting while others wait
    if path and os.path.exists(path):
        # 1. Load the state (Weights, Optimizer, Scheduler)
        # This must happen on ALL processes simultaneously
        accelerator.load_state(path)
        
        # 2. Extract step number for logging and skipping
        try:
            resume_step = int(os.path.basename(path).split("_")[-1])
            global_step = resume_step
        except ValueError:
            resume_step = 0

        if accelerator.is_main_process:
            console.print(f"[bold yellow]Resuming training from: {path} (Step: {resume_step})[/bold yellow]")

        # 3. Synchronized Data Skipping
        # We must skip resume_step * gradient_accumulation_steps to align the stream
        # accelerator.skip_first_batches handles the sync across GPUs automatically
        active_dataloader = accelerator.skip_first_batches(train_loader, resume_step * args.grad_accum_steps)
        
        if accelerator.is_main_process:
            console.print(f"[dim]Skipping {resume_step * args.grad_accum_steps} total batches to resume at step {resume_step}...[/dim]")
    else:
        if accelerator.is_main_process:
            console.print(f"[bold red]Checkpoint '{path}' not found. Starting from scratch.[/bold red]")
        active_dataloader = train_loader
else:
    active_dataloader = train_loader

# Ensure all processes are synced before entering the loop
accelerator.wait_for_everyone()


# ===============================
# 5. The Evaluation Function
# ===============================
def evaluate(model, val_loader, max_eval_batches=50):
    """
    Runs a quick evaluation loop. 
    We cap it at 50 batches to avoid waiting too long on huge datasets.
    """
    model.eval() # Switch to evaluation mode (disable dropout, etc.)
    losses = []
    
    # We create a new iterator for validation to ensure we start fresh or continue
    val_iter = iter(val_loader)
    
    with torch.no_grad(): # Disable gradient calculation (saves massive memory)
        for i in range(max_eval_batches):
            try:
                batch = next(val_iter)
            except StopIteration:
                break
            outputs = model(input_ids=batch["input_ids"], labels=batch["input_ids"])
            # Gather loss from all GPUs
            gathered_loss = accelerator.gather(outputs.loss).mean()
            losses.append(gathered_loss.item())

    model.train() # Switch back to training mode
    return sum(losses) / len(losses) if losses else float("inf")

# ===============================
# 6. Training Loop
# ===============================
model.train()
data_iter = iter(train_loader)

# --- GLOBAL UI STATE VARIABLES ---
# These hold the current content of the validation panel
val_message = "[dim]Waiting for first evaluation...[/dim]"
val_border_style = "dim" 
last_eval_step = 0
system_message = None

# Setup Dashboard
progress_bar = utils.create_progress_bar()

if accelerator.is_main_process:
    # Initialize TensorBoard with dynamic config
    accelerator.init_trackers(args.log_dir, config=vars(args))    
    train_task_id = progress_bar.add_task("[green]Training...", total=args.max_steps)
    live = Live(
        console=console, 
        refresh_per_second = 1,
        redirect_stdout = True,  # Traps prints so they don't break UI
        redirect_stderr = True   # Traps warnings so they don't break UI
    )
    live.start()
    
    # Save tokenizer once at start (safe because it doesn't change)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    tokenizer.save_pretrained(os.path.join(args.output_dir, "tokenizer"))

try:
    while global_step < args.max_steps:
        # --- A. TRAINING STEP ---
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

        # --- B. POST-STEP LOGIC ---
        if accelerator.sync_gradients:
            global_step += 1
            
            # 1. Metric Calculations
            current_loss = loss_metric.compute().item()
            try:
                current_ppl = math.exp(current_loss)
            except OverflowError:
                current_ppl = float("inf")
            current_lr = scheduler.get_last_lr()[0]

            # 2. UI UPDATE (The "Global Panel" Logic)
            if accelerator.is_main_process:
                progress_bar.update(train_task_id, completed=global_step)
                
                # Create the tables/panels using the current state variables
                metrics_panel = utils.create_metrics_table(
                    global_step, args.max_steps, current_loss, current_ppl, current_lr
                )
                
                # We always create the panel here, using the global 'val_message' variable
                val_panel = Panel(
                    val_message, 
                    title="Validation Status", 
                    border_style=val_border_style,
                    width=60
                )

                # --- NEW: Build the UI Group dynamically ---
                ui_elements = [metrics_panel, val_panel]
                # If there is a system message (like saving), add it to the stack
                if system_message:
                    ui_elements.append(Panel(system_message, border_style="yellow", title="System"))
                ui_elements.append(progress_bar)
                
                live.update(Group(*ui_elements))

            
            # 3. Log Training Data
            accelerator.log({
                "train_loss": current_loss, 
                "train_ppl": current_ppl, 
                "lr": current_lr
            }, step=global_step)
            
            # --- C. EVALUATION TRIGGER ---
            if global_step % args.eval_steps == 0:
                
                # 1. Update State to "Running" & Force Refresh
                if accelerator.is_main_process:
                    val_message = "[bold yellow]Running Evaluation... (Please Wait)[/bold yellow]"
                    val_border_style = "yellow"
                    
                    # Force an immediate update so the user sees "Running..." right now
                    live.update(Group(
                        metrics_panel, 
                        Panel(val_message, title="Validation Status", border_style=val_border_style, width=60), 
                        progress_bar
                    ))
                
                # 2. Run Evaluation
                val_loss = evaluate(model, val_loader)
                val_ppl = math.exp(val_loss) if val_loss < 20 else float("inf")
                last_eval_step = global_step
                
                accelerator.log({"val_loss": val_loss, "val_ppl": val_ppl}, step=global_step)

                # 3. Update State to "Result"
                # The next loop iteration will pick this up automatically, keeping the UI persistent
                if accelerator.is_main_process:
                    val_message = f"[bold cyan]Last Eval (Step {last_eval_step}):[/bold cyan]\nLoss: {val_loss:.4f} | PPL: {val_ppl:.2f}"
                    val_border_style = "cyan"        
                    

            # --- D. SAVE TRIGGER ---
            if global_step % args.save_steps == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    system_message = f"[bold yellow]Saving checkpoint to step_{global_step}...[/bold yellow]"                    
                    live.update(Group(metrics_panel, val_panel, Panel(system_message, border_style="yellow"), progress_bar))
                    
                accelerator.save_state(f"{args.output_dir}/step_{global_step}")
                
                if accelerator.is_main_process:
                    system_message = None
                    
                accelerator.wait_for_everyone()

    # Final Save (End of Loop)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        # Update UI to show we are in final phase
        console.print("[bold yellow]Training Complete. Running Final Evaluation...[/bold yellow]")
        
    # --- Final Eval on Train Split ---
    # We use a larger batch count (e.g., 100) for a more stable final number
    final_train_loss = evaluate(model, train_loader, max_eval_batches=100)
    final_train_ppl = math.exp(final_train_loss) if final_train_loss < 20 else float("inf")

    # --- Final Eval on Validation Split ---
    final_val_loss = evaluate(model, val_loader, max_eval_batches=100)
    final_val_ppl = math.exp(final_val_loss) if final_val_loss < 20 else float("inf")
    
    if accelerator.is_main_process:
        live.stop() # Stop the live display to print the final static summary
        
        # 1. Final Model Saving
        console.print("[bold blue]Saving final model weights...[/bold blue]")
        unwrapped_model = accelerator.unwrap_model(model)
        
        final_save_path = "./holo_final_model"
        
        unwrapped_model.save_pretrained(
            final_save_path, 
            is_main_process=accelerator.is_main_process, 
            save_function=accelerator.save,
            safe_serialization=False
        )
        
        tokenizer.save_pretrained(final_save_path)
        
        # 2. Final Results Summary Table
        from rich.table import Table
        from rich.box import DOUBLE_EDGE
        summary_table = Table(title="[bold green]Final Training Summary[/bold green]", box = DOUBLE_EDGE)
        summary_table.add_column("Split", style="cyan")
        summary_table.add_column("Loss", style="magenta")
        summary_table.add_column("Perplexity", style="magenta")

        summary_table.add_row("Training", f"{final_train_loss:.4f}", f"{final_train_ppl:.2f}")
        summary_table.add_row("Validation", f"{final_val_loss:.4f}", f"{final_val_ppl:.2f}")

        console.print(summary_table)
        console.print(f"[bold green]✅ Model and Tokenizer saved to: {final_save_path}[/bold green]")
        
finally:
    if accelerator.is_main_process:
        try:
            live.stop()
        except:
            pass
    accelerator.end_training()
    if accelerator.is_main_process:
        console.print("[bold green]Script Finished![/bold green]")