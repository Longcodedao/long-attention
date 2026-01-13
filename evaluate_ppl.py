import sys
import os
import math
import torch
import argparse
import warnings
import datasets
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from torchmetrics import MeanMetric

# --- RICH UI IMPORTS ---
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn
)

# --- LOCAL IMPORTS ---
# Ensure these match your file structure
from model import model_loader
from dataset import data_loader
import utils


# Disable noisy logging
datasets.utils.logging.set_verbosity_error()
transformers.utils.logging.set_verbosity_error()

# ===============================
# 1. Argument Parsing
# ===============================
def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Model Evaluation Script (Holo/GPT2/Mamba)")
    
    # Model Selection
    parser.add_argument("--model_type", type=str, default="holo", 
                        choices=["holo", "gpt2", "mamba"], 
                        help="Type of model to evaluate")
    parser.add_argument("--model_path", type=str, default="./output/holo-small", 
                        help="Path to the directory containing the saved model and tokenizer")
    
    # Dataset
    parser.add_argument("--dataset", type=str, default="slimpajama_6b", help="Dataset name")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate on")
    
    # Eval Parameters
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--seq_len", type=int, default=1024, help="Sequence length (must match training)")
    parser.add_argument("--max_eval_batches", type=int, default=100, 
                        help="Cap evaluation at X batches (set to 0 for full dataset)")
    
    return parser.parse_args()


# ===============================
# 3. Main Execution
# ===============================
def main():
    
    # We need to disable DeepSpeed to Evaluate the model
    os.environ["ACCELERATE_USE_DEEPSPEED"] = "false"
        
    args = parse_args()
    
    # Initialize Accelerator
    # Mamba/Holo often benefit from BF16. FP16 is standard for GPT-2.
    accelerator = Accelerator(mixed_precision="bf16") 
    set_seed(42)
    
    # Setup Console (Main Process Only)
    console = Console() if accelerator.is_main_process else None
    
    if accelerator.is_main_process:
        console.rule(f"[bold green]Evaluation: {args.model_type.upper()}[/bold green]")
        utils.print_config_table(console, accelerator, args)

    # --- Load Model, Tokenizer ---
    model, tokenizer = model_loader.load_model_from_path(
        model_type=args.model_type, 
        model_path=args.model_path, 
        device=accelerator.device
    )

    # --- Load Data ---
    test_loader = data_loader.get_dataloader(
        console if accelerator.is_main_process else None, 
        accelerator, 
        tokenizer, 
        args.dataset, 
        args.batch_size, 
        args.seq_len,
        split=args.split,
        num_workers=2 # slightly faster loading
    )

    # --- Prepare for Accelerator ---
    model, test_loader = accelerator.prepare(model, test_loader)
    model.eval()

    loss_metric = MeanMetric().to(accelerator.device)

    # Determine Steps
    if args.max_eval_batches > 0:
        total_steps = args.max_eval_batches
    else:
        total_steps = len(test_loader)

    accelerator.wait_for_everyone()

    # --- UI Setup ---
    live_display = None
    progress_bar = None
    task_id = None

    if accelerator.is_main_process:
        progress_bar = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        )
        task_id = progress_bar.add_task(f"[cyan]Evaluating {args.model_type}...", total=total_steps)
        
        status_panel = Panel(
            "Starting Evaluation...", 
            title=f"[bold green]{args.model_type.upper()} Eval[/bold green]", 
            border_style="green"
        )
        live_display = Live(Group(status_panel, progress_bar), console=console, refresh_per_second=5)
        live_display.start()

    # --- Evaluation Loop ---
    data_iter = iter(test_loader)
    
    try:
        with torch.no_grad():
            for i in range(total_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break
                
                # --- INPUT MASKING (Critical for GPT-2 / Mamba) ---
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                
                # Create labels: same as input, but set padding tokens to -100
                labels = input_ids.clone()
                labels[attention_mask == 0] = -100
                
                # Forward Pass
                outputs = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    labels=labels
                )
                loss = outputs.loss

                # Gather and Update
                gathered_losses = accelerator.gather(loss.repeat(args.batch_size))
                loss_metric.update(gathered_losses)
                
                # --- UI Update ---
                if accelerator.is_main_process:
                    batch_loss = gathered_losses.mean().item()
                    batch_ppl = math.exp(batch_loss) if batch_loss < 20 else float("inf")
                    
                    msg = (
                        f"[bold white]Batch:[/bold white] {i+1}/{total_steps}\n"
                        f"[bold yellow]Current Loss:[/bold yellow] {batch_loss:.4f}\n"
                        f"[bold yellow]Current PPL:[/bold yellow]  {batch_ppl:.2f}"
                    )
                    new_panel = Panel(msg, title=f"[bold green]{args.model_type.upper()} Eval[/bold green]", border_style="green")
                    live_display.update(Group(new_panel, progress_bar))
                    progress_bar.advance(task_id)

    except Exception as e:
        if accelerator.is_main_process:
            if live_display: live_display.stop()
            console.print(f"[bold red]Error during evaluation:[/bold red] {e}")
            import traceback
            traceback.print_exc()

    finally:
        if accelerator.is_main_process and live_display:
            live_display.stop()
        accelerator.wait_for_everyone()
        
        # --- Final Results ---
        try:
            final_loss = loss_metric.compute().item()
            final_ppl = math.exp(min(final_loss, 20)) # clip for overflow

            if accelerator.is_main_process:
                results_table = Table(title="Final Evaluation Results", show_header=True, header_style="bold green")
                results_table.add_column("Metric", style="cyan")
                results_table.add_column("Value", style="yellow")
                results_table.add_row("Model Type", args.model_type.upper())
                results_table.add_row("Model Path", args.model_path)
                results_table.add_row("Dataset", args.dataset)
                results_table.add_row("Final Loss", f"{final_loss:.4f}")
                results_table.add_row("Final PPL", f"{final_ppl:.2f}")
                console.print(results_table)
        except Exception as e:
            if accelerator.is_main_process:
                console.print(f"[red]Could not compute final metrics: {e}[/red]")

if __name__ == "__main__":
    main()