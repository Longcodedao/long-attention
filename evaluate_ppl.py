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
    parser = argparse.ArgumentParser(description="Universal Model Evaluation Script")
    
    # Model Selection
    parser.add_argument("--model_type", type=str, default="long", 
                        choices=["holo", "long", "gpt2", "mamba", "mamba2"], 
                        help="Type of model to evaluate")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the directory containing the saved model and tokenizer")
    
    # Dataset Selection
    parser.add_argument("--dataset", type=str, default="wikitext", 
                        help="Name of dataset: slimpajama_6b, slimpajama_627b, wikitext, pg19, etc.")
    parser.add_argument("--split", type=str, default="validation", 
                        help="Split to evaluate on (validation, test)")
    
    # Eval Parameters
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--seq_len", type=int, default=1024, help="Sequence length (must match training)")
    parser.add_argument("--max_eval_batches", type=int, default=200, 
                        help="Maximum batches to evaluate. CRITICAL for streaming large datasets like SlimPajama.")
    
    return parser.parse_args()

# ===============================
# 2. Main Execution
# ===============================
def main():
    # We need to disable DeepSpeed to Evaluate the model comfortably
    os.environ["ACCELERATE_USE_DEEPSPEED"] = "false"
        
    args = parse_args()
    
    # Initialize Accelerator
    accelerator = Accelerator(mixed_precision="bf16") 
    set_seed(42)
    
    # Setup Console (Main Process Only)
    console = Console() if accelerator.is_main_process else None
    
    if accelerator.is_main_process:
        console.rule(f"[bold green]Evaluation: {args.model_type.upper()} on {args.dataset}[/bold green]")
        utils.print_config_table(console, accelerator, args)

    # --- Load Model & Tokenizer ---
    try:
        model, tokenizer = model_loader.load_model_from_path(
            model_type=args.model_type, 
            model_path=args.model_path, 
            device=accelerator.device
        )
    except Exception as e:
        if accelerator.is_main_process:
            console.print(f"[bold red]Failed to load model from {args.model_path}: {e}[/bold red]")
        return

    # --- Load Data (Universal) ---
    # We use the same data_loader.get_dataloader logic as training to ensure
    # consistent tokenization and packing.
    try:
        # Determine strict limit for massive datasets
        is_massive = "slimpajama" in args.dataset.lower() or "pg19" in args.dataset.lower()
        if is_massive and args.max_eval_batches <= 0:
            if accelerator.is_main_process:
                console.print("[bold red]WARNING: Evaluating massive dataset without a batch limit![/bold red]")
                console.print("[yellow]Setting default limit of 500 batches to prevent infinite loop.[/yellow]")
            args.max_eval_batches = 500

        test_loader = data_loader.get_dataloader(
            console=console if accelerator.is_main_process else None,
            accelerator=accelerator,
            tokenizer=tokenizer,
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            split=args.split,
            num_workers=2,
        )
    except ValueError as ve:
        if accelerator.is_main_process:
            console.print(f"[bold red]Dataset Error:[/bold red] {ve}")
            console.print(f"Ensure '{args.dataset}' is handled in your data_loader.load_data_source function.")
        return

    # --- Prepare for Accelerator ---
    model, test_loader = accelerator.prepare(model, test_loader)
    model.eval()

    loss_metric = MeanMetric().to(accelerator.device)

    # Determine Total Steps for Progress Bar
    # Since it's an iterable dataset, len() might not exist. We use the cap.
    total_steps = args.max_eval_batches if args.max_eval_batches > 0 else 1000 

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
            TextColumn("{task.fields[info]}")
        )
        task_id = progress_bar.add_task(
            f"[cyan]Evaluating {args.dataset}...", 
            total=total_steps, 
            info="init..."
        )
        
        status_panel = Panel(
            "Starting Evaluation...", 
            title=f"[bold green]{args.model_type.upper()} Eval[/bold green]", 
            border_style="green"
        )
        live_display = Live(Group(status_panel, progress_bar), console=console, refresh_per_second=5)
        live_display.start()

    # --- Evaluation Loop ---
    data_iter = iter(test_loader)
    step_count = 0
    
    try:
        with torch.no_grad():
            while step_count < args.max_eval_batches:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    if accelerator.is_main_process:
                        console.print("[yellow]Dataset exhausted before max_eval_batches reached.[/yellow]")
                    break
                
                # --- Forward Pass ---
                # Your data_loader already packs inputs and labels, so we just pass them.
                outputs = model(
                    input_ids=batch["input_ids"], 
                    labels=batch["labels"] # PackedDataset provides "labels"
                )
                loss = outputs.loss

                # Gather and Update
                gathered_losses = accelerator.gather(loss.repeat(args.batch_size))
                loss_metric.update(gathered_losses.mean())
                
                step_count += 1
                
                # --- UI Update ---
                if accelerator.is_main_process:
                    current_avg_loss = loss_metric.compute().item()
                    current_ppl = math.exp(min(current_avg_loss, 20))
                    
                    msg = (
                        f"[bold white]Batch:[/bold white] {step_count}/{args.max_eval_batches}\n"
                        f"[bold yellow]Running Loss:[/bold yellow] {current_avg_loss:.4f}\n"
                        f"[bold yellow]Running PPL:[/bold yellow]  {current_ppl:.2f}"
                    )
                    new_panel = Panel(msg, title=f"[bold green]{args.model_type.upper()} Eval[/bold green]", border_style="green")
                    
                    progress_bar.update(
                        task_id, 
                        completed=step_count, 
                        info=f"Loss: {current_avg_loss:.3f}"
                    )
                    live_display.update(Group(new_panel, progress_bar))

    except KeyboardInterrupt:
        if accelerator.is_main_process:
            console.print("[bold red]Evaluation Interrupted![/bold red]")
    except Exception as e:
        if accelerator.is_main_process:
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
                results_table.add_row("Dataset", args.dataset)
                results_table.add_row("Split", args.split)
                results_table.add_row("Batches Evaluated", str(step_count))
                results_table.add_row("Final Loss", f"{final_loss:.4f}")
                results_table.add_row("Final PPL", f"{final_ppl:.2f}")
                
                console.print(results_table)
                
                # Optional: Append to a CSV for tracking multiple experiments
                with open("eval_results.csv", "a") as f:
                    f.write(f"{args.model_type},{args.dataset},{final_loss:.4f},{final_ppl:.2f}\n")
                    
        except Exception as e:
            if accelerator.is_main_process:
                console.print(f"[red]Could not compute final metrics: {e}[/red]")

if __name__ == "__main__":
    main()