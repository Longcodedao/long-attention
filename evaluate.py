import sys
import os
import math
import torch
import argparse
import warnings
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
from model.configuration_holo import HoloConfig
from model.modeling_holo import HoloForCausalLM
from dataset import data_loader
import utils
import datasets
import transformers

def parse_args():
    parser = argparse.ArgumentParser(description="Holo Evaluation Script")
    parser.add_argument("--model_path", type=str, default="./holo_final_model", 
                        help="Path to the directory containing the saved model and tokenizer")
    parser.add_argument("--dataset", type=str, default="slimpajama_6b", 
                        help="Dataset name or path to local test file")
    parser.add_argument("--split", type=str, default="test", 
                        help="Dataset split to evaluate on (usually 'test')")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--seq_len", type=int, default=2048, help="Sequence length")
    parser.add_argument("--max_eval_batches", type=int, default=100, 
                        help="Cap evaluation at X batches (set to 0 for full dataset)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize Accelerator
    accelerator = Accelerator(mixed_precision="bf16")
    set_seed(42)
    
    # Initialize Rich Console (only on main process to prevent messy overlapping)
    console = Console() if accelerator.is_main_process else None

    # 1. Load Model
    config = HoloConfig.from_pretrained(args.model_path)
    model = HoloForCausalLM.from_pretrained(args.model_path, config=config)
    
    # 2. Load Data (Using your robust loader)
    test_loader, tokenizer = data_loader.get_dataloader(
        console if accelerator.is_main_process else None, 
        accelerator, 
        args.dataset, 
        args.batch_size, 
        args.seq_len,
        split=args.split, 
        num_workers=0, 
        prefetch_factor=None
    )

    # 3. PREPARE MODEL ONLY
    # CRITICAL FIX: We do NOT prepare the dataloader. 
    # We will manually move data to GPU to avoid 'accelerate' messing up the sharding logic.
    model = accelerator.prepare(model)
    model.eval()

    loss_metric = MeanMetric().to(accelerator.device)

    # Determine Total Steps
    if args.max_eval_batches > 0:
        total_steps = args.max_eval_batches
    else:
        try:
            total_steps = len(test_loader)
        except:
            total_steps = 100 # Fallback if length is unknown

    accelerator.wait_for_everyone()

    # --- UI SETUP (Main Process Only) ---
    live_display = None
    progress_bar = None
    task_id = None

    if accelerator.is_main_process:
        # Define the Progress Bar
        progress_bar = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        )
        task_id = progress_bar.add_task("[cyan]Evaluating...", total=total_steps)
        
        # Define the Initial Panel
        status_panel = Panel(
            "Starting Evaluation...", 
            title="[bold green]Holo Eval[/bold green]", 
            border_style="green"
        )
        
        # Create the Layout Group
        ui_group = Group(status_panel, progress_bar)
        
        # Start Live Display
        live_display = Live(ui_group, console=console, refresh_per_second=5)
        live_display.start()

    # 4. Evaluation Loop
    # We iterate on the raw loader. Since it wasn't prepared, tensors are on CPU.
    data_iter = iter(test_loader)

    try:
        with torch.no_grad():
            for i in range(total_steps):
                # Manual Batch Extraction
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break
                
                # --- MANUAL MOVE TO DEVICE (Fixes Deadlock) ---
                batch = {k: v.to(accelerator.device) for k, v in batch.items()}

                # Forward pass
                outputs = model(input_ids=batch["input_ids"], labels=batch["input_ids"])
                loss = outputs.loss

                # GATHER: Syncs all GPUs just to collect the loss
                # repeated to match batch size for accurate weighting
                gathered_losses = accelerator.gather(loss.repeat(args.batch_size))
                
                # --- FAST UPDATE (No Sync) ---
                # We calculate the mean for THIS batch purely for the UI.
                # .item() pulls the scalar to CPU without triggering a full sync barrier.
                batch_loss_scalar = gathered_losses.mean().item()
                
                # Update the global metric (stores data, does not compute yet)
                loss_metric.update(gathered_losses)
                
                # --- UI UPDATE (Main Process Only) ---
                if accelerator.is_main_process:
                    # Calculate PPL for this specific batch
                    batch_ppl = math.exp(batch_loss_scalar) if batch_loss_scalar < 20 else float("inf")
                    
                    # Update Panel Text
                    msg = (
                        f"[bold white]Batch:[/bold white] {i+1}/{total_steps}\n"
                        f"[bold yellow]Current Batch Loss:[/bold yellow] {batch_loss_scalar:.4f}\n"
                        f"[bold yellow]Current Batch PPL:[/bold yellow]  {batch_ppl:.2f}"
                    )
                    new_panel = Panel(msg, title="[bold green]Holo Eval[/bold green]", border_style="green")
                    
                    # Update Live Display
                    live_display.update(Group(new_panel, progress_bar))
                    progress_bar.advance(task_id)

    except Exception as e:
        # Print error cleanly
        if accelerator.is_local_main_process:
            if live_display: live_display.stop()
            console.print(f"[bold red]Error during evaluation:[/bold red] {e}")
            import traceback
            traceback.print_exc()
        
    finally:
        # Stop UI
        if accelerator.is_main_process and live_display:
            live_display.stop()

        # Final Barrier
        accelerator.wait_for_everyone()
        
        # 5. Final Results
        # NOW it is safe to compute the global metric because the loop is done
        try:
            final_loss = loss_metric.compute().item()
            final_ppl = math.exp(final_loss) if final_loss < 20 else float("inf")

            if accelerator.is_main_process:
                results_table = Table(title="Final Test Results", show_header=True, header_style="bold green")
                results_table.add_column("Metric", style="cyan")
                results_table.add_column("Value", style="yellow")
                results_table.add_row("Dataset", args.dataset)
                results_table.add_row("Split", args.split)
                results_table.add_row("Sequence Length", str(args.seq_len))
                results_table.add_row("Final Loss", f"{final_loss:.4f}")
                results_table.add_row("Final Perplexity", f"{final_ppl:.2f}")
                console.print(results_table)
        except Exception as e:
            if accelerator.is_main_process:
                console.print(f"[red]Could not compute final metrics: {e}[/red]")

if __name__ == "__main__":
    main()