import sys
import os
import math
import torch
import argparse
from accelerate import Accelerator
from accelerate.utils import set_seed
from torchmetrics import MeanMetric
from rich.live import Live
from rich.panel import Panel
from rich.console import Group
from rich.table import Table

# --- LOCAL IMPORTS ---
from model.configuration_holo import HoloConfig
from model.modeling_holo import HoloForCausalLM
from dataset import data_loader
import utils
import datasets
import transformers
import warnings

# Silence background noise
# datasets.disable_progress_bar()
# datasets.utils.logging.set_verbosity_error()
# transformers.utils.logging.set_verbosity_error()
# warnings.filterwarnings("ignore", message=".*barrier().*")

def parse_args():
    parser = argparse.ArgumentParser(description="Holo Evaluation Script")

    # Path to the saved model
    parser.add_argument("--model_path", type=str, default="./holo_final_model", 
                        help="Path to the directory containing the saved model and tokenizer")
    
    # Dataset 
    parser.add_argument("--dataset", type=str, default="slimpajama_6b", 
                        help="Dataset name or path to local test file")
    parser.add_argument("--split", type=str, default="test", 
                        help="Dataset split to evaluate on (usually 'test')")
    
    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--seq_len", type=int, default=2048, help="Sequence length")
    parser.add_argument("--max_eval_batches", type=int, default = 100, 
                        help="Cap evaluation at X batches (set to 0 for full dataset)")

    return parser.parse_args()

def main():
    args = parse_args()
    accelerator = Accelerator(mixed_precision="bf16")
    set_seed(42)
    console = utils.get_console(accelerator)

    # 1. Load Model and Tokenizer
    config = HoloConfig.from_pretrained(args.model_path)
    model = HoloForCausalLM.from_pretrained(args.model_path, config=config)
    
    # 2. Load Data
    test_loader, tokenizer = data_loader.get_dataloader(
        console, accelerator, args.dataset, args.batch_size, args.seq_len,
        split=args.split, num_workers=0, prefetch_factor = None
    )

    # 3. PREPARE (Now includes the loader for automatic distributed sharding)
    # This prevents ranks from getting out of sync with each other
    model, test_loader = accelerator.prepare(model, test_loader)
    model.eval()

    loss_metric = MeanMetric().to(accelerator.device)
    progress_bar = utils.create_progress_bar()

    if accelerator.is_main_process:
        # Determine total steps for the progress bar
        if args.max_eval_batches > 0:
            total_steps = args.max_eval_batches
        else:
            try:
                total_steps = len(test_loader)
            except:
                total_steps = None

        eval_task_id = progress_bar.add_task("[cyan]Evaluating...", total=total_steps)
        live = Live(console=console, refresh_per_second=4, redirect_stdout=True, redirect_stderr=True)
        live.start()

    # 4. Evaluation Loop
    try:
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                # Check for manual cap
                if args.max_eval_batches > 0 and i >= args.max_eval_batches:
                    break

                # Forward pass
                outputs = model(input_ids=batch["input_ids"], labels=batch["input_ids"])
                loss = outputs.loss

                # CRITICAL: gather_for_metrics handles the cross-GPU sync safely
                # It collects the loss from all GPUs and handles uneven batch sizes
                gathered_losses = accelerator.gather_for_metrics(loss)
                loss_metric.update(gathered_losses)
                
                # UI Update on Main Process
                if accelerator.is_main_process:
                    current_loss = loss_metric.compute().item()
                    current_ppl = math.exp(current_loss) if current_loss < 20 else float("inf")


                    status_msg = f"[bold cyan]Batch {i+1}[/bold cyan]\nLoss: {current_loss:.4f} | PPL: {current_ppl:.2f}"
                    live.update(Group(Panel(status_msg, title="Evaluating", border_style="cyan"), progress_bar))
                    progress_bar.advance(eval_task_id)

    except Exception as e:
        accelerator.print(f"[red]Error during evaluation: {e}[/red]")
        
    finally:
        # Final Sync
        accelerator.wait_for_everyone()
        
        # 5. Final Results
        final_loss = loss_metric.compute().item()
        final_ppl = math.exp(final_loss) if final_loss < 20 else float("inf")

        if accelerator.is_main_process:
            live.stop()
            results_table = Table(title="Final Test Results", show_header=True, header_style="bold green")
            results_table.add_column("Metric", style="cyan")
            results_table.add_column("Value", style="yellow")
            results_table.add_row("Dataset", args.dataset)
            results_table.add_row("Split", args.split)
            results_table.add_row("Sequence Length", str(args.seq_len))
            results_table.add_row("Final Loss", f"{final_loss:.4f}")
            results_table.add_row("Final Perplexity", f"{final_ppl:.2f}")
            console.print(results_table)

if __name__ == "__main__":
    main()