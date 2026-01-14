import argparse
import json
import torch
import lm_eval
from lm_eval.models.huggingface import HFLM
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# --- IMPORT YOUR UNIFIED LOADER ---
from model.model_loader import load_model_from_path

def parse_args():
    parser = argparse.ArgumentParser(description="Run Downstream Evaluation Tasks")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model or HF ID")
    parser.add_argument("--model_type", type=str, default="holo", choices=["holo", "gpt2", "mamba", "mamba2"], help="Type of model to load")
    parser.add_argument("--tasks", type=str, default="lambada_standard,lambada_openai,wikitext", help="Comma-separated tasks")
    parser.add_argument("--batch_size", type=str, default="auto", help="Batch size (e.g. '8' or 'auto')")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_file", type=str, default="benchmark_results.json")
    return parser.parse_args()

def main():
    args = parse_args()
    console = Console()
    task_list = [t.strip() for t in args.tasks.split(",")]

    console.print(Panel(f"Model: {args.model_path}\nType: {args.model_type}\nDevice: {args.device}", title="Starting Evaluation", border_style="green"))

    if args.device == "cpu" and args.model_type == "holo":
        console.print("[bold red]CRITICAL WARNING: Triton kernels (Holo) CANNOT run on CPU. You must use a GPU.[/bold red]")
        return

    # 1. LOAD MODEL (Using Unified Loader)
    try:
        console.print(f"[bold cyan]Loading {args.model_type.upper()} model...[/bold cyan]")
        
        # This handles Holo, GPT-2, and Mamba automatically
        model_obj, tokenizer = load_model_from_path(
            model_type=args.model_type,
            model_path=args.model_path,
            device=args.device
        )
        
        model_obj.eval() # Ensure evaluation mode

        # 2. WRAP FOR LM-EVAL
        # HFLM requires a standard HF-style model and tokenizer
        lm_obj = HFLM(
            pretrained=model_obj,
            tokenizer=tokenizer,
            batch_size=int(args.batch_size) if args.batch_size != "auto" else "auto",
            device=args.device 
        )
        
    except Exception as e:
        console.print(f"[bold red]Failed to load model:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. RUN EVALUATION
    try:
        console.print("[dim]Starting lm-eval tasks...[/dim]")
        results = lm_eval.simple_evaluate(
            model=lm_obj, 
            tasks=task_list,
            device=args.device
        )

    except Exception as e:
        console.print(f"[bold red]Error during evaluation:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. FORMAT OUTPUT
    if results is not None:
        table = Table(title=f"Results: {args.model_path}")
        table.add_column("Task", style="cyan")
        table.add_column("Metric", style="magenta")
        table.add_column("Value", style="green", justify="right")
        table.add_column("Stderr", style="dim", justify="right")

        raw_results = results["results"]
        desired_metrics = ["acc", "acc_norm", "perplexity"]
        
        for task_name, metrics in raw_results.items():
            for metric_key, value in metrics.items():
                # Filter for useful metrics to display in table
                if any(x in metric_key for x in desired_metrics) and "stderr" not in metric_key and "alias" not in metric_key:
                    stderr_key = metric_key + "_stderr"
                    stderr_val = metrics.get(stderr_key, "N/A")
                    
                    val_str = f"{value:.4f}" if isinstance(value, float) else str(value)
                    err_str = f"± {stderr_val:.4f}" if isinstance(stderr_val, float) else ""
                    
                    table.add_row(task_name, metric_key, val_str, err_str)

        console.print("\n")
        console.print(table)
        
        # Save JSON
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        console.print(f"[dim]Full results saved to {args.output_file}[/dim]")

if __name__ == "__main__":
    main()