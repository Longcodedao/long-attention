import argparse
import json
import torch
import lm_eval
from lm_eval.models.huggingface import HFLM
from transformers import AutoTokenizer

# --- IMPORT YOUR CUSTOM MODEL ---
try:
    from model.configuration_holo import HoloConfig
    from model.modeling_holo import HoloForCausalLM
    HOLO_AVAILABLE = True
except ImportError:
    print("Warning: Could not import Holo classes. 'holo' models will fail to load.")
    HOLO_AVAILABLE = False

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

def parse_args():
    parser = argparse.ArgumentParser(description="Run Downstream Evaluation Tasks")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model or HF ID")
    parser.add_argument("--model_type", type=str, default="auto", choices=["auto", "holo"], help="Use 'holo' for your custom model")
    parser.add_argument("--tasks", type=str, default="hellaswag,piqa,arc_easy,lambada_openai", help="Comma-separated tasks")
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

    # 1. LOAD MODEL
    if args.model_type == "holo":
        if not HOLO_AVAILABLE:
            console.print("[bold red]Error:[/bold red] You selected --model_type holo but the classes could not be imported.")
            return

        console.print("[bold cyan]Loading custom Holo model...[/bold cyan]")
        
        # Load Config & Model
        config = HoloConfig.from_pretrained(args.model_path)
        model_obj = HoloForCausalLM.from_pretrained(args.model_path, config=config)
        
        # --- CRITICAL FIX: MOVE MODEL TO GPU EXPLICITLY ---
        console.print(f"[dim]Moving model to {args.device}...[/dim]")
        model_obj.to(args.device)
        model_obj.eval() # Set to evaluation mode
        # --------------------------------------------------

        # Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Wrap it for lm-eval
        lm_obj = HFLM(
            pretrained=model_obj,
            tokenizer=tokenizer,
            batch_size=int(args.batch_size) if args.batch_size != "auto" else "auto",
            device=args.device 
        )
        
    else:
        # Standard loading for baselines
        console.print("[bold cyan]Loading standard HF model...[/bold cyan]")
        lm_obj = "hf" 

    # 2. RUN EVALUATION
    try:
        if args.model_type == "holo":
             results = lm_eval.simple_evaluate(
                model=lm_obj, 
                tasks=task_list,
                device=args.device
            )
        else:
            results = lm_eval.simple_evaluate(
                model="hf",
                model_args=f"pretrained={args.model_path},trust_remote_code=True,dtype=bfloat16",
                tasks=task_list,
                batch_size=args.batch_size,
                device=args.device
            )

    except Exception as e:
        console.print(f"[bold red]Error during evaluation:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. FORMAT OUTPUT
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
                if any(x in metric_key for x in desired_metrics) and "stderr" not in metric_key and "alias" not in metric_key:
                    stderr_key = metric_key + "_stderr"
                    stderr_val = metrics.get(stderr_key, "N/A")
                    val_str = f"{value:.4f}" if isinstance(value, float) else str(value)
                    err_str = f"± {stderr_val:.4f}" if isinstance(stderr_val, float) else ""
                    table.add_row(task_name, metric_key, val_str, err_str)

        console.print("\n")
        console.print(table)
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

if __name__ == "__main__":
    main()