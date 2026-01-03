from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.box import DOUBLE_EDGE

def get_console(accelerator):
    """Returns a Rich Console only on the main process."""
    return Console() if accelerator.is_main_process else None

def create_progress_bar():
    """Defines the visual style of the progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=None),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        expand=True
    )

def create_metrics_table(step, max_steps, loss, ppl, lr):
    """Generates the metrics table for the dashboard."""
    table = Table(show_header=True, header_style="bold magenta", expand=True)
    table.add_column("Global Step", style="dim", width=12)
    table.add_column("Loss", justify="right")
    table.add_column("Perplexity", justify="right")
    table.add_column("LR", justify="right")
    
    loss_str = f"{loss:.4f}" if loss is not None else "..."
    ppl_str = f"{ppl:.2f}" if ppl is not None else "..."
    
    table.add_row(f"{step}/{max_steps}", loss_str, ppl_str, f"{lr:.2e}")
    return Panel(table, title="Holo Training Metrics", border_style="blue")

def print_config_table(console, accelerator, args):
    """Prints a noticeable configuration table at startup."""
    if not accelerator.is_main_process:
        return

    table = Table(
        title="[bold magenta]🚀 Holo Training Configuration[/bold magenta]", 
        box=DOUBLE_EDGE, 
        header_style="bold cyan",
        show_lines=True
    )
    
    table.add_column("Parameter", style="green", no_wrap=True)
    table.add_column("Value", style="yellow")

    # Hardware Info
    table.add_row("Num Processes (GPUs)", str(accelerator.num_processes))
    table.add_row("Mixed Precision", str(accelerator.mixed_precision))
    
    # Arguments
    for arg, value in sorted(vars(args).items()):
        table.add_row(arg, str(value))

    console.print(table)
    console.print("\n")