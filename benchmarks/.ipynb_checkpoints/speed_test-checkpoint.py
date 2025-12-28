import torch
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoConfig, AutoModelForCausalLM
from model import HoloConfig, HoloForCausalLM

# --- Configuration ---
BATCH_SIZE = 1
D_MODEL = 1024
LAYERS = 12
VOCAB_SIZE = 32000

# Sequence lengths to stress test (Powers of 2)
# We go up to 64k to force OOM on standard models
SEQ_LENS = [2048, 4096, 8192, 16384, 32768, 65536] 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running Speed Benchmark on: {torch.cuda.get_device_name(0)}")

def get_peak_memory_mb():
    return torch.cuda.max_memory_allocated() / 1024 / 1024

def benchmark_model(model_variant, seq_len):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    model = None
    config = None

    # --- 1. Instantiate Model Variants ---
    try:
        if model_variant == "FlashAttention-2":
            # Standard Llama-2 Architecture using FlashAttn 2 kernel
            config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
            config.hidden_size = D_MODEL
            config.num_hidden_layers = LAYERS
            config.num_attention_heads = 16 
            config.vocab_size = VOCAB_SIZE
            config.intermediate_size = D_MODEL * 4
            
            # Try loading with FA2, fallback if not available
            try:
                model = AutoModelForCausalLM.from_config(
                    config, 
                    attn_implementation="flash_attention_2", 
                    torch_dtype=torch.bfloat16
                ).to(device)
            except ImportError:
                print("Warning: FlashAttention-2 not found. Using SDPA/Eager (Slower).")
                model = AutoModelForCausalLM.from_config(
                    config, 
                    torch_dtype=torch.bfloat16
                ).to(device)

        elif model_variant == "Holo-2k":
            # Efficiency Config: Small state (2048)
            config = HoloConfig(
                hidden_size=D_MODEL,
                hd_dim=2048,  # Small State
                num_hidden_layers=LAYERS,
                vocab_size=VOCAB_SIZE
            )
            model = HoloForCausalLM(config).to(device).bfloat16()

        elif model_variant == "Holo-8k":
            # Capacity Config: Large state (8192) for Long Context
            config = HoloConfig(
                hidden_size=D_MODEL,
                hd_dim=8192,  # Large State (4x larger)
                num_hidden_layers=LAYERS,
                vocab_size=VOCAB_SIZE
            )
            model = HoloForCausalLM(config).to(device).bfloat16()
            
    except Exception as e:
        print(f"Error initializing {model_variant}: {e}")
        return "Error", 0.0, "Error"

    # --- 2. Setup Data ---
    # Random tokens
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, seq_len)).to(device)

    # --- 3. Training Benchmark (Throughput) ---
    train_mem = "OOM"
    throughput = 0.0
    
    try:
        # Warmup pass
        _ = model(input_ids)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        
        # Measured pass (Forward + Backward)
        start_time = time.time()
        
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        train_mem = get_peak_memory_mb()
        # Tokens per second (Forfward + Backward)
        throughput = (seq_len * BATCH_SIZE) / (end_time - start_time)
        
    except torch.cuda.OutOfMemoryError:
        train_mem = "OOM"
        throughput = 0.0
    except Exception as e:
        print(f"Train Error {model_variant} @ {seq_len}: {e}")

    # --- 4. Inference Benchmark (State/Cache Memory) ---
    # We simulate the memory cost of generating text (Filling the KV Cache)
    infer_mem = "OOM"
    
    if train_mem != "OOM": # Only run if it fits in memory roughly
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            with torch.no_grad():
                if "Holo" in model_variant:
                    # Holo has constant state, standard forward reveals peak
                    _ = model(input_ids)
                else:
                    # Transformers allocate KV Cache during generation/forward with cache
                    # We verify memory usage of a full context loaded into cache
                    _ = model(input_ids, use_cache=True)
            
            infer_mem = get_peak_memory_mb()
            
        except torch.cuda.OutOfMemoryError:
            infer_mem = "OOM"
        except Exception as e:
            print(f"Infer Error {model_variant} @ {seq_len}: {e}")

    # Cleanup
    del model, config, input_ids
    return train_mem, throughput, infer_mem

# --- Main Execution Loop ---
results = []
# models_to_test = ["Holo-2k", "Holo-8k", "FlashAttention-2"]
models_to_test = ["Holo-2k", "Holo-8k"]

print(f"\n{'Model':<18} | {'SeqLen':<8} | {'Train Mem':<12} | {'Speed (T/s)':<12} | {'Infer Mem':<12}")
print("-" * 75)

for seq_len in SEQ_LENS:
    for model_name in models_to_test:
        t_mem, speed, i_mem = benchmark_model(model_name, seq_len)
        
        # Formatting for print
        t_mem_str = f"{t_mem:.0f} MB" if isinstance(t_mem, float) else t_mem
        i_mem_str = f"{i_mem:.0f} MB" if isinstance(i_mem, float) else i_mem
        
        print(f"{model_name:<18} | {seq_len:<8} | {t_mem_str:<12} | {speed:<12.1f} | {i_mem_str:<12}")
        
        # Save valid results
        if isinstance(t_mem, float):
            results.append({
                "Model": model_name,
                "SeqLen": seq_len,
                "Train Memory (MB)": t_mem,
                "Throughput": speed,
                "Inference Memory (MB)": i_mem if isinstance(i_mem, float) else None
            })

# --- Plotting ---
if not results:
    print("No results collected (Everything OOM'd?). Check GPU.")
else:
    df = pd.DataFrame(results)
    
    # Set style
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Plot 1: Inference Memory (The Main Claim)
    # We expect Holo lines to be flat, FlashAttn to spike
    sns.lineplot(data=df, x="SeqLen", y="Inference Memory (MB)", hue="Model", style="Model", markers=True, ax=axes[0], linewidth=2.5)
    axes[0].set_title("Inference Memory (Cache/State)", fontsize=14, fontweight='bold')
    axes[0].set_ylabel("VRAM (MB)")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log") # Log scale helps see the massive gap

    # Plot 2: Training Throughput
    sns.lineplot(data=df, x="SeqLen", y="Throughput", hue="Model", style="Model", markers=True, ax=axes[1], linewidth=2.5)
    axes[1].set_title("Training Speed (Tokens/sec)", fontsize=14, fontweight='bold')
    axes[1].set_ylabel("Throughput")
    axes[1].set_xscale("log")

    # Plot 3: Training VRAM (Activations)
    sns.lineplot(data=df, x="SeqLen", y="Train Memory (MB)", hue="Model", style="Model", markers=True, ax=axes[2], linewidth=2.5)
    axes[2].set_title("Training VRAM (Activations)", fontsize=14, fontweight='bold')
    axes[2].set_xscale("log")

    plt.tight_layout()
    plt.savefig("benchmark_comparison_holo8k.png", dpi=300)
    print("\n✅ Benchmark Complete. Results saved to 'benchmark_comparison_holo8k.png'")
