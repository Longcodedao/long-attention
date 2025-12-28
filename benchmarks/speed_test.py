import torch
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoConfig, AutoModelForCausalLM
from long_attention import HoloConfig, HoloForCausalLM

# --- Configuration ---
BATCH_SIZE = 1
D_MODEL = 1024
LAYERS = 12
HD_DIM = 2048  # will change
# Sequence lengths to test (Powers of 2)
SEQ_LENS = [2048, 4096, 8192, 16384, 32768, 65536] 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_peak_memory_mb():
    return torch.cuda.max_memory_allocated() / 1024 / 1024

def benchmark_model(model_type, seq_len):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # 1. Setup Model
    if model_type == "FlashAttention-2":
        config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
        config.hidden_size = D_MODEL
        config.num_hidden_layers = LAYERS
        config.num_attention_heads = 16 
        config.vocab_size = 32000
        # Force Flash Attention 2
        try:
            model = AutoModelForCausalLM.from_config(
                config, 
                attn_implementation="flash_attention_2", 
                torch_dtype=torch.bfloat16
            ).to(device)
        except ImportError:
            print("FlashAttention-2 not found, falling back to SDPA...")
            model = AutoModelForCausalLM.from_config(
                config, 
                torch_dtype=torch.bfloat16
            ).to(device)
            
    elif model_type == "Holo-Transformer":
        config = HoloConfig(
            hidden_size=D_MODEL,
            hd_dim=HD_DIM,
            num_hidden_layers=LAYERS,
            vocab_size=32000
        )
        model = HoloForCausalLM(config).to(device).bfloat16()

    # 2. Setup Data
    input_ids = torch.randint(0, 32000, (BATCH_SIZE, seq_len)).to(device)
    
    # --- A. Training Test (Forward + Backward) ---
    try:
        # Warmup
        _ = model(input_ids)
        torch.cuda.synchronize()
        
        # Timing
        start_time = time.time()
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        torch.cuda.synchronize()
        end_time = time.time()
        
        train_mem = get_peak_memory_mb()
        throughput = (seq_len * BATCH_SIZE) / (end_time - start_time)
        
    except torch.cuda.OutOfMemoryError:
        train_mem = "OOM"
        throughput = 0.0
    
    # --- B. Inference Test (KV Cache Growth) ---
    # We simulate generation memory by checking the cache size at step T
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    try:
        with torch.no_grad():
            if model_type == "Holo-Transformer":
                # Holo just runs forward, state is implicit/fixed
                _ = model(input_ids)
            else:
                # Transformer must init KV cache
                # We simulate a full context forward pass which fills the cache
                _ = model(input_ids, use_cache=True)
                
        infer_mem = get_peak_memory_mb()
        
    except torch.cuda.OutOfMemoryError:
        infer_mem = "OOM"

    del model
    return train_mem, throughput, infer_mem

# --- Main Loop ---
results = []
print(f"{'Model':<20} | {'SeqLen':<8} | {'Train Mem (MB)':<15} | {'Speed (Tok/s)':<15} | {'Infer Mem (MB)':<15}")
print("-" * 85)

for seq_len in SEQ_LENS:
    for model_name in ["Holo-Transformer", "FlashAttention-2"]:
        t_mem, speed, i_mem = benchmark_model(model_name, seq_len)
        
        print(f"{model_name:<20} | {seq_len:<8} | {str(t_mem):<15} | {speed:<15.1f} | {str(i_mem):<15}")
        
        # Store for Plotting
        if t_mem != "OOM":
            results.append({
                "Model": model_name,
                "SeqLen": seq_len,
                "Train Memory (MB)": t_mem,
                "Throughput": speed,
                "Inference Memory (MB)": i_mem if i_mem != "OOM" else None
            })

# --- Plotting (Saves 'benchmark_results.png') ---
df = pd.DataFrame(results)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Inference Memory (The "Flat Line" Win)
sns.lineplot(data=df, x="SeqLen", y="Inference Memory (MB)", hue="Model", marker="o", ax=axes[0])
axes[0].set_title("Inference Memory Scaling (Lower is Better)")
axes[0].set_ylabel("VRAM (MB)")
axes[0].set_xscale("log")

# Plot 2: Training Speed
sns.lineplot(data=df, x="SeqLen", y="Throughput", hue="Model", marker="o", ax=axes[1])
axes[1].set_title("Training Throughput (Higher is Better)")
axes[1].set_ylabel("Tokens / Sec")
axes[1].set_xscale("log")

# Plot 3: Training Memory
sns.lineplot(data=df, x="SeqLen", y="Train Memory (MB)", hue="Model", marker="o", ax=axes[2])
axes[2].set_title("Training Memory (Activations)")
axes[2].set_xscale("log")

plt.tight_layout()
plt.savefig("benchmark_results.png")
print("\nPlot saved to benchmark_results.png")
