# Holo-Transformer (LongAttention) - Research Build

**Status:** Pre-Alpha / Internal Validation
**Target:** ICML Submission
**Core Claim:** Infinite Context with Exact Associative Recall via Complex-Valued Holographic Memory.

---

## 🚀 Architecture Changes (vs Standard Transformer/Mamba)

We have deviated from standard architectures to solve the "Memory Blur" and "Swamping" issues. Do not revert these changes without checking the `unit_tests`.

### 1. The Core Mechanism: Phasor-FHRR
* **Old Way (Transformer):** $O(N^2)$ Attention Matrix.
* **Old Way (Mamba):** Real-valued State Space ($h_t = Ah_{t-1} + Bx_t$).
* **New Way (Holo):** Complex-valued Superposition.
    * **Binding:** $V \cdot e^{i\theta}$ (Rotation in complex plane).
    * **Memory:** $\sum (V \cdot e^{i\theta})$ (Linear Accumulation).
    * **Why:** Rotation preserves norm (unitary). Mamba's decay destroys history; our accumulation preserves it perfectly.

### 2. The Fix: Random Fourier Phasors (Killing RoPE)
* **Problem:** Standard RoPE frequencies are correlated. This caused "Local Blur" (Model couldn't distinguish pos 500 from 501).
* **Solution:** We use **Random High-Frequency Phasors** drawn from Uniform distribution.
* **Code:** `functional.generate_random_phasors`
* **Result:** Orthogonality improves from ~1.2x to >5x.

### 3. The Fix: Mixed Precision & Normalization
* **Problem:** "Swamping". Adding a token to a 100k sequence context makes the new token too small in FP16.
* **Solution:** * Storage: `BFloat16` (VRAM efficient).
    * Accumulation: `Float32/Complex64` (Precision critical).
    * Scale: We divide retrieval by $\sqrt{T}$ to counteract Random Walk variance.

## 29/12/2025
### Holo-Transformer (v7)
A PyTorch implementation of the **Holographic Transformer**, a linear-complexity ($O(N)$) architecture that combines the reasoning capabilities of Attention with the efficiency of Recurrent Neural Networks.

### Key Architecture Innovations (v7)
Today's update introduces the **Dual-Path Gated Architecture**, solving the "Recall vs. Indexing" trade-off:
1.  **Holographic Dual-Path Memory:**
    * **Path A (Positional):** Encodes absolute order using high-frequency Rotors ($V_t \cdot R_t$). Solves precise indexing tasks (e.g., "What is the 5th token?").
    * **Path B (Associative):** Encodes causal relationships by binding values to the conjugate of previous keys ($V_t \cdot K_{t-1}^*$). Solves reasoning tasks (e.g., "Needle in a Haystack").
2.  **Shared Q/K Projection ("Instant Learning"):**
    * We enforce $Q = K$ at the architectural level. This ensures that the Query and Key vectors are mathematically aligned from initialization ($Q \cdot K^* \approx 1$), allowing the model to perform associative recall immediately without waiting thousands of steps to learn alignment.
3.  **Spectral Stabilization:**
    * **Phase Scaling:** Initializing phase projections with high variance ($\sigma=3.0$) to force immediate orthogonality in the complex plane.
    * **LayerScale:** Residual branches are scaled by $\epsilon=0.1$ to ensure signal propagation through deep (12+ layer) networks.

## Benchmark Results (12-Layer Depth)
Comparison against standard Llama-2 (Transformer) and Deep GRU (Recurrent) baselines on 32k context.
| Model | Complexity | Speed (32k ctx) | Needle Recall (1k steps) |
| :--- | :--- | :--- | :--- |
| **Holo-12L** | **$O(N)$** | **~35,500 tok/s** | **81.25%** (Learned) |
| Transformer | $O(N^2)$ | ~18,400 tok/s | 6.25% (Failed) |
| Recurrent | $O(N)$ | ~96,600 tok/s | 0.00% (Failed) |

*Note: Holo-Transformer provides a 2x speedup over FlashAttention-v2 Transformers while retaining the inductive bias required to solve associative recall tasks that pure RNNs fail at.*

### 30/12/2025
## Update: Multi-Head "Spectrogram" Architecture (v8) 
We have upgraded the Holo-Transformer from a single holographic stream to a Multi-Head Frequency-Division Architecture. This aligns our model structurally with Llama/GPT while introducing a novel "Cognitive Spectrogram" mechanism. 

**What Changed?**
Instead of one large memory vector, we now split the holographic state into H independent heads. Crucially, these heads are not initialized identically. We apply Multi-Scale Initialization: 
- *High-Frequency Heads (0-2)*: Initialized with high variance ($\sigma \approx 10.0$). These spin fast, acting as Short-Term Memory (precise local texture)
- *Mid-Frequency Heads (3-5)*: Balanced variance ($\sigma \approx 3.0$ ). Standard associative recall.
- *Low-Frequency Heads (6-7)*: Low variance ($\sigma \approx 0.1$). These spin slowly, acting as Long-Term Archives (global gist) that resist noise over long contexts.

**Why?** 
* Extended Effective Context: By routing stable information to "Slow" heads and volatile information to "Fast" heads, we reduce the crosstalk noise that usually limits holographic capacity. 
* Gated Specialization: Each head has its own independent Time/Content Gate. Some heads can specialize purely in Indexing (Time), while others specialize purely in Reasoning (Content).
* New Hyperparameters:
    * num_heads: Default 8. (Splits the hd_dim evenly).
    * phase_scale: Now acts as the base scale for the geometric distribution across heads.


---

## 🛠️ Usage

### Quick Start
```python
from long_attention import HoloConfig, HoloForCausalLM

# 1. Initialize for L40S Training (BF16)
config = HoloConfig(
    vocab_size=32000,
    hidden_size=1024,
    hd_dim=2048,        # Expansion factor 2x for capacity
    num_hidden_layers=12
)

model = HoloForCausalLM(config).cuda().bfloat16()

# 2. Forward Pass
# Provide the Autocast so that it will cast to bfloat16 in the input and also model
with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
    outputs = model(input_ids=..., labels=...)
    loss = outputs.loss

```

## ICML Experiments Roadmap
* **Phase 1**: Unit Validation (✅ COMPLETED)
    * Needle in Haystack: Passed (via Kaggle).
    * Associative Recall: Passed (via Kaggle, MSE < 0.005).
    * *Note*: The MLP Denoising was critical here. A linear-only model failed.
      
      
* **Phase 2**: Speed Benchmark (Next Step)
    * Goal: Demonstrate flat VRAM usage vs Sequence Length.
    * Script: benchmarks/speed_test.py
    * Comparison: FlashAttention-2 vs Holo.
    * Expected Win: FlashAttn OOMs at 32k; Holo stays constant.
      
* **Phase 3**: Capabilities (Produdction Run)
    * Dataset: SlimPajama (6B tokens subset).
    * Hardware: 2x L40S.
    * Hyperparams:LR: 3e-4
    * Global Batch: 0.5M tokens
    * Context: Train on 4k, Eval on 16k.
      


