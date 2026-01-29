# LongAttention: A Holographic Linear-Time Mechanism for Infinite Context

**Status:** Pre-Alpha / Internal Validation
**Target:** ICML Submission
**Core Claim:** Infinite Context with Exact Associative Recall via Complex-Valued Holographic Memory.

---

## 🚀 Architectures: 



---

## 🛠️ Usage

### Quick Start
1. Install libraries
```bash
uv sync
source .venv/bin/activate
```

2. Creating the model
```python
from model.long import LongConfig, LongForCausalLM

# 1. Initialize for Usage (BF16)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = LongConfig(
   vocab_size=50304,
   hidden_size=768,
   num_hidden_layers = 12,    # Expansion factor 2x for capacity
   num_heads=12,
   expansion_ratio = 8/3,
   hybrid_ratio = 4,
   gate_init_bias = 0.0
)

model = LongForCausalLM(config).to(device)

# 2. You can use the default small setting of us
from model import model_loader

model, tokenizer = model_loader.get_model_and_tokenizer(
   model_type = "long",
   model_size = "small",
   vocab_size = 50257,     # Default for GPT-2 Token
   seq_len = 2048,         # User Input
   device = "cuda"
)
```

### Training on WikiText-103 Dataset
We experiment 1 GPU with this training setting:
- Model Type: Small (184M parameters)
- Sequence Length: 1024
- Batch Size: 4
- Gradient Accumulation Steps 4
- Training Steps: 6500

```bash
accelerate launch train_wikitext103.py \
	--model_type long \
	--model_size small \
	--seq_len 1024 \
	--batch_size 4 \
	--grad_accum_steps 4 \
	--max_steps 6500 \
	--lr 7e-4 \
	--save_steps 1000 \
	--eval_steps 50 \
	--warmup_steps 500 \
	--checkpoint_dir ./checkpoints/wikitext-long-small \
    --output_dir ./output/wikitext-long-small \
```

### Training on PG19 Dataset
We train 2 GPUS of RTX 4090 with this training setting:
- Model Type: Small (184M parameters)
- Sequence Length: 4096
- Batch Size: 2
- Gradient Accumulation Steps: 8
- Training Steps: 20000

```bash
   accelerate launch train_pg19.py \
	--model_type long \
	--model_size small \
	--seq_len 4096 \
	--batch_size 2 \
	--grad_accum_steps 8 \
	--max_steps 20000 \
	--lr 2e-4 \
	--save_steps 1000 \
	--eval_steps 250 \
	--checkpoint_dir ./checkpoints/pg19-long \
        --output_dir ./output/pg19-long \
	--gradient_checkpointing
```

### Experimental Results
1. WikiText-103
   

| Metric | Split | Long-Model | Mamba-2 |
| :--- | :--- | :--- | :--- |
| **PPL** | Train | 29.80 | 46.10 |
| | Validation | 29.76 | 44.69 |
| | Test | **29.24** | **44.10** |
| **Loss** | Train | 3.3944 | 3.8307 |
| | Validation | 3.3930 | 3.7997 |
| | Test | 3.3754 | 3.7865 |

2. PG19

| Metric | Split | Long-Model | Mamba-2 | GPT-2  |
| :--- | :--- | :--- | :--- | :--- |
| **Loss** | Validation | 3.3858 | 3.4762 | 3.4611 |
| | Test | 3.3425 | 3.4386 | 3.4261 |
| **PPL** | Validation | 29.54 | 32.34 | 31.85 |
| | Test | 28.29 | 31.14 | 30.76 |


