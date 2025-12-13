# TallyFormer-Finance-51M

**TallyFormer-Finance-51M** is a compact, efficient 51-million-parameter decoder-only transformer language model built entirely from scratch. It was trained in three progressive stages to achieve strong general language understanding followed by specialized performance in the **finance domain**.

The model excels at financial reasoning, stock analysis, investment explanations, and instruction-following tasks while remaining lightweight and fast for inference.

Fully containerized deployment: FastAPI backend + modern React chat frontend, orchestrated with Docker Compose.

## Key Specifications

- **Parameters**: ~51 million (~197 MB in bfloat16/float16)
- **Architecture**: Custom GPT-2-style transformer with Rotary Positional Embeddings (RoPE), pre-norm RMSNorm, GELU activations, weight-tied embeddings, z-loss regularization
- **Context length**: 256 tokens
- **Vocabulary**: 50,257 (GPT-2 base + `<|user|>`, `<|assistant|>`, pad token)
- **Specialization**: Finance (trained on Finance-Alpaca instructions)
- **Training hardware**: Single RTX A6000 (48 GB VRAM), 50 GB RAM, 8 vCPU, 40 GB disk (Pod.ai platform)

## Repository Structure
workflow/
├── Preprocess_Data/
│   └── Prepare_Data.ipynb         # Full data cleaning & preparation pipeline
├── Tallyformer.ipynb              # Complete training notebook (all 3 stages)
├── precomputed/                   # Memmapped tokenized datasets
├── PreTrainResult/                # Pretraining checkpoint
├── DistillationResult/            # Distilled checkpoints
└── SFT_LoRA/                      # LoRA adapter weights
tallyformer-system/                # Production deployment
├── tallyformer-api/               # FastAPI inference server
├── tallyformer-frontend/          # React chat UI
└── docker-compose.yml             # One-command deployment
TallyFormer-Finance-51M/           # Final merged model (HF-ready)
├── model.pth
├── tokenizer files
└── config
text## Training Pipeline & Results

All data preprocessing is performed in `workflow/Preprocess_Data/Prepare_Data.ipynb`.

### 1. Pretraining (Continual Pretraining from GPT-2 Weights)
- **Tokens seen**: ~1.3 billion
- **Data mix**: 60% Falcon-RefinedWeb, 15% PES2O, 25% SlimPajama
- **Tokenization**: Offline (memmapped binaries for maximum DataLoader speed)
- **Training details**:
  - 60 epochs
  - Effective batch size: 192 (32 × gradient accumulation 6)
  - Optimizer: AdamW (LR 8e-4, cosine decay after warmup)
  - Mixed precision (bfloat16 preferred)
  - Time per epoch: ~1.1 hours
- **Result**: Validation perplexity = **39.0**

Checkpoint: `PreTrainResult/pretrain_tallyformer.pth`

### 2. Knowledge Distillation
- **Tokens seen**: ~0.7 billion
- **Data mix**: 58% Falcon-RefinedWeb, 16% PES2O, 26% SlimPajama
- **Teacher model**: `gpt2-medium` (355M parameters)
- **Two-phase approach**:

  **Phase 1**
  - Temperature: 2.5
  - Alpha (KL weight): 0.7
  - Learning rate: 2.5e-4
  - Epochs: 19
  - Scheduler: OneCycleLR
  - Time per epoch: ~2.46 hours

  **Phase 2**
  - Temperature: 3.0
  - Alpha (KL weight): 0.4
  - Learning rate: 2e-4
  - Epochs: 11
  - Final validation perplexity: **38.0**

Final checkpoint: `DistillationResult/tallyformer-distilled-phase2.pth`

### 3. Supervised Fine-Tuning (SFT)
- **Dataset**: Finance-Alpaca (~28k high-quality financial instruction examples)
- **Method**: PEFT LoRA (r=16, α=32, dropout=0.05)
  - Target modules: query, key, value, proj, fc1, fc2
  - modules_to_save: lm_head
- **Formatting**: `<|user|> {prompt} <|assistant|> {response}`
  - Loss applied only to assistant response tokens
- **Training details**:
  - 3 epochs
  - Batch size 32 × gradient accumulation 8
  - Optimizer: AdamW (LR 2e-4)
  - Scheduler: Cosine annealing
  - Time per epoch: ~5 minutes
- **Result**: Validation perplexity on held-out QA data = **13.0**

LoRA adapter saved → merged into base distilled model → final clean checkpoint in `TallyFormer-Finance-51M/`

## Features

- High-performance custom generation: top-k, top-p, frequency/presence penalties, multi-token EOS support
- KV caching with automatic context trimming
- Flash Attention compatible
- Efficient mixed-precision inference
- Streaming-ready API endpoints

## Requirements

- Docker >= 20.x
- Docker Compose >= 2.0
- (Optional, training) Python 3.11+, PyTorch 2.x, CUDA GPU

## Quick Deployment

```bash
git clone https://github.com/YOUR_USERNAME/tallyformer.git
cd tallyformer/tallyformer-system
docker compose up -d --build
→ Chat interface: http://localhost
→ API : http://localhost:8000
Docker Hub Images

API: haidaryousif/tallyformer-api:latest
Frontend: haidaryousif/tallyformer-frontend:latest

Model Usage Example
Pythonfrom transformers import AutoTokenizer
import torch
from workflow.model import Transformer, ModelConfig, InferenceConfig  # adjust import as needed

tokenizer = AutoTokenizer.from_pretrained("./TallyFormer-Finance-51M")
model, config, _ = Transformer.load("./TallyFormer-Finance-51M/model.pth", device="cuda")

prompt = "<|user|> Explain the difference between value and growth investing. <|assistant|>"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.inference_mode():
    output_ids, _ = model.generate(
        inputs["input_ids"],
        InferenceConfig(max_new_tokens=256, temperature=0.7, top_p=0.9)
    )

response = tokenizer.decode(output_ids[0], skip_special_tokens=False)
print(response)
Contributing
Contributions are welcome! Please open issues or submit pull requests with clear descriptions.
License
[Specify your license here – e.g., MIT License]
Acknowledgments

Architecture and training methodology inspired by modern open-source LLMs
Teacher model: GPT-2 Medium (OpenAI)
SFT dataset: Finance-Alpaca
Training platform: Pod.ai (RTX A6000 instance)
Built with PyTorch, Transformers, PEFT, FastAPI, React, and Docker

TallyFormer-Finance-51M is production-ready and optimized for financial applications. Upload the TallyFormer-Finance-51M/ directory directly to the Hugging Face Hub to share or deploy via Transformers!