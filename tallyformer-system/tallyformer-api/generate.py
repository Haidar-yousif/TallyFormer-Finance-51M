# test the generation before containarize the app
import torch
from transformer import Transformer, ModelConfig, InferenceConfig, generate_text
from transformers import AutoTokenizer
model_paths = {
    "pretrain": "./models/pretrain_tallyformer.pth",
    "distilled": "./models/tallyformer-distilled-phase2.pth",
    "sft": "./models/model.pth"
}

tokenizer = AutoTokenizer.from_pretrained("./tokenizer/gpt2_local_cache_final", local_files_only=True)

model_name = "sft"  # "pretrain", "distilled", or "sft"
model_path = model_paths[model_name]

# load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, model_config, _ = Transformer.load(model_path, device=device)
print(f"Device {device}")
prompt = "One way to make money is"

# inference configuration
inference_config = InferenceConfig(
    max_new_tokens=50,
    temperature=0.7,
    topk=500,
    topp=0.9,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    mode="combined",
    return_only_generated=True
)

# generate the text
print(" Generated Text ... \n")
output = generate_text(model, tokenizer, prompt, inference_config, is_sft=(model_name=="sft"))
print(output)
