# make sure to run this as administartor if you face permission issues
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
import os

cache_dir = "./tokenizer/gpt2_local_cache"

local_model_path = snapshot_download(
    repo_id="gpt2",
    cache_dir=cache_dir,
    local_dir_use_symlinks=False,  
    ignore_patterns=["*.h5"],       
    force_download=True            
)
print(f"Downloaded GPT-2 to: {local_model_path}")
tokenizer = AutoTokenizer.from_pretrained(
    local_model_path,
    local_files_only=True,
    use_fast=True
)
print(f"Vocab size: {tokenizer.vocab_size}")

# add special tokens
special_tokens = {
    "pad_token": "<|pad|>",
    "additional_special_tokens": ["<|user|>", "<|assistant|>"]
}
num_added_tokens = tokenizer.add_special_tokens(special_tokens)
print(f"Added {num_added_tokens} new tokens")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

PAD_TOKEN_ID = tokenizer.pad_token_id
print(f"PAD_TOKEN_ID: {PAD_TOKEN_ID}")
print(f"Vocab size: {tokenizer.vocab_size}")
print(f"Special tokens: {tokenizer.all_special_tokens}")

# save tokenizer for future use without needing snapshot_download
tokenizer.save_pretrained("./tokenizer/gpt2_local_cache_final")
print("Tokenizer saved locally for future use.")
