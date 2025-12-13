import time
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from transformer import Transformer, ModelConfig, InferenceConfig
from transformers import AutoTokenizer
from fastapi.middleware.cors import CORSMiddleware

app=FastAPI(title="TallyFormer LLM API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
tokenizer=AutoTokenizer.from_pretrained("./tokenizer/gpt2_local_cache_final", use_fast=True)
MODEL_PATHS = {
    "pretrain": "./models/pretrain_tallyformer.pth",
    "distilled": "./models/tallyformer-distilled-phase2.pth",
    "sft": "./models/model.pth"
}
loaded_models={}
def get_model(name:str):
    if name not in MODEL_PATHS:
        raise ValueError(f"Unknown model: {name}")
    if name not in loaded_models:
        model,_,_=Transformer.load(MODEL_PATHS[name],device='cuda' if torch.cuda.is_available() else 'cpu')
        loaded_models[name]=model.eval()
    return loaded_models[name]

# The request schema
class GenerationRequest(BaseModel):
    prompt: str
    model_name: Optional[str] = "sft"
    max_new_tokens: Optional[int] = 50
    temperature: Optional[float] = 0.7
    topk: Optional[int] = 500
    topp: Optional[float] = 0.9
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    mode: Optional[str] = "combined"  # "combined" or "independent"
    return_metrics: Optional[bool] = False  #flag for timing metrics

# model characteristics
MODEL_NAME = "TallyFormer"
MODEL_SIZE = "51M parameters"

CONTEXT_LENGTH = 256
N_EMBED = 512
N_HEAD = 8
N_BLOCKS = 8


#/welcome GET request
@app.get("/welcome")
def welcome():
    return {
        "message": "Welcome to TallyFormer LLM inference API!",
        "model_name": MODEL_NAME,
        "model_size": MODEL_SIZE,
        "context_length": CONTEXT_LENGTH,
        "layers": N_BLOCKS,
        "heads": N_HEAD,
        "hidden_dim": N_EMBED
    }
#/generate POST request
@app.post('/generate')
def generate(req:GenerationRequest):
    try:
        model=get_model(req.model_name)
        config = InferenceConfig(
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        topk=req.topk,
        topp=req.topp,
        frequency_penalty=req.frequency_penalty,
        presence_penalty=req.presence_penalty,
        mode=req.mode,
        return_only_generated=True)
        #prepare prompt
        input_text=req.prompt
        input_text=input_text.strip()
        if req.model_name=='sft':
            user_token='<|user|>'
            assistant_token='<|assistant|>'
            input_text=f"{user_token} {input_text} {assistant_token}"
        #tokenize
        encoded=tokenizer(
            [input_text],
            truncation=True,
            max_length=model.context_length,
            padding=False,
            return_tensors='pt'
        )
        device=next(model.parameters()).device
        input_ids=encoded['input_ids'].to(device)
        #generate
        start_time=time.time()
        output_ids,ttft=model.generate(input_ids=input_ids,config=config,tokenizer=tokenizer)
        ttft_ms=0.0 if ttft is None else ttft*1000.0 # in ms
        total_time=time.time()-start_time
        #metrics
        prompt_tokens=input_ids.numel()
        output_tokens=output_ids.numel()
        generated_tokens=output_tokens - prompt_tokens
        gen_tps=generated_tokens/total_time
        total_tps=output_tokens/total_time
        #decode
        seq=output_ids[0].tolist()
        if config.return_only_generated and req.model_name=='sft':
            assistant_id=tokenizer.convert_tokens_to_ids(assistant_token)
            if assistant_id in seq:
                idx=seq.index(assistant_id)+1
                seq=seq[idx:]
        else:
            seq=seq[len(input_ids[0]):]
        decoded=tokenizer.decode(seq,skip_special_tokens=True)
        response={"generated_text":decoded}
        if req.return_metrics:
            response["metrics"] = {
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "total_time_sec": total_time,
                "generated_tokens_per_sec": gen_tps,
                "total_tokens_per_sec": total_tps,
                "ttft_ms": ttft_ms
            }
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
