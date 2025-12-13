import math
import warnings
from collections import defaultdict
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic.dataclasses import dataclass
from pydantic import Field
from rich.table import Table
from rich.console import Console
from transformers import PretrainedConfig

# MLP Class
class MLP(nn.Module):
  """
  Simple Multi-Layer Perceptron with one hidden Layer
  It expands the input embedding size,applies GELU activation and then projects it back
  """
  def __init__(self,n_embed,dropout=0.1,rms_norm=False):
    super().__init__()
    self.hidden=nn.Linear(n_embed,4*n_embed,bias=not rms_norm)
    self.gelu=nn.GELU()
    self.proj=nn.Linear(4*n_embed,n_embed)
    self.dropout=nn.Dropout(dropout)
  def forward(self,x):
    # standard MLP in transformer: Linear -> GELU -> Linear -> Dropout
    return  self.dropout(self.proj(self.gelu(self.hidden(x))))

#MultiHead Attention Class
class MultiHeadAttention(nn.Module):
  """
  This module combines multiple attention heads in parallel .The outputs of each head
  are concatenated to form the final output
  """
  def __init__(self,n_head,n_embed,context_length,dropout=0.1):
    super().__init__()
    assert n_embed % n_head==0,"n_embed must be devisible by n_head"
    self.context_length=context_length
    self.n_head=n_head
    self.head_size=n_embed//n_head
    assert self.head_size%2==0,"head_size must be even for RoPE (split into pairs)."
    self.n_embed=n_embed
    #single linear layer for multi-head projections
    self.key=nn.Linear(n_embed,n_embed,bias=False)
    self.query=nn.Linear(n_embed,n_embed,bias=False)
    self.value=nn.Linear(n_embed,n_embed,bias=False)
    self.proj = nn.Linear(n_embed, n_embed)
    self.attn_dropout=nn.Dropout(dropout)
    self.resid_dropout=nn.Dropout(dropout)


    # RoPE cache
    self.register_buffer('cos_cached',torch.empty(0),persistent=False)
    self.register_buffer('sin_cached',torch.empty(0),persistent=False)
    self.register_buffer('causal_mask',torch.tril(torch.ones(context_length,context_length)))
   
    self._build_rope_cache()

  def _build_rope_cache(self,device=None,seq_len=None):
    device=device or next(self.parameters()).device
    seq_len=seq_len or self.context_length
    i_th=torch.arange(0,self.head_size,2,dtype=torch.float32,device=device)
    omega=10000**(-i_th/self.head_size) # head_size//2
    position=torch.arange(seq_len,dtype=torch.float32,device=device)
    theta=torch.outer(position,omega) # (context_length ,head_size//2)

    sin=torch.sin(theta).unsqueeze(0).unsqueeze(0) # (1,1,context_length ,head_size//2)
    cos=torch.cos(theta).unsqueeze(0).unsqueeze(0) # (1,1,context_length ,head_size//2)
    self.sin_cached=sin
    self.cos_cached=cos
    self.causal_mask=self.causal_mask.to(device)
  def apply_rope(self,q,k,start_pos=0):
    # q,k : (B,n_head,T,head_size)
    B,n_head,T,C=q.shape
    max_pos=start_pos+T
    if self.sin_cached.numel() == 0 or self.sin_cached.shape[2]<max_pos or self.sin_cached.device != q.device:
        self._build_rope_cache(device=q.device,seq_len=max_pos)
    sin=self.sin_cached[:,:,start_pos:start_pos+T,:] # (1,1,T,head_size//2)
    cos=self.cos_cached[:,:,start_pos:start_pos+T,:] # (1,1,T,head_size//2)
    def rotate(x):
      x1=x[...,0::2]
      x2=x[...,1::2]
      return torch.cat([x1*cos-x2*sin,x1*sin+x2*cos],dim=-1)
    return rotate(q),rotate(k)
  def forward(self,x,attn_mask=None,cache=None,use_flash_attn=True):
    """
    x: (B, T, C)
    cache: dict with keys 'k' and 'v' for cached past keys and values
    Returns:
        out: attention output
        updated_cache: dict with updated 'k' and 'v'
    """
    #assure input is the same dtype as weights
    if x.dtype != self.query.weight.dtype:
        x = x.to(dtype=self.query.weight.dtype)
    #assure input on the same device as rope cache
    if self.sin_cached.device != x.device:
      self._build_rope_cache(device=x.device)
    B,T,C=x.shape
    #project all heads at once
    q=self.query(x).view(B,T,self.n_head,self.head_size).transpose(1,2) # (B,n_head,T,head_size)
    k=self.key(x).view(B,T,self.n_head,self.head_size).transpose(1,2) # (B,n_head,T,head_size)
    v=self.value(x).view(B,T,self.n_head,self.head_size).transpose(1,2) # (B,n_head,T,head_size)

    # Compute old_len before concatentation
    old_len=cache['k'].shape[2] if (cache is not None and 'k' in cache) else 0
    #Apply RoPE
    q,k=self.apply_rope(q,k,start_pos=old_len)

    #Append last kv-cache if they exist

    if cache is not None and 'k' in cache and 'v' in cache:
        old_k=cache['k'].to(device=k.device,dtype=k.dtype)
        old_v=cache['v'].to(device=v.device,dtype=v.dtype)
        k=torch.cat([old_k,k],dim=2) #concatenate along sequence dimention
        v=torch.cat([old_v,v],dim=2)
    #update the cache
    cache={'k':k,'v':v}
    scale_factor = 1.0 /math.sqrt(self.head_size)
    S=k.shape[2] # full key length (old+new)

    use_flash=(hasattr(F, 'scaled_dot_product_attention') and
                 x.is_cuda and
                 x.dtype in (torch.float16, torch.float32, torch.bfloat16) and
                  attn_mask is None and use_flash_attn)
    if use_flash:
      out = F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=self.attn_dropout.p if self.training else 0.0)
    else:
      # manual attention
      attn_weights=(q @ k.transpose(-2,-1))*scale_factor # (B,n_head,T,S)
      row_start=old_len
      row_end=old_len+T

      # compute causal mask (T x S)
      if S <= self.context_length:
         causal_mask = self.causal_mask[row_start:row_end, :S].to(attn_weights.device)
      else:
         full_mask = torch.tril(torch.ones(self.context_length, S, device=attn_weights.device))
         causal_mask = full_mask[row_start:row_end, :S]

      # causal_mask: (T,S) -> expand to (1,1,T,S)
      causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
      combined_mask=causal_mask
      if attn_mask is not None: # Training padding
        attn_mask_broadcast = attn_mask[:, None, None, :S].to(dtype=causal_mask.dtype,device=attn_weights.device)
        combined_mask = causal_mask * attn_mask_broadcast  # (B,1,T,S)
      attn_weights = attn_weights.float()
      neg_inf = float(torch.finfo(attn_weights.dtype).min / 2)
      attn_weights = attn_weights.masked_fill(combined_mask == 0, neg_inf)
      attn_weights = attn_weights - attn_weights.amax(dim=-1, keepdim=True)
      attn_weights = F.softmax(attn_weights, dim=-1)
      attn_weights=self.attn_dropout(attn_weights) # apply attention dropout
      attn_weights=torch.nan_to_num(attn_weights,nan=0.0)
      out=attn_weights @ v # (B,n_head,T,head_size)
    out=out.transpose(1,2).reshape(B,T,C) # (B,T,n_head*head_size)
    out=self.proj(out)
    out=self.resid_dropout(out) # apply residual dropout

    return out,cache
#RMSNorm
class RMSNorm(nn.Module):
  def __init__(self,n_embed,eps=1e-8):
    super().__init__()
    self.eps=eps
    self.scale=nn.Parameter(torch.ones(n_embed)) #learnable scale
  def forward(self,x):
    # x: (B,T,n_embed) or (B,n_embed)
    rms=x.pow(2).mean(-1,keepdim=True)
    norm=torch.sqrt(rms+self.eps)
    scale=self.scale
    if x.ndim==3:
      scale=scale.view(1,1,-1)
    elif x.ndim==2:
      scale=scale.view(1,-1)
    # return x/(sqrt(sum(x**2))) * scale :  (B,T,n_embed) or (B,n_embed)
    return (x/norm) *scale

#Transformer Block Class
class Block(nn.Module):
  """
  Transformer block:
    - Multi-Head Attention + residual
    - MLP + residual
    - LayerNorm before each module

  This block consists of a multi-head attention layer followed by an MLP,
  with layer normalization and residual connections.
  """
  def __init__(self,n_head,n_embed,context_length,norm_type="pre",rms_norm=False):
    super().__init__()
    self.rms_norm=rms_norm
    self.norm_type=norm_type
    self.n_head=n_head
    self.n_embed=n_embed
    if rms_norm and norm_type != "pre":
            print("[Warning] RMSNorm with post-norm is uncommon and may reduce stability.")

    NormClass=RMSNorm if rms_norm else nn.LayerNorm
    self.ln1=NormClass(n_embed)
    self.attn=MultiHeadAttention(n_head=n_head,n_embed=n_embed,context_length=context_length)
    self.ln2=NormClass(n_embed)
    self.mlp=MLP(n_embed=n_embed,rms_norm=rms_norm)
    #residual scale for RMS Norm with post_norm
    self.res_scale=0.9 if rms_norm and norm_type!='pre' else 1.0

  def forward(self,x,attn_mask=None, cache=None,use_flash_attn=True):
    """
    x: (B, T, C)
    cache: dict with keys 'k' and 'v' for cached past keys and values
    Returns:
        out: block output
        updated_cache: dict with updated 'k' and 'v'
    """
    if self.norm_type == "pre":
        attn_out,new_cache = self.attn(self.ln1(x),attn_mask=attn_mask,cache=cache,use_flash_attn=use_flash_attn)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))

    else:
        attn_out,new_cache = self.attn(x,attn_mask=attn_mask,cache=cache,use_flash_attn=use_flash_attn)
        x = self.ln1(x + attn_out*self.res_scale)
        x = self.ln2(x + self.mlp(x)*self.res_scale)
    return x,new_cache

# MODEL CONFIG
@dataclass
class ModelConfig:
    context_length: int = Field(..., gt=0, description="Maximum context length")
    n_embed: int = Field(..., gt=0, description="Embedding dimension")
    n_head: int = Field(..., gt=0, description="Number of attention heads")
    n_block: int = Field(..., gt=0, description="Number of transformer blocks")
    vocab_size: int = Field(..., gt=1, description="Vocabulary size")
    pad_token_id: int = Field(..., ge=0, description="Padding token index")
    use_zloss: bool = Field(False, description="Enable z-loss regularization")
    zloss_coeff: float = Field(1e-4, ge=0.0, description="Coefficient for z-loss")
    dropout: float = Field(0.1, ge=0.0, le=1.0, description="Dropout probability")
    norm_type: str = Field("pre", description="Normalization type: 'pre' or 'post'")
    rms_norm: bool = Field(False, description="Use RMSNorm instead of LayerNorm")
    model_type : str = Field("custom", description="Model type")
    tie_word_embeddings: bool = Field(True, description="Tie word embeddings and output embeddings")

# INFERENCE CONFIG
@dataclass
class InferenceConfig:
    max_new_tokens: int = Field(...,gt=0,description="Maximum number of new tokens to generate")
    temperature: float = Field(1.0,ge=1e-5,description="Sampling temperature")
    topk: Optional[int] = Field(None,ge=0,description="Top-k sampling")
    topp: Optional[float] = Field(None,ge=0.0,le=1.0,description="Top-p (nucleus) sampling probability")
    frequency_penalty: float = Field(0.0,ge=0.0,description="Penalty for repeated tokens based on frequency")
    presence_penalty: float = Field(0.0,ge=0.0,description="Penalty for repeated tokens based on presence")
    mode: str = Field("combined", description="Sampling mode: 'combined' or 'independent'")
    eos_tokens: list|int|str | None = Field(None,description="End of sequence tokens")
    return_only_generated: bool = Field(True, description="Return only generated text after assistant token")



#Final Transformer
class Transformer(nn.Module):
  """
  This class combines token and position embeddings with a sequence of Transformer blocks
  and a final linear layer for language modeling.
  """
  def __init__(self,config:ModelConfig):
    super().__init__()

    # PEFT (LoRA) prerequisites
    cfg_dict = dict(vars(config))
    self.config = PretrainedConfig(**cfg_dict)
    # Model setup
    self.ignore_index=-100
    self.context_length= config.context_length
    self.n_block=config.n_block
    self.n_embed=config.n_embed
    self.n_head=config.n_head
    self.vocab_size=config.vocab_size
    self.pad_token_id=config.pad_token_id
    self.use_zloss=config.use_zloss
    self.zloss_coeff=config.zloss_coeff
    self.dropout_p=config.dropout
    self.norm_type=config.norm_type
    self.rms_norm=config.rms_norm
    NormClass = RMSNorm if self.rms_norm else nn.LayerNorm
    self.token_embed=nn.Embedding(self.vocab_size,self.n_embed)
    self.dropout=nn.Dropout(self.dropout_p)
    self.attn_blocks=nn.ModuleList([Block(self.n_head,self.n_embed,self.context_length,self.norm_type,self.rms_norm) for _ in range(self.n_block)])
    self.layer_norm=NormClass(self.n_embed)
    self.lm_head=nn.Linear(self.n_embed,self.vocab_size,bias=False) #projects back to vocabulary logits for prediction
    self.lm_head.weight=self.token_embed.weight # weight tying ,this will reduce the model size by (n_embed*vocab_size)
    self.apply(self._init_weights)

  # HuggingFace helper methods expected by some wrappers
  def get_input_embeddings(self):
      return self.token_embed

  def set_input_embeddings(self, new_embeddings):
      self.token_embed = new_embeddings
      # re-tie lm_head if needed
      if hasattr(self, "lm_head"):
          self.lm_head.weight = self.token_embed.weight

  def get_output_embeddings(self):
      return self.lm_head

  def set_output_embeddings(self, new_output):
      self.lm_head = new_output

  # Initialize module weights
  def _init_weights(self, m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        # Special initialization for projection and MLP layers
        if hasattr(m, 'in_features') and hasattr(m, 'out_features'):
            if any(m is block.attn.proj or m is block.mlp.proj for block in self.attn_blocks):
                nn.init.normal_(m.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.n_block))
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)



  # Initialize rope-cache
  def _init_rope_cache(self,device=None):
    #to make sure that the rope_cache are on the same device of the Model
    for block in self.attn_blocks:
        block.attn._build_rope_cache(device=device)

  # Compute model weights and memory size
  def _get_size(self):
      from rich.table import Table
      from rich.console import Console
      # compute the number of parameters of the model
      num_params=sum(p.numel() for p in self.parameters())

      # compute the memory size of the model
      dtype_sizes=defaultdict(int)
      for p in self.parameters():
        dtype_sizes[str(p.dtype)]+=p.numel()*p.element_size()
      for b in self.buffers():
        dtype_sizes[str(b.dtype)]+=b.numel()*b.element_size()
      total_size_mb=sum(dtype_sizes.values())/(1024**2)
      #create rich table
      table=Table(title='Model Memory Summary',show_lines=True)
      table.add_column("Data Type",justify='center',style='bold yellow',no_wrap=True)
      table.add_column("Memory (MB)",justify='right',style='green')
      table.add_column("Share (%)",justify='right',style='green')
      for dtype,size in dtype_sizes.items():
        size_mb=size/(1024**2)
        share=(size_mb/total_size_mb)*100
        table.add_row(dtype,f"{size_mb:.2f}",f"{share:.2f}%")
      console=Console()
      console.print(f"[bold yellow]Number of parameters:[/bold yellow] {num_params:,}")
      console.print(f"[bold yellow]Total memory usage:[/bold yellow] {total_size_mb:.2f} MB")
      console.print(table)
  # Apply token embedding with dropout
  def _pre_attn_pass(self,idx):
    """
    Combines token and position embeddings
    idx:Input token indices (B,T)

    """
    B,T=idx.shape
    token_emb=self.token_embed(idx)
    x=self.dropout(token_emb) # (B , T , n_embed)
    return x

  # Save the Model
  @staticmethod

  def save(model, path, config: ModelConfig = None, extra_dict: dict = None):
        state_dict = (
            model.module.state_dict()
            if isinstance(model, nn.DataParallel)
            else model.state_dict()
        )
    
        save_dict = {
            "state_dict": state_dict
        }
    
        if config is not None:
            save_dict["config_dict"] = config.__dict__
    
        if extra_dict is not None:
            save_dict["extra"] = extra_dict     # safer
    
        torch.save(save_dict, path)
        print(f"Model saved to {path}")

  # Load the Model
  @classmethod

  def load(cls, path, device=None, strict=True):
        """
        Universal load: works with any saved checkpoint (clean or dirty).
        Auto-fixes _orig_mod. prefixes, works with weights_only=True, gives clear errors.
        """
        import torch
        
        device = device or 'cpu'
        
        try:
            # This works on ALL PyTorch versions + fixes weights_only issue
            saved = torch.load(path, map_location=device, weights_only=False)
        except Exception as e:
            if "weights_only" in str(e) or "UnpicklingError" in str(e):
                raise RuntimeError(
                    f"Failed to load {path}\n"
                    "→ This file is either corrupted (e.g. HTML from Google Drive) or saved with complex objects.\n"
                    "→ Run: !ls -lh '{path}' and !head -c 200 '{path}'\n"
                    "→ Real .pth files are ~195MB and show binary garbage."
                ) from e
            raise
    
        # Extract config
        config_dict = saved.get("config_dict") or saved.get("config", None)
        if config_dict is None:
            raise ValueError("No model config found in checkpoint!")
    
        config = ModelConfig(**config_dict)
        model = cls(config).to(device)
    
        # Extract state_dict (could be directly saved or nested)
        state_dict = saved.get("state_dict", saved if isinstance(saved, dict) else None)
        if state_dict is None:
            raise ValueError("No state_dict found in checkpoint!")
    
        # === AUTO-FIX _orig_mod. prefixes (from DataParallel) ===
        if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
            print("Detected _orig_mod. prefixes → auto-cleaning state_dict...")
            new_sd = {}
            for k, v in state_dict.items():
                new_k = k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k
                new_sd[new_k] = v
            state_dict = new_sd
    
        # Load weights
        model.load_state_dict(state_dict, strict=strict)
    
        # Extra info
        extra = saved.get("extra", {})
        epoch = extra.get("epoch", "?")
        val_ppl = extra.get("val_ppl", "?")
        if isinstance(val_ppl, (int, float)): val_ppl = f"{val_ppl:.2f}"
    
        print(f"Model loaded successfully!")
        print(f"   Epoch: {epoch} | Val PPL: {val_ppl} | Params: ~51M | Device: {device}")
    
        return model, config, extra
    

  # Override the to() method
  def to(self,*args,**kwargs):
    """
    Overrides nn.Module.to() to ensure that rope cache is initialized
    after the model is moved to a new device.
    """
    model=super().to(*args,**kwargs)
    device = args[0] if len(args) > 0 else kwargs.get('device')
    if device is None and len(args) > 0:
        device = args[0]
    # normalize to torch.device if it's a string
    if isinstance(device, str):
        device = torch.device(device)
    if  device is None:
      try:
        device=next(model.parameters()).device
      except StopIteration:
        device=None
    if device is not None:
      model._init_rope_cache(device=device)
      if hasattr(self, "lm_head") and hasattr(self, "token_embed"):
            print("The tite done")
            self.lm_head.weight = self.token_embed.weight
    return model

  # Define perplexity as metric
  def perplexity(self, dataloader, device):
    was_training=self.training
    self.eval()
    total_loss = 0.0
    total_tokens = 0.0
    with torch.inference_mode():
        for xb, yb, attn_mask in tqdm(dataloader, desc="Computing Perplexity"):
            xb, yb, attn_mask = xb.to(device), yb.to(device), attn_mask.to(device)
            logits, loss , _ = self(input_ids=xb, labels=yb, attention_mask=attn_mask,use_flash_attn=False)
            if loss is not None and not torch.isnan(loss):
                valid_tokens= int((yb != self.ignore_index).sum().item())
                total_loss += loss.item() * valid_tokens
                total_tokens += valid_tokens
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    ppl = math.exp(avg_loss)
    if was_training:
      self.train() #restore the original one
    return ppl

  # Apply forward propagation
  def forward(self, input_ids=None, inputs_embeds=None, labels=None,attention_mask=None,past_key_values=None,use_flash_attn=True,** kwargs):
      # alias target labels to targets
      if 'targets' in kwargs and labels is None:
          labels = kwargs.pop('targets')
  
      if inputs_embeds is not None:
        x = inputs_embeds
        B, T = x.shape[:2]
      else:
        if input_ids is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided")
        x = self._pre_attn_pass(input_ids)
        B, T = input_ids.shape

      if T>self.context_length:
        raise ValueError(f"Input sequence length {T} exceeds context length {self.context_length};ensure tokenization truncates.")


      if past_key_values is not None:

        assert isinstance(past_key_values,(list,tuple)) and len(past_key_values)==len(self.attn_blocks),"past_cache must be a list/tuple with one dict per block"
        #Trim cache across layers if total seq exceed

        for i,layer_cache in enumerate(past_key_values):
          if 'k' not in layer_cache or 'v' not in layer_cache:
            continue
          old_len=layer_cache['k'].shape[2]
          total_len=old_len+T
          if total_len>self.context_length:
            trim_len=total_len-self.context_length
            keep=max(0,old_len-trim_len)
            #slice to keep the last position
            if keep==0:
              layer_cache['k']=layer_cache['k'][:,:,0:0,:]
              layer_cache['v']=layer_cache['v'][:,:,0:0,:]
            else:
              layer_cache['k']=layer_cache['k'][:,:,-keep:,:]
              layer_cache['v']=layer_cache['v'][:,:,-keep:,:]
            import warnings
            warnings.warn(
                        f"Warning: Trimmed {trim_len} positions from layer {i+1} cache to keep context_length={self.context_length}. "
                        "RoPE absolute positions will shift accordingly (trimming older positions)."
                    )

      new_cache=[]
      for i,block in enumerate(self.attn_blocks):
          cache=past_key_values[i] if past_key_values is not None else None
          x,cache=block(x,attention_mask,cache,use_flash_attn=use_flash_attn)
          new_cache.append(cache)
      x = self.layer_norm(x)
      logits = self.lm_head(x) # B x T x vocab_size
      loss = None
      if labels is not None:
          B, T, C = logits.shape
          flat_logits =logits.reshape(-1,C) #logits.view(B * T, C)
          labels = labels.view(B * T).long()
          valid_count = (labels != self.ignore_index).sum().item()
          if valid_count > 0:
              ce_loss = F.cross_entropy(flat_logits, labels, ignore_index=self.ignore_index)
              loss = ce_loss
              if self.use_zloss:

                mask = (labels != self.ignore_index)  # (B*T,)
                if mask.any():
                    lse = torch.logsumexp(flat_logits, dim=-1)  # (B*T,)
                    z_reg = (lse[mask] ** 2).mean()             # mean squared over valid tokens
                    loss += self.zloss_coeff * z_reg

          else:
              print("Warning: No valid targets in batch, skipping loss computation")
      return logits, loss,new_cache

  # Generate tokens
  @torch.inference_mode()
  def generate(self, input_ids, config : InferenceConfig,tokenizer=None):
      """
      Generate tokens auto-regressively.

      Args:
          idx: torch.LongTensor (B, T) input token indices
          max_new_tokens: int, number of new tokens to generate
          temperature: float, softmax temperature
          topk: int, keep top-k tokens
          topp: float, cumulative probability for nucleus sampling
          frequency_penalty: float, frequency penalty
          presence_penalty: float, presence penalty
          mode: "combined" or "independent"
              - combined: topp is applied on topk-masked logits
              - independent: topp is applied on original logits regardless of topk
          tokenizer: transformers.PreTrainedTokenizer, tokenizer for decoding tokens

      """
      # Helper functions for handling eos-tokens
      def normalize_eos_tokens(eos_tokens,tokenizer):
            """
            Normalize eos_tokens into a list of token-id sequences (list[list[int]]).
            Returns:
            [
                [50256],                # single-token EOS
                [198,198,198],          # multi-token EOS ("\n\n\n")
                [10009]                 # special chat eos
            ]
            """
            if eos_tokens is None or tokenizer is None:
              return []
            if isinstance(eos_tokens,(int,str)):
              eos_tokens=[eos_tokens]
            normalized=[]
            for item in eos_tokens:
              if isinstance(item,int):
                normalized.append([item]) # single token ids
                continue

              if isinstance(item,str):
                ids=tokenizer.encode(item,add_special_tokens=False)
                normalized.append(ids)
                continue

              raise ValueError("eos_tokens must contain ints or strings")
            return normalized

      def ends_with_pattern(sequence,pattern):
        if len(sequence)<len(pattern):
          return False
        return sequence[-len(pattern):]==pattern

      # Helper function to apply frequence and presence penalty to the output logits
      def apply_penalties(logits, input_ids, frequency_penalty=0.0, presence_penalty=0.0):
            """
            Vectorized version — applies frequency and presence penalties to logits.
            Args:
                logits: (B, vocab_size)
                idx: (B, T)
            """
            if frequency_penalty==0.0 and presence_penalty==0.0:
              return logits
            B,vocab_size=logits.shape
            counts=torch.zeros(B,vocab_size,device=logits.device) # B x vocab_size
            counts.scatter_add_(dim=1,index=input_ids,src=torch.ones_like(input_ids,dtype=torch.float,device=logits.device))
            logits-=frequency_penalty*counts
            logits-=presence_penalty*(counts>0).float()
            return logits
      # unpack config
      max_new_tokens = config.max_new_tokens
      if max_new_tokens==0:
          return input_ids,0.0
      temperature = config.temperature if config.temperature is not None and config.temperature >0.0 else 1.0
      topk = config.topk
      topp = config.topp
      frequency_penalty = config.frequency_penalty
      presence_penalty = config.presence_penalty
      mode = config.mode
      # normalize eos_tokens
      eos_patterns=normalize_eos_tokens(config.eos_tokens,tokenizer)
      #setup
      idx = input_ids.long()
      device=next(self.parameters()).device
      idx=idx.to(device)
      B,prompt_len=idx.shape
      #compute effective prompt length (handle legacy padding)
      effective_prompt_len=prompt_len
      if self.pad_token_id is not None:
        #find last non padding position
        last_non_pad_per_batch = (idx != self.pad_token_id).sum(dim=-1)
        effective_prompt_len = int(last_non_pad_per_batch.min().item())
      #soft check
      if effective_prompt_len>self.context_length:
        import warnings
        warnings.warn(f"Effective prompt length {effective_prompt_len} exceeds context length {self.context_length}; truncating.")
        idx=idx[:,-self.context_length:]
        effective_prompt_len=min(effective_prompt_len,self.context_length)
      # Initial forward pass on full prompt : exact RoPE ,no cache
      with torch.amp.autocast(device_type="cuda", enabled=False):
          logits,_,past_key_values=self(input_ids=idx,labels=None,attention_mask=None,past_key_values=None,use_flash_attn=False)

      # Cap to remaining context
      remaining_ctx = self.context_length-effective_prompt_len
      max_gen = min(max_new_tokens, remaining_ctx)
      if max_gen<=0:
        import warnings
        warnings.warn("No room in context; returning prompt unchanged.")
        return idx,0.0

      # start time
      import time
      start_time=time.time()
      ttft=None

      for step in range(max_gen):
          logits = logits.float()
          logits = logits[:, -1, :] / temperature # B x vocab_size
          logits = apply_penalties(logits, idx, frequency_penalty, presence_penalty)
          logits = torch.nan_to_num(logits, nan=0.0, neginf=-1e9, posinf=1e9)

          # Top-k filtering
          if topk is not None:
              topk_logits, topk_indices = torch.topk(logits, k=topk, dim=-1) # B x topk
              mask = torch.full_like(logits, float('-inf')) # B x vocab_size
              mask.scatter_(-1, topk_indices, topk_logits)
              logits_topk = mask
          else:
              logits_topk = logits

          # Top-p sampling
          if topp is not None:
              probs = F.softmax(logits_topk if mode == 'combined' else logits, dim=-1) # B x vocab_size
              probs = torch.nan_to_num(probs, nan=0.0)
              sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1) # B x vocab_size
              cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
              sorted_probs[cumulative_probs > topp] = 0
              sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-9)
              sorted_probs = torch.clamp(sorted_probs, min=1e-9)
              idx_next = torch.multinomial(sorted_probs, num_samples=1) # B x 1
              idx_next = sorted_indices.gather(dim=-1, index=idx_next)
          else:
              probs = F.softmax(logits_topk, dim=-1)
              probs = torch.nan_to_num(probs, nan=0.0)
              probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-9)
              probs = torch.clamp(probs, min=1e-9)
              idx_next = torch.multinomial(probs, num_samples=1) # B x 1
          # append token to the sequence
          idx = torch.cat((idx, idx_next), dim=-1)
          if ttft is None:
            ttft=time.time()-start_time
          #check eos_tokens : only on generated tokens
          if eos_patterns:

            hit=False
            gen_suffix=idx[:,effective_prompt_len:].tolist()
            for seq in gen_suffix:
              for pattern in eos_patterns:
                if ends_with_pattern(seq,pattern):
                  hit=True
                  break
              if hit:
                break
            if hit:
              import warnings
              warnings.warn(f"EOS pattern detected in generated tokens after {step+1} new tokens; early stop.")
              break
          #continue with kv-cache
          with torch.amp.autocast(device_type="cuda", enabled=False):
              logits,_,past_key_values=self(input_ids=idx[:,-1:].to(device),labels=None,attention_mask=None,past_key_values=past_key_values,use_flash_attn=False)

      return idx,ttft
  def prepare_inputs_for_generation(self, input_ids,past_key_values,attention_mask, **kwargs):
      # PEFT calls this during .generate()
      """
      Returns a dict that huggingface/PEFT expects when preparing next-step generation.
      """
      return {
          "input_ids": input_ids,
          "past_key_values": past_key_values,
          "attention_mask": attention_mask
      }
    




def generate_text(model, tokenizer, prompt, config: InferenceConfig,is_sft=False):
    import time
    from rich.console import Console
    console = Console()
    start_time = time.time() # timer

    # prepare the prompt
    if isinstance(prompt, str):
        prompt = [prompt]
    if is_sft:
        user_token = '<|user|>'
        assistant_token = '<|assistant|>'
        prompt = [f"{user_token} {p.strip()} {assistant_token}" for p in prompt]

    # tokenize the prompt
    encoded = tokenizer(
        prompt,
        truncation=True,
        max_length=model.context_length,
        padding=False,
        return_tensors='pt'
    )

    device = next(model.parameters()).device
    input_ids = encoded['input_ids'].to(device)

    was_training = model.training
    model.eval()

    # generate

    output_ids,ttft=model.generate(input_ids=input_ids, config=config, tokenizer=tokenizer)

    
    ttft=0.0 if ttft is None else ttft*1000 # in ms


    total_time = time.time() - start_time
    prompt_tokens = input_ids.numel()
    output_tokens = output_ids.numel()
    generated_tokens = output_tokens - prompt_tokens
    gen_tps = generated_tokens / total_time
    total_tps = output_tokens / total_time

    console.print(f"[bold yellow]Prompt Tokens:[/bold yellow] {prompt_tokens}")
    console.print(f"[bold yellow]Generated Tokens:[/bold yellow] {generated_tokens}")
    console.print(f"[bold yellow]Total Time:[/bold yellow] {total_time:.2f} seconds")
    console.print(f"[bold yellow]Generated Tokens/sec:[/bold yellow] {gen_tps:.2f}")
    console.print(f"[bold yellow]Total Tokens/sec (prompt+output):[/bold yellow] {total_tps:.2f}")
    console.print(f"[bold yellow]TTFT:[/bold yellow] {ttft:.2f} ms")

    # decode the output
    output_text=[]
    for i,ids in enumerate(output_ids):
      seq=ids.tolist()
      if config.return_only_generated:
        if is_sft:
              assistant_id=tokenizer.convert_tokens_to_ids(assistant_token)
              if assistant_id in seq:
                idx=seq.index(assistant_id)+1
                seq=seq[idx:]
        else:
              idx=len(input_ids[i])
              seq=seq[idx:]
      decoded=tokenizer.decode(seq, skip_special_tokens=True)
      output_text.append(decoded)


    if len(prompt) == 1:
        output_text = output_text[0]

    if was_training:
        model.train()

    return output_text

