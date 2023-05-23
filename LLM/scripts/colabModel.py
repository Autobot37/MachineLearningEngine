#@title
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):
  def __init__(self,config):
    super().__init__()
    assert config.n_embd & config.n_head == 0
    self.c_attn = nn.Linear(config.n_embd,3*config.n_embd,bias=config.bias)
    self.c_proj = nn.Linear(config.n_embd,config.n_embd,bias=config.bias)
    self.attn_dropout = nn.Dropout(config.dropout)
    self.resid_dropout = nn.Dropout(config.dropout)

    self.n_head = config.n_head
    self.n_embd = config.n_embd
    self.dropout = config.dropout

  def forward(self,x):#B,T,C
    B,T,C = x.size()
    q,k,v = self.c_attn(x).split(self.n_embd,dim=-1)
    q = q.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
    k = k.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
    v = v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)#B T nh headDim->b nh t hs

    y = torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=None,dropout_p=self.dropout if self.training else 0,is_causal=True) #B NH T HS
    y = y.transpose(1,2).contiguous().view(B,T,C)

    out = self.c_proj(y)
    out = self.resid_dropout(y)
    return out #b  t c

#@title
class MLP(nn.Module):
  def __init__(self,config):
    super().__init__()
    self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd,bias=config.bias)
    self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd,bias=config.bias)
    self.dropout= nn.Dropout(config.dropout)
    self.act = nn.GELU()
  def forward(self,x):
    x = self.dropout(self.c_proj(self.act(self.c_fc(x))))
    return x

#@title
class Block(nn.Module):
  def __init__(self,config):
    super().__init__()
    self.attn = CausalSelfAttention(config)
    self.ln_1 = nn.LayerNorm(config.n_embd)
    self.ln_2 = nn.LayerNorm(config.n_embd)
    self.mlp = MLP(config) # layernorm 2 a attention a mlp
  
  def forward(self,x):
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    return x
  
#@title
class Model(nn.Module):
  def __init__(self,config):
    super().__init__()
    self.config = config
    self.transformer = nn.ModuleDict(dict(
        wte = nn.Embedding(config.vocab_size,config.n_embd),
        wpe = nn.Embedding(config.block_size,config.n_embd),
        h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        drop = nn.Dropout(config.dropout),
        ln_f = nn.LayerNorm(config.n_embd)
    ))
    self.lm_head = nn.Linear(config.n_embd,config.vocab_size,bias=False)

    self.apply(self._init_weights)

    for a,b in self.named_parameters():
      if a.endswith("c_proj.weight"):
        torch.nn.init.normal_(b,mean=0.0,std=0.02/math.sqrt(2 * config.n_layer))
  
    print(sum(p.numel() for p in self.parameters()))

  def _init_weights(self,module):
    if isinstance(module,nn.Linear):
      torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module,nn.Embedding):
      torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
    
    
  def forward(self,x,target=None):#X=[B,CONTEXT_SIZE] = B,T
    b,t = x.size()
    pos = torch.arange(0,t,dtype=torch.float32,device=device).unsqueeze(0)
    t_emb = self.transformer.wte(x)
    #B,CONTEXT_SIZE,VOCAB_SIZE = B,T,C
    p_emb = self.transformer.wpe(x)
    x = self.transformer.drop(t_emb+p_emb)
    for block in self.transformer.h:
      x = block(x)
    x = self.transformer.ln_f(x)

    if target is None:#inference 
      loss = None
      logits = self.lm_head(x[:,[-1],:])
    else:
      logits = self.lm_head(x)
      B,T,C = logits.shape
      logits = logits.view(B*T,C)
      target = target.view(B*T)
      loss = F.cross_entropy(logits,target)

    return logits, loss


  @torch.no_grad()
  def generate(self,idx,max_tokens):
    for _ in range(max_tokens):
      idx = idx if idx.shape[-1] <= self.config.block_size else idx[:-self.config.block_size:]
      logits, _ = self(idx)
      last = logits[:,-1,:]#B C
      probs = F.log_softmax(last,dim=-1) # B C
      new_idx = torch.multinomial(last.exp(),num_samples=1)
      idx = torch.cat((idx,new_idx),dim=1)#B T+1
    return idx
    


from dataclasses import dataclass
scale = 12
@dataclass
class ModelConfig:
  block_size:int=1024
  vocab_size:int=50304
  n_layer:int=12//scale
  n_embd:int=768//scale
  n_head:int=12//scale
  dropout:float=0.0
  bias:bool=True
