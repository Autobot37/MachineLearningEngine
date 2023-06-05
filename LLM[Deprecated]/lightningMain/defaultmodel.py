##trainer = pl.Trainer(use_distributed_sampler=False,max_epochs=1)
#from dataclasses import dataclass
###IF CUDA ASSERTION ERROR RUN CUSTOM OR COLAB GPU OR STEP BY STEP or cpus
import math
from typing import List, Optional, Tuple, Any, Union
from config import ModelConfig

class CausalSelfAttention(nn.Module):
  def __init__(self,config:Config) -> None:
    super().__init__()
    assert config.n_embd & config.n_head == 0
    self.c_attn = nn.Linear(config.n_embd,3*config.n_embd,bias=config.bias)
    self.c_proj = nn.Linear(config.n_embd,config.n_embd,bias=config.bias)
    self.attn_dropout = nn.Dropout(config.dropout)
    self.resid_dropout = nn.Dropout(config.dropout)

    self.n_head = config.n_head
    self.n_embd = config.n_embd
    self.dropout = config.dropout

  def forward(self,x:torch.Tensor) -> torch.Tensor:#B,T,C
    B,T,C = x.size()
    q,k,v = self.c_attn(x).split(self.n_embd,dim=2)
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
  def __init__(self,config:Config) -> None:
    super().__init__()
    self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd,bias=config.bias)
    self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd,bias=config.bias)
    self.dropout= nn.Dropout(config.dropout)
    self.act = nn.GELU()
  def forward(self,x:torch.Tensor) -> torch.Tensor:
    x = self.dropout(self.c_proj(self.act(self.c_fc(x))))
    return x

#@title
class Block(nn.Module):
  def __init__(self,config:Config) -> None:
    super().__init__()
    self.attn = CausalSelfAttention(config)
    self.ln_1 = LayerNorm(config.n_embd,config.bias)
    self.ln_2 = LayerNorm(config.n_embd,config.bias)
    self.mlp = MLP(config) # layernorm 2 a attention a mlp
  
  def forward(self,x:torch.Tensor) -> torch.Tensor:
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    return x

class LayerNorm(nn.Module):
  def __init__(self,ndim:int,bias:bool) -> None:
    super().__init__()
    self.weight = nn.Parameter(torch.ones(ndim))
    self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
  
  def forward(self,x:torch.Tensor) -> torch.Tensor:
    return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)
  
class plModel(pl.LightningModule):
  def __init__(self,config:Config) -> None:
    super().__init__()
    self.config = config
    print(f"this model is on {self.device}")
    self.transformer = nn.ModuleDict(dict(
        wte = nn.Embedding(config.vocab_size,config.n_embd),
        wpe = nn.Embedding(config.block_size,config.n_embd),
        h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        drop = nn.Dropout(config.dropout),
        ln_f = LayerNorm(config.n_embd,config.bias)
    ))
    self.lm_head = nn.Linear(config.n_embd,config.vocab_size,bias=False)

    self.apply(self._init_weights)

    for a,b in self.named_parameters():
      if a.endswith("c_proj.weight"):
        torch.nn.init.normal_(b,mean=0.0,std=0.02/math.sqrt(2 * config.n_layer))
  
    print(sum(p.numel() for p in self.parameters()))

  def _init_weights(self,module:nn.Module) -> None:
    if isinstance(module,nn.Linear):
      torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    if isinstance(module,nn.Embedding):
      torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
    
    
  def forward(self,x:torch.Tensor,y:Optional[torch.Tensor]= None) -> torch.Tensor:#X=[B,CONTEXT_SIZE] = B,T
    b,t = x.size()
    pos = torch.arange(0,t,dtype=torch.int32, device=self.device).unsqueeze(0)
    t_emb = self.transformer.wte(x)
    #B,CONTEXT_SIZE,VOCAB_SIZE = B,T,C
    p_emb = self.transformer.wpe(pos)
    x = self.transformer.drop(t_emb+p_emb)
    for block in self.transformer.h:
      x = block(x) # b t c
    out = self.transformer.ln_f(x)###BTC WE HAVE TO PASS THRROUGH LM_HEAD FOR B T VOCAB_SIZE
    logits = self.lm_head(out)
    return logits
  
  def training_step(self,batch:torch.Tensor,batch_idx:Union[int,torch.Tensor]) -> torch.Tensor:
    x,y = batch
    x = x.to(self.device)
    y = y.to(self.device)
    logits = self(x)#b t vocab_dim
    B,T,vdim = logits.shape

    logits = logits.view(B*T,vdim)
    target = y.view(B*T)
    loss = F.cross_entropy(logits,target)
    return loss

  def configure_optimizers(self) -> torch.optim:
    param_dict = {pn: p for pn, p in self.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': self.config.weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    optimizer = torch.optim.AdamW(optim_groups, lr=self.config.learning_rate, betas=(self.config.beta1,self.config.beta2))

    return optimizer
 

  @torch.no_grad()
  def generate(self,idx:torch.Tensor, max_new_tokens:Optional[int]=100, temperature:Optional[float]=1.0, top_k:Optional[int]=10):
    for _ in range(max_new_tokens):
      idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
      logits, _ = self(idx_cond)
      logits = logits[:, -1, :] / temperature
      if top_k is not None:
          v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
          logits[logits < v[:, [-1]]] = -float('Inf')
      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)

    return idx

    
#   model = plModel(config)
# model.device

