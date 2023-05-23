import lightning.pytorch as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from colabModel import *
#@title
class plModel(pl.LightningModule):
  def __init__(self,config):
    super().__init__()
    self.losslist = []
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
    
    
  def forward(self,x):#X=[B,CONTEXT_SIZE] = B,T
    b,t = x.size()
    pos = torch.arange(0,t,dtype=torch.float32,device=device).unsqueeze(0)
    t_emb = self.transformer.wte(x)
    #B,CONTEXT_SIZE,VOCAB_SIZE = B,T,C
    p_emb = self.transformer.wpe(x)
    x = self.transformer.drop(t_emb+p_emb)
    for block in self.transformer.h:
      x = block(x)
    x = self.transformer.ln_f(x)
    logits = self.lm_head(x)
    return logits
  
  def training_step(self,batch,batch_idx):
    x,y = batch
    logits  = self(x)
    B,T,C = logits.shape
    logits = logits.view(B*T,C)
    target = y.view(B*T)
    loss = F.cross_entropy(logits,target)
    self.losslist.append(loss)
    return loss

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(),lr=0.02)

  def predict_step(self,batch,batch_idx):
    x,y = batch
    logits = self(x)
    last = logits[:,-1,:]#B C
    probs = F.log_softmax(last,dim=-1) # B C
    new_idx = torch.multinomial(last.exp(),num_samples=1)
    idx = torch.cat((idx,new_idx),dim=1)#B T+1
    return idx
  