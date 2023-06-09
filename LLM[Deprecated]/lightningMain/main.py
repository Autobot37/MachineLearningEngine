# %%


# %%
import lightning.pytorch as pl
from torch.utils.data import DataLoader
import os
import numpy as np
import torch
import lightning as L
from lightning import Fabric as f
import torch.nn as nn
from dataclasses import dataclass
from torch.nn import functional as F
import math

# %%
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# %%
import sentencepiece
from sentencepiece import SentencePieceProcessor,SentencePieceTrainer

# %%
from typing import Optional

# %%
#@title
#Tokenizer[model path]

class Tokenizer:

  def __init__(self, model_path):
    self.processor = SentencePieceProcessor(model_file=str(model_path))
    self.bos_id = self.processor.bos_id()
    self.eos_id = self.processor.eos_id()
    self.pad_id = self.processor.pad_id()
  
  @property
  def vocab_size(self) -> int:
    return self.processor.vocab_size()


#encode
  def encode(
        self,
        string: str,
        bos: bool = False,
        eos: bool = False,
        max_length: int = -1,
        pad: bool = False,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
    tokens = self.processor.encode(string)
    if bos:
        tokens = [self.bos_id] + tokens
    if eos:
        tokens = tokens + [self.eos_id]
    if max_length > 0:
        tokens = tokens[:max_length]
    if pad and len(tokens) < max_length:
        tokens += [self.pad_id] * (max_length - len(tokens))

    return torch.tensor(tokens, dtype=torch.long, device=device)


#decode
  def decode(self,tokens:torch.Tensor) -> str:
    return self.processor.decode(tokens.tolist())
  
#train[take input txt bro][have export it on path]
  @staticmethod
  def train(input: str, destination: str, vocab_size=2000) -> None:
    model_prefix = os.path.join(destination, "tokenizer")
    SentencePieceTrainer.Train(input=input, model_prefix=model_prefix, vocab_size=vocab_size)

#@title
Tokenizer.train(input="C:\\Users\\SHIVA SINGH\\Documents\\GitHub\\MachineLearningEngine\\LLM[Deprecated]\\lightningMain\\data\\input.txt",destination="C:\\Users\\SHIVA SINGH\\Documents\\GitHub\\MachineLearningEngine\\LLM[Deprecated]\\lightningMain\\tokenizer")

#@title
enc = Tokenizer("C:\\Users\\SHIVA SINGH\\Documents\\GitHub\\MachineLearningEngine\\LLM[Deprecated]\\lightningMain\\tokenizer\\tokenizer.model")




# %%


# %%
scale = 12
block_size = 8
learning_rate = 6e-4 # max learning rate
max_iters = 600 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95 
n_layer = 2
n_head = 2
n_embd = 96
betas = (beta1,beta2)
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 24
vocab_size = 2000
out_dir = 'out'
eval_interval = 200
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = True # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 64
# model
 # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 20 # how many steps to warm up for
lr_decay_iters = 600 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'gloo' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False

# %%
config = {k:v for k,v in globals().items() if not k.startswith("_") and isinstance(v,(int,float,bool,str))}


# %%
class Config(dict):
  def __init__(self,config):
    self.__dict__.update(config)
    

# %%
config = Config(config)

# %%
config.block_size

# %%
class NextToken(torch.utils.data.Dataset):##custom daatset
  def __init__(self,bin_file,config):
    super().__init__()
    self.data = np.memmap(bin_file,dtype=np.uint16,mode='r')#memmap
    self.config = config
  
  def __len__(self):
    return (len(self.data) - self.config.block_size)

  def __getitem__(self,i):
    data = self.data
    start_index = i
    end_index = start_index + self.config.block_size
    x = torch.from_numpy((self.data[start_index:end_index]).astype(np.int64))
    y = torch.from_numpy((self.data[start_index + 1:end_index + 1]).astype(np.int64))
    return x, y

# %%
from torch.utils.data.distributed import DistributedSampler
class Sampler(DistributedSampler):
  def __init__(self,dataset,world_size,rank,shuffle):
    super().__init__(dataset=dataset,num_replicas=world_size,rank=rank,shuffle=shuffle)



# %%
class DataModule(pl.LightningDataModule):
  def __init__(self,dir_path,config,trainer=None):
    self.dir_path = dir_path
    self.prepare_data_per_node = True
    self.config = config
    self.trainer = trainer
    self.current_epoch = 0
  
  def prepare_data(self):
    input_file_path = os.path.join(self.dir_path,"input.txt")

    with open(input_file_path,'r') as f:
      text = f.read()
    n = len(text)

    train_data = text[:int(n*0.9)]
    val_data = text[int(n*0.9):]
    #encode
    train_ids = enc.encode(train_data)
    val_ids = enc.encode(val_data)
    #print
    if self.trainer.is_global_zero:
      print(f"train_ids has {len(train_ids)} tokens")
      print(f"val_ids has {len(val_ids)} tokens")
    #convert to numpy
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    #export_to_file
    train_ids.tofile(os.path.join(self.dir_path,'train.bin'))
    val_ids.tofile(os.path.join(self.dir_path,'val.bin'))

  def setup(self, stage):
    if stage == "fit":
      self.train_data = NextToken(bin_file=os.path.join(self.dir_path, "train.bin"), config=self.config)#train_dataset = torch.utild.dataset(nextTokendataset)
      if self.trainer.is_global_zero:
        self.val_data = NextToken(bin_file=os.path.join(self.dir_path, "val.bin"), config=self.config)
    
    if stage == "predict":
      self.predict_data = NextToken(bin_file=os.path.join(self.dir_path, "predict.bin"), config=self.config)

  def train_dataloader(self):
    if isinstance(self.trainer,pl.Trainer):
      return DataLoader(self.train_data, batch_size=64, pin_memory=True, num_workers=0,shuffle=True)
    else:
      self.sampler = Sampler(dataset=self.train_data,world_size=self.trainer.world_size, rank=self.trainer.local_rank, shuffle=True)
      self.sampler.set_epoch(self.current_epoch)
      return DataLoader(self.train_data, batch_size=64, pin_memory=True, num_workers=0, sampler=self.sampler)  
  
  def val_dataloader(self):
    if self.trainer.is_global_zero:
      return DataLoader(self.val_data, batch_size=64, pin_memory=False, num_workers=0,shuffle=True)
    else:
      print("only master node have val_data")
      return None
  
  def on_epoch_start(self):
    self.current_eoch = self.trainer.current_epoch


# %%
trainer = pl.Trainer(accelerator="cuda",max_epochs=1,use_distributed_sampler=True)

# %%
trainer.current_epoch

# # %%
# dm  = DataModule("C:\\Users\\SHIVA SINGH\\Documents\\GitHub\\MachineLearningEngine\\LLM[Deprecated]\\lightningMain\\data" ,config=config,trainer=trainer)


# # %%
# dm.prepare_data()

# # %%
# dm.setup(stage="fit")

# # %%
# dm.current_epoch = 12

# # %%
# dt = dm.train_dataloader()

# # %%


# %%
# for a,b in dt:
#   print(a,b)
#   break

# # %%
# dm.current_epoch = 9
# for a,b in dt:
#   print(a,b)
#   break

# %%
#########MODEL#####################

# %%
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
    self.ln_1 = LayerNorm(config.n_embd,config.bias)
    self.ln_2 = LayerNorm(config.n_embd,config.bias)
    self.mlp = MLP(config) # layernorm 2 a attention a mlp
  
  def forward(self,x):
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    return x

class LayerNorm(nn.Module):
  def __init__(self,ndim,bias):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(ndim))
    self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
  
  def forward(self,x):
    return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)
  

# %%
class plModel(pl.LightningModule):
  def __init__(self,config):
    super().__init__()
    self.losslist = []
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

  def _init_weights(self,module):
    if isinstance(module,nn.Linear):
      torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    if isinstance(module,nn.Embedding):
      torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
    
    
  def forward(self,x,y=None):#X=[B,CONTEXT_SIZE] = B,T
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
  
  def training_step(self,batch,batch_idx):
    x,y = batch
    x = x.to(self.device)
    y = y.to(self.device)
    logits = self(x)#b t vocab_dim
    B,T,vdim = logits.shape

    logits = logits.view(B*T,vdim)
    target = y.view(B*T)
    loss = F.cross_entropy(logits,target)
    self.losslist.append(loss)
    return loss

  def configure_optimizers(self):
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
  def generate(self,idx, max_new_tokens, temperature=1.0, top_k=None):
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

    
  

# %%
from dataclasses import dataclass
scale = 6
@dataclass
class ModelConfig:
  block_size:int=1024
  vocab_size:int=50304
  n_layer:int=12//scale
  n_embd:int=768//scale
  n_head:int=12//scale
  dropout:float=0.0
  bias:bool=True

# %%
model = plModel(config)
#model.device

# %%
#trainer.fit(model,dt)

# %%
##############Trainer######################

# %%
#now approach is forget about data and model, focus on trainer only using fabric customed thats it

# %%
#!pip install wandb

# %%
import wandb
#wandb.login()

# %%
import lightning as L

# %%
from lightning.fabric import Fabric

# %%
import time

# %%
import random

class Trainer:
  def __init__(self,config,model,datamodule):
    
    self.config = config
    self.fabric = Fabric(precision="16-mixed",accelerator="cuda",strategy="auto")
    self.model = model
    self.optimizer = model.configure_optimizers()
    self.device = self.fabric.device
    self.dm = datamodule(dir_path="C:\\Users\\SHIVA SINGH\\Documents\\GitHub\\MachineLearningEngine\\LLM[Deprecated]\\lightningMain\\data",config=self.config,trainer=self.fabric)
    self.dm.prepare_data()
    self.dm.setup(stage="fit")
    self.dm.setup(stage="val")

  def train(self):
    
    master_process = self.fabric.global_rank == 0

    if master_process:
      os.makedirs(self.config.out_dir, exist_ok=True)
    
    self.fabric.seed_everything(1227+self.fabric.global_rank)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    iter_num = 0
    best_val_loss = 1e9
    t0 = time.time()

    if self.config.wandb_log and master_process:
      wandb.init(project=self.config.wandb_project, name=self.config.wandb_run_name)

    self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)

    for iter_num in range(self.config.max_iters):
      
      lr = self.get_lr(iter_num,self.config) if self.config.decay_lr else self.config.learning_rate
      for param_group in self.optimizer.param_groups:
        param_group['lr'] = lr
  
      
      if iter_num % self.config.eval_interval == 0 and master_process:
        losses = self.estimate_loss(self.config)
        print(f"step {iter_num}: eval_loss {losses['val']:.4f}")

      
        if losses["val"] < best_val_loss or self.always_save_checkpoints:
          best_val_loss = losses["val"]
          self.checkpoint(model = self.model.state_dict(),iter=iter_num)

     
      

      self.optimizer.zero_grad(set_to_none=True)
      
      self.model.current_epoch = iter_num
      for batch_idx, batch in enumerate(self.dm.train_dataloader()):
        is_accumulating = batch_idx < self.config.gradient_accumulation_steps 
        with self.fabric.no_backward_sync(self.model,enabled=(is_accumulating)):
          loss = self.model.training_step(batch, batch_idx)
          self.wandblog(iter=iter_num,loss=loss)
          self.fabric.backward(loss)
        
        if not is_accumulating:
          self.optimizer.step()
          self.optimizer.zero_grad()
      

      t1 = time.time()
      dt = t1 - t0
      t0 = t1

      if master_process:
        lossf = loss.item()
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
      
  @torch.no_grad()
  def estimate_loss(self,config):
    out = {}
    self.model.eval()
    for split in ["val"]:
      losses = torch.zeros(config.eval_iters)
      for k,batch in enumerate(self.dm.val_dataloader()):
        loss = model.training_step(batch, k)
        losses[k] = loss.item()
        if k>config.eval_iters:
          break
      out[split] = losses.mean()
    self.model.train()
    return out
  
  def get_lr(self,iter,config):
    if iter < config.warmup_iters:
        return config.learning_rate * iter / config.warmup_iters
    if iter > config.lr_decay_iters:
        return config.min_lr
    decay_ratio = (iter - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)
  
  def checkpoint(self,model,iter):
    checkpoint = {
        "model":model,
        "iter_num":iter,
    }
    print(f"saving checkpoint{iter}")
    torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
  
  def wandblog(self,iter,loss):
    wandb.log({
        "iter": iter,
        "train_loss": loss,
    })
  



# %%
t = Trainer(config, model, DataModule)

# %%
t.train()

# %%
torch.cuda.memory_allocated()

# %%
torch.cuda.reset_max_memory_allocated()

# %%



