vocab_size = 50304
batch_size = 8
import math
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import lightning.pytorch as pl
from contextlib import nullcontext
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
####################################
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'gloo' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False
###################################
config = {k:v for k,v in globals().items() if not k.startswith("_") and isinstance(v,(int,float,bool,str))}
config['compile'] = False

#@title
class Model(nn.Module):
  def __init__(self,vocab_size):
    super().__init__()
    self.embedding_table = nn.Embedding(vocab_size,64)
    self.vocab = nn.Linear(64,vocab_size)
  
  def forward(self,idx,target=None):#X=[B,CONTEXT_SIZE] = B,T
    logits = self.embedding_table(idx) #B,CONTEXT_SIZE,VOCAB_SIZE = B,T,C
    if target is None:
      loss = None
    else:
      B,T,C = logits.shape
      logits = logits.view(B*T,C)
      logits = self.vocab(logits)
      target = target.view(B*T)
      loss = F.cross_entropy(logits,target)

    return logits, loss

  
  
  def generate(self,idx,max_tokens):
    for _ in range(max_tokens):
      logits, loss = self(idx)
      last = logits[:,-1,:]#B C
      probs = F.log_softmax(last,dim=-1) # B C
      new_idx = torch.multinomial(last.exp(),num_samples=1)
      idx = torch.cat((idx,new_idx),dim=1)#B T+1
    return idx
    



#@title
class dataset(torch.utils.data.Dataset):
  def __init__(self):
    self.l = torch.arange(1000)
  def __getitem__(self,idx):
    x = self.l[idx:idx+10]
    y = self.l[idx+1:idx+11]
    return x,y
  def __len__(self):
    return len(self.l.tolist())-11


dataset = dataset()
dataloader = torch.utils.data.DataLoader(dataset,batch_size=8,pin_memory=True,shuffle=True)

class ModelwithAddons(Model):
  def __init__(self,vocab_size=2000):
    super().__init__(vocab_size)
  
  def configure_optimizers(self,lr=0.02):
    return torch.optim.Adam(self.parameters(),lr=lr)

model = ModelwithAddons()
optim = torch.optim.Adam(model.parameters(),lr=0.02)
#rookie train
#scratch train bloodingnerves
class Trainer:
  def __init__(self,model,optim,dataloader,config):
    self.model = model
    self.optim = model.configure_optimizers() if hasattr(model,"configure_optimizers") else optim  #or configure optimizers
    self.dataloader = dataloader
    self.config = config
    for k, v in config.items():
        setattr(self, k, v)
    print(self.gradient_accumulation_steps)



  def setup_ddp(self):
    init_process_group(backend=self.backend)
    self.ddp_rank = int(os.environ['RANK'])
    self.ddp_local_rank = int(os.environ['LOCAL_RANK'])
    self.ddp_world_size = int(os.environ['WORLD_SIZE'])
    self.device = f"cuda:{self.ddp_local_rank}"
    torch.cuda.set_device(self.device)
    self.master_process = self.ddp_rank == 0
    self.seed_offset = self.ddp_rank
    self.gradient_accumulation_steps //= torch.cuda.device_count()

  def setup_numbers(self):
    torch.manual_seed(69+self.seed_offset)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ptdtype = torch.float16
    self.ctx = torch.amp.autocast(device_type='cuda',dtype=ptdtype)
    self.scaler = torch.cuda.amp.GradScaler(enabled=True)

  def train(self,max_epochs):
    self.ddp = int(os.environ.get('RANK',-1)) != -1
    if self.ddp:
      self.setup_ddp()
    else:
      self.master_process = True
      self.seed_offset = 0
      self.ddp_world_size = 1
      self.device = 'cuda:0'
      torch.cuda.set_device(self.device)
    
    tokens_per_iter = self.gradient_accumulation_steps*self.ddp_world_size*self.block_size*self.batch_size
    print(f"using tokens_per_iteration:{tokens_per_iter} tokens")

    if self.master_process:
      os.makedirs(out_dir,exist_ok=True)
    
    self.setup_numbers()

    iter_num = 0
    best_val_loss = 1e9

    self.model.to(self.device)

    self.model.train()

    if self.compile:
      print("compiling")
      self.model = torch.compile(self.model)
    

    if self.ddp:
      model = DDP(self.model, device_ids=[self.ddp_local_rank])
    

    for _ in range(self.max_iters):

      lr = self.get_lr(iter_num)
      self.optim.defaults['lr'] = lr

      if self.max_iters % self.eval_interval == 0 and self.master_process:
        losses = self.estimate_loss(['train'])
        print(f"step {iter_num}: train loss {losses['train']:.4f}")

        #self.wandblog(iter_num,losses['train'],lr)

        if losses['train'] < best_val_loss or self.always_save_checkpoint:
          best_val_loss = losses['train']
          self.save_checkpoints(self.model.state_dict(),iter_num,best_val_loss)


      for batch,(x,y) in enumerate(dataloader):
        x = x.to(self.device)
        y = y.to(self.device)

        with self.ctx:
          logits, loss = self.model(x,y)

        self.scaler.scale(loss).backward()

        if self.grad_clip != 0:
          self.scaler.unscale_(self.optim)
          torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.grad_clip)
        
        self.scaler.step(self.optim)
        self.scaler.update()
        self.optim.zero_grad(set_to_none=True)

        self.log()
    
    
    if ddp:
      destroy_process_group()      


  @torch.no_grad()
  def estimate_loss(self,splits):
    out = {}
    self.model.eval()
    for split in splits:
      losses = torch.zeros(self.eval_iters)
      for k in range(self.eval_iters):
        idx = np.random.randint(0,len(self.dataloader.dataset))
        x,y = self.dataloader.dataset[idx]
        x = x.to(self.device).unsqueeze(0)
        y = y.to(self.device).unsqueeze(0)
        with self.ctx:
          logits, loss = self.model(x,y)
        losses[k] = loss.item()
      out[split] = losses.mean()
    self.model.train()
    return out
  
  def get_lr(self,it):
    return 0.01
  
  def save_checkpoints(self,*args,**kwargs):
    pass
  
  def wandblog(self,**kwargs):
    pass
  
  def log(self):
    pass
    



t = Trainer(model,optim,dataloader,config)
t.train(10)