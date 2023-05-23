###ADVANCED TRAINING
from tqdm.auto import tqdm
import os
from pathlib import Path
from typing import Optional
import torch.nn as nn

############################
#@title
#hyperparams

out_dir = 'out'
eval_interval = 2
log_interval = 1
eval_iters = 2
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 10 # used to simulate larger batch sizes
batch_size = 4 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 8
# model
n_layer = 12
n_head = 12
n_embd = 10
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 60 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 20 # how many steps to warm up for
lr_decay_iters = 60 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'gloo' # 'nccl', 'gloo', etc.
# system
vocab_size = 50207
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # use PyTorch 2.0 to compile the model to be faster

############################
import torch
#from sentencepiece import SentencePieceProcessor, SentencePieceTrainer
from torch.nn import functional as F
import numpy as np


#advanced dataset
dir_path = "C:\\Users\\SHIVA SINGH\\OneDrive\\Documents\\LLM\\data"
#can insert requests method
# input_file_path = os.path.join(dir_path,"input.txt")
# with open(input_file_path,'r') as f:
#   text = f.read()
# n = len(text)
# train_data = text[:int(n*0.9)]
# val_data = text[int(n*0.9):]

# train_ids = enc.encode_ordinary(train_data)
# val_ids = enc.encode_ordinary(val_data)

# print(f"train_ids has {len(train_ids)} tokens")
# print(f"train_ids has {len(val_ids)} tokens")

# train_ids = np.array(train_ids, dtype=np.uint16)
# val_ids = np.array(val_ids, dtype=np.uint16)

# train_ids.tofile(os.path.join(dir_path,'train.bin'))
# val_ids.tofile(os.path.join(dir_path,'val.bin'))

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = 'cuda' if torch.cuda.is_available() else 'cpu'






from contextlib import nullcontext
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as dist
import os
import time
import math
import pickle
import os
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp



class Model(nn.Module):
  def __init__(self,vocab_size):
    super().__init__()
    self.embedding_table = nn.Embedding(vocab_size,1)
    self.vocab = nn.Linear(1,vocab_size)
  
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
    



  

#GPU id -> NO OF ALL GLOBAL GPUS
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    assert gradient_accumulation_steps % torch.cuda.device_count() == 0
    gradient_accumulation_steps //= torch.cuda.device_count()
else:
  # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
print(f"tokens per iteration will be:10")
if master_process:
  os.makedirs(out_dir,exist_ok=True)

#tokens_per_iter = gradient_accumulation_steps*args.world_size* batch_size * block_size
######################### DATA ##################################
data_dir = 'C:\\Users\\SHIVA SINGH\\OneDrive\\Documents\\LLM\\data'

def get_batch(split):
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
  y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
  if device == 'cuda':
      # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
      x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
  else:
      x, y = x.to(device), y.to(device)
  return x, y
########################### DATA ###############################

########################### NUMERICAL CONFIGS ###################

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast('cuda',dtype=torch.float16)
torch.manual_seed(1337 )
torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cudnn.allow_tf32 = True

########################### NUMERICAL CONFIGS ###################

########################### MODEL CONFIGS ###################
model = Model(vocab_size)

model.to(device)

########################### MODEL CONFIGS###################

########################### HYPERPARAMETER CONFIGS###################

iter_num = 0
best_val_loss = 1e9

########################### HYPERPARAMETER CONFIGS###################
scaler = torch.cuda.amp.GradScaler(enabled=(dtype=='float16'))

##optimizer
optimizer = torch.optim.Adam(model.parameters(),1e-4)


model = nn.parallel.DistributedDataParallel(model,device_ids=[ddp_local_rank])

@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  for split in ['train','val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      x, y = get_batch(split)
      with ctx:
        logits, loss = model(x,y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out

def get_lr(it):
  if it<warmup_iters:
    return learning_rate*it/warmup_iters
  if it<lr_decay_iters:
    return min_lr
  decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
  assert 0 <= decay_ratio <= 1
  coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
  return min_lr + coeff * (learning_rate - min_lr)

X, Y = get_batch('train')
t0 = time.time()

local_iter_num = 0

while True:
  lr = get_lr(iter_num) if decay_lr else learning_rate and master_proces

  if iter_num % eval_interval ==0 and master_process:
      losses = estimate_loss()
      print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    #save checkpoint
      if wandb_log:
          wandb.log({
          "iter":iter_num,
          "train/loss":losses['train'],
          "val/loss":losses['val'],
          "lr":lr
          })
    
      if losses['val'] < best_val_loss or always_save_checkpoint:
          best_val_loss = losses['val']
          if iter_num>0:
              checkpoint = {
                  'model': model.state_dict(),
                  #'optimizer': optimizer.state_dict(),
                  #'model_args': model_args,
                  'iter_num': iter_num,
                  'best_val_loss': best_val_loss,
                  #'config': config,
              }
              print(f"saving checkpoint to {out_dir}")
              torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
  if iter_num == 0 and eval_only:
    break

  for micro_step in range(gradient_accumulation_steps):

    
      model.require_backward_grad_sync = (micro_step==(gradient_accumulation_steps-1))
      with ctx:
          logits, loss = model(X,Y)
          loss = loss/gradient_accumulation_steps

      X, Y = get_batch('train') 
      scaler.scale(loss).backward()
  
  scaler.step(optimizer)
  scaler.update()
  optimizer.zero_grad(set_to_none=True)

  iter_num+=1
  local_iter_num +=1

  if iter_num > max_iters:
    break











