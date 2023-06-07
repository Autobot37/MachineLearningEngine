scale = 12
block_size = 8
learning_rate = 6e-4 
max_iters = 600 
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95 
n_layer = 2
n_head = 2
n_embd = 96
betas = (beta1,beta2)
dropout = 0.0 
bias = False
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 24
vocab_size = 2000
out_dir = 'out'
eval_interval = 200
log_interval = 1
eval_iters = 200
eval_only = False 
always_save_checkpoint = True 
init_from = 'scratch' 
# wandb logging
wandb_log = True # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2'
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 64
# adamw optimizer
learning_rate = 6e-4
max_iters = 600 
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 
# learning rate decay settings
decay_lr = True 
warmup_iters = 20 
lr_decay_iters = 600
min_lr = 6e-5 
# DDP settings
backend = 'gloo' 
# system
device = 'cuda' 
dtype = 'float16' 
compile = False

configdict = {k:v for k,v in globals().items() if not k.startswith("_") and isinstance(v,(int,float,bool,str))}

class Config(dict):
  def __init__(self,config):
    self.__dict__.update(config)
    
from dataclasses import dataclass
@dataclass
class ModelConfig:
  scale:int=6
  block_size:int=1024
  vocab_size:int=50304
  n_layer:int=12//scale
  n_embd:int=768//scale
  n_head:int=12//scale
  dropout:float=0.0
  bias:bool=True