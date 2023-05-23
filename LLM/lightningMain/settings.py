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

configdict = {k:v for k,v in globals().items() if not k.startswith("_") and isinstance(v,(int,float,bool,str))}

class Config(dict):
  def __init__(self,config):
    self.__dict__.update(config)
    
