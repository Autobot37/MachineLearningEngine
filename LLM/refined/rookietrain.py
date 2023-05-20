vocab_size = 50304
batch_size = 8
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
import lightning.pytorch as pl
from contextlib import nullcontext
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

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
class Trainer:
  def __init__(self,model,optim,dataloader):
    self.model = model
    self.optim = model.configure_optimizers() if hasattr(model,"configure_optimizers") else optim  #or configure optimizers
    self.dataloader = dataloader
  
  def train(self,max_epochs):
    model.train()
    for i in range(max_epochs):
      for x,y in self.dataloader:
        logits, loss = model(x,y)
        print(f"iter:{i} -- loss:{loss}")
        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),0.5)
        self.optim.step()

t = Trainer(model,optim,dataloader)
t.train(10)