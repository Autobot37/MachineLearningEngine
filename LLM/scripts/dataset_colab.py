from datasets import load_dataset
#dataset = load_dataset()#but if using this method then it is already formed  no need to form class
import os
import numpy as np
import torch
import requests
from tiktoken import encoding_for_model as enc
device='cuda'
block_size = 8

class Tokenizer:
  def __init__(self):
    self.encoder = enc("gpt2")

  def __call__(self,text):
    idx = self.encoder.encode(text)
    return idx
  
  def decode(self,idx):
    assert idx.ndim == 1
    text = self.encoder.decode(idx)
    return text

tokenizer = Tokenizer()

def write(url):
  input_file_path = os.path.join(os.getcwd(),'input.txt')
  with open(input_file_path,'w')  as f:
    f.write(requests.get(url).text)

def export(path):
  with open(path,'r') as f:
    data = f.read()
    ids = tokenizer(data)
    print(f"total ids:{len(ids)}")

    ids = np.array(ids,dtype=np.uint16)
    ids.tofile(os.path.join(os.getcwd(),'train.bin'))
  
class data(torch.utils.data.Dataset):##custom daatset
  def __init__(self,bin_file,lightning=False):
    super().__init__()
    self.data = np.memmap(bin_file,dtype=np.uint16,mode='r')#memmap
  
  def __len__(self):
    return len(self.data)-block_size

  def __getitem__(self,i):
    data = self.data
    x = torch.from_numpy((data[i:i+block_size]).astype(np.int64))
    y = torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64))
    if device == 'cuda' and lightning==False:
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x, y
    return x, y




