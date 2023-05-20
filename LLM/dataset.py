#!pip install sentencepiece
!pip install tiktoken

from tqdm.auto import tqdm

import os
from pathlib import Path
from typing import Optional

import torch
#from sentencepiece import SentencePieceProcessor, SentencePieceTrainer
import tiktoken

import torch

import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

import tiktoken
import numpy as np
enc = tiktoken.get_encoding("gpt2")


#advanced dataset
dir_path = '/content/data'
#can insert requests method
input_file_path = os.path.join(dir_path,"input.txt")
with open(input_file_path,'r') as f:
  text = f.read()
n = len(text)
train_data = text[:int(n*0.9)]
val_data = text[int(n*0.9):]

train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)

print(f"train_ids has {len(train_ids)} tokens")
print(f"train_ids has {len(val_ids)} tokens")

train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

train_ids.tofile(os.path.join(dir_path,'train.bin'))
val_ids.tofile(os.path.join(dir_path,'val.bin'))


dataset_name = "input.txt"
data_dir = '/content/data'
train_data = np.memmap(os.path.join(data_dir,'train.bin'),dtype=np.uint16,mode='r')
val_data = np.memmap(os.path.join(data_dir,'val.bin'),dtype=np.uint16,mode='r')

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


#should replace with advanced datasettings
# dataset = tokenizer.encode(total)
# n = int(0.9*(len(dataset)))
# train_data = dataset[:n]
# test_data = dataset[n:]
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# def get_batch(split):
#   data = train_data if split=='train' else val_data 
#   ix = torch.randint(len(data) - context_size, (batch_size,))
#   x = torch.stack([torch.from_numpy((data[i:i+context_size]).astype(np.int64)) for i in ix])
#   y = torch.stack([torch.from_numpy((data[i+1:i+1+context_size].astype(np.int64))) for i in ix])

#   if device == 'cuda':
#       x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
#   else:
#       x, y = x.to(device), y.to(device)
#   return x, y





