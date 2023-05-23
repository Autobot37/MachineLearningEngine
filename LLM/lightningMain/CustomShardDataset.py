import lightning.pytorch as pl
from torch.utils.data import DataLoader
import os
import numpy as np
import torch
import lightning as L
from lightning import Fabric as f

import tiktoken
enc = tiktoken.get_encoding("gpt2")

block_size = 8

config = {k:v for k,v in globals().items() if not k.startswith("_") and isinstance(v,(int,float,bool,str))}

class Config(dict):
  def __init__(self,config):
    self.__dict__.update(config)

config = Config(config)

class NextToken(torch.utils.data.Dataset):##custom daatset
  def __init__(self,bin_file,config,shard_index,num_shards):
    super().__init__()
    self.data = np.memmap(bin_file,dtype=np.uint16,mode='r')#memmap
    self.config = config
    self.num_shards = num_shards
    self.shard_index = shard_index
  
  def __len__(self):
    self.shard_size = (len(self.data)-self.config.block_size)//(self.num_shards)
    return self.shard_size

  def __getitem__(self,i):
    data = self.data
    start_index = self.shard_index * self.shard_size + i
    end_index = start_index  + self.config.block_size 
    x = torch.from_numpy((data[start_index : end_index]).astype(np.int64))
    y = torch.from_numpy((data[start_index+1 : end_index+1]).astype(np.int64))
    return x, y

class DataModule(pl.LightningDataModule):
  def __init__(self,dir_path,config):
    self.dir_path = dir_path
    self.prepare_data_per_node = True
    self.config = config
  
  def prepare_data(self):
    input_file_path = os.path.join(self.dir_path,"input.txt")

    with open(input_file_path,'r') as f:
      text = f.read()
    n = len(text)

    train_data = text[:int(n*0.9)]
    val_data = text[int(n*0.9):]
    #encode
    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)
    #print
    if fabric.is_global_zero:
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
      self.train_data = NextToken(bin_file=os.path.join(self.dir_path,"train.bin"),config=self.config,shard_index=fabric.local_rank,num_shards=fabric.world_size)#train_dataset = torch.utild.dataset(nextTokendataset)
      if fabric.is_global_zero:
        self.val_data = NextToken(bin_file=os.path.join(self.dir_path,"val.bin"),config=self.config,shard_index=fabric.local_rank,num_shards=fabric.world_size)
    
    if stage == "predict":
      self.predict_data = NextToken(bin_file=os.path.join(self.dir_path,"predict.bin"),config=self.config,shard_index=fabric.local_rank,num_shards=fabric.world_size)

  def train_dataloader(self):
    return DataLoader(self.train_data, batch_size=64, pin_memory=True, num_workers=0)  
  
  def val_dataloader(self):
    return DataLoader(self.val_data, batch_size=64, pin_memory=True, num_workers=0)
  
  def predict_dataloader(self):
    return DataLoader(self.predict_data, batch_size=64, pin_memory=True, num_workers=0)
    


####testing individual file

##dm  = DataModule("C:\\Users\\SHIVA SINGH\\Documents\\GitHub\\MachineLearningEngine\\LLM\\data",config=config)
##dm.setup(stage="fit")
##dt = dm.train_dataloader()
##for a,b in dt:
#   print(a)
#   print(b)
#   break


###
#Additional docs
by dedault we are addingg functionaly of distrbiuted sampler which will be turned off by trainer(replace_ddp=False)
###
