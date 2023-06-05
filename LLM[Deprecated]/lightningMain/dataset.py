import lightning.pytorch as pl
from torch.utils.data import DataLoader
import os
import numpy as np
import torch
import lightning as L
from lightning import Fabric as f
from torch.utils.data.distributed import DistributedSampler
from setting import configdict,Config
config = Config(configdict)

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

class Sampler(DistributedSampler):
  def __init__(self,dataset,world_size,rank,shuffle):
    super().__init__(dataset=dataset,num_replicas=world_size,rank=rank,shuffle=shuffle)

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
      return DataLoader(self.train_data, batch_size=64, pin_memory=True, num_workers=2,shuffle=True)
    else:
      self.sampler = Sampler(dataset=self.train_data,world_size=self.trainer.world_size, rank=self.trainer.local_rank, shuffle=True)
      self.sampler.set_epoch(self.current_epoch)
      return DataLoader(self.train_data, batch_size=64, pin_memory=True, num_workers=2, sampler=self.sampler)  
  
  def val_dataloader(self):
    if self.trainer.is_global_zero:
      return DataLoader(self.val_data, batch_size=64, pin_memory=False, num_workers=0,shuffle=True)
    else:
      print("only master node have val_data")
      return None
  
  def on_epoch_start(self):
    self.current_eoch = self.trainer.current_epoch

trainer = pl.Trainer(accelerator="cuda",max_epochs=10,use_distributed_sampler=True,strategy="auto")
dm  = DataModule("C:\\Users\\SHIVA SINGH\\OneDrive\\Documents\\LLM\\data\\input.txt",config=config,trainer=trainer)
dm.prepare_data()
dm.setup(stage="fit")
dm.current_epoch = 12
dt = dm.train_dataloader()
##additional notes

# you have to do two things

# datamodule.sampler.set_epoch(current_epoch)
# also since we have used sampler 
# turn off Trainer sampler frok pl
# Trainer(replace_sampler_ddp=False)