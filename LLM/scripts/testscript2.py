import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import numpy as np

train_data = np.arange(1, 101)

class dataset(torch.utils.data.Dataset):
  def __init__(self):
    self.train_data = train_data
  
  def __getitem__(self,i):
    data = self.train_data
    x = torch.from_numpy(data[i:i+8]).float()
    y = torch.from_numpy((data[i+1:i+1+8])).float()
    return x.view(1,-1), y

  def __len__(self):
    return len(self.train_data)-9


data = dataset()

test = torch.utils.data.DataLoader(data,batch_size=2)

class LitModel(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.l1 = nn.Linear(8,8)
    self.loss = []
  
  def forward(self,x):
    return torch.relu(self.l1(x.view(x.size(0),-1)))
  
  def training_step(self,batch,batch_idx):
    x, y = batch
    y_hat = self(x)
    loss = F.cross_entropy(y_hat, y)
    self.loss.append(loss)
    return loss
  
  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(),lr=0.02)

from lightning.pytorch.strategies import DDPStrategy
ddp = DDPStrategy(process_group_backend="gloo")
trainer = pl.Trainer(max_epochs=10,accelerator="gpu",devices=1,strategy="ddp",num_nodes=1)
model = LitModel()
trainer.fit(model,train_dataloaders=test)

