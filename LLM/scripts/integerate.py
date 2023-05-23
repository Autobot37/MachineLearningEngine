import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import lightning.pytorch as pl
import tiktoken
from tiktoken import encoding_for_model as enc
from dataclasses import dataclass
from colabModel import *
from colabModelpl import *
from dataset_colab import *





device = 'cuda'

model = Model(ModelConfig)

modelpl = plModel(ModelConfig)

write("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
export(os.path.join(os.getcwd(),'input.txt'))

dataset = data(os.path.join(os.getcwd(),'train.bin'))

dataloader = torch.utils.data.DataLoader(
    dataset,
    shuffle = True,
    batch_size = 12
)




