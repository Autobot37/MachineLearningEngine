#@title
# class Model(nn.Module):
#   def __init__(self,vocab_size):
#     super().__init__()
#     self.embedding_table = nn.Embedding(vocab_size,64)
#     self.vocab = nn.Linear(64,vocab_size)
  
#   def forward(self,idx,target=None):#X=[B,CONTEXT_SIZE] = B,T
#     logits = self.embedding_table(idx) #B,CONTEXT_SIZE,VOCAB_SIZE = B,T,C
#     if target is None:
#       loss = None
#     else:
#       B,T,C = logits.shape
#       logits = logits.view(B*T,C)
#       logits = self.vocab(logits)
#       target = target.view(B*T)
#       loss = F.cross_entropy(logits,target)

#     return logits, loss
  
#   def generate(self,idx,max_tokens):
#     for _ in range(max_tokens):
#       logits, loss = self(idx)
#       last = logits[:,-1,:]#B C
#       probs = F.log_softmax(last,dim=-1) # B C
#       new_idx = torch.multinomial(last.exp(),num_samples=1)
#       idx = torch.cat((idx,new_idx),dim=1)#B T+1
#     return idx
    



