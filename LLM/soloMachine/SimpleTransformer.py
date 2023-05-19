#@title
class Head(nn.Module):
  def __init__(self,head_size):
    super().__init__()
    self.key = nn.Linear(n_embd,head_size,bias=False)# B T H
    self.query = nn.Linear(n_embd,head_size,bias=False) # B T H
    self.value = nn.Linear(n_embd,head_size,bias=False) # B T H
  
  def forward(self,x):
    out = F.scaled_dot_product_attention(self.query(x),self.key(x),self.value(x))#B T H
    return out
  


#@title
class MultiHead(nn.Module):
    def __init__(self,num_heads,head_size):
      super().__init__()
      self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
      self.proj = nn.Linear(n_embd,n_embd)
      self.dropout = nn.Dropout(dropout)
    
    def forward(self,x):
      out = torch.cat([h(x) for h in self.heads],dim=-1)
      out = self.dropout(self.proj(out))
      return out


#@title
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

#@title
class Block(nn.Module):
  def __init__(self,n_embd,n_head):
    super().__init__()
    head_size = n_embd // n_head
    self.sa = MultiHead(n_head,head_size)
    self.ffwd = FeedForward(n_embd)
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)
  
  def forward(self,x):
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x
  


#@title
class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
    self.pos_encoding = nn.Embedding(context_size, n_embd)
    self.blocks = nn.Sequential(*[Block(n_embd,n_head=n_head) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embd)
    self.lm_head = nn.Linear(n_embd,vocab_size)
  
  def forward(self,idx,target=None):#X=[B,CONTEXT_SIZE] = B,T
    print("forward")
    B, T = idx.shape
    pos_emb = self.pos_encoding(torch.arange(T,device=device)) # T,C
    print("pos passed")
    tok_emb = self.token_embedding_table(idx)  #B,CONTEXT_SIZE,VOCAB_SIZE = B,T,C
    print("emb passed")
    x = pos_emb + tok_emb#B T C
    print("enc passed")
    x = self.blocks(x) # B T C
    print("blocks passed")
    x = self.ln_f(x) # B,T ,C
    logits = self.lm_head(x) # B T vocab_size
    if target is None:
      loss = None
    else:
      B,T,C = logits.shape
      logits = logits.view(B*T,C)
      target = target.view(B*T)
      loss = F.cross_entropy(logits,target)

    return logits, loss
  
  def generate(self,idx,max_tokens):
    for _ in range(max_tokens):
      idx = idx[:,-context_size:]
      logits, loss = self(idx)
      last = logits[:,-1,:]#B C
      probs = F.log_softmax(last,dim=-1) # B C
      new_idx = torch.multinomial(last.exp(),num_samples=1)
      idx = torch.cat((idx,new_idx),dim=1)#B T+1
    return idx
  

