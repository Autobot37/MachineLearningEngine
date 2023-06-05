model = Model()
model = model.to(device)

print(sum(p.numel() for p in model.parameters()))

import time

#SUSSY TRAINING
optim = torch.optim.Adam(model.parameters(),lr=learning_rate)
start = time.time()
for iter in tqdm(range(1)):

  xb, yb = get_batch('train')

  logits, loss = model(xb, yb)

  optim.zero_grad()
  loss.backward()
  optim.step()
  print(loss)
end = time.time()
print("elapsed",end-start)
  

