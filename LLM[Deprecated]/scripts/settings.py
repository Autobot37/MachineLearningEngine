batch_size = 12 
context_size = 8 
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 10
n_head = 1
n_layer = 1
vocab_size = 50257
dropout = 0.0