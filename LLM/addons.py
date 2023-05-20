#@title
class ModelWithAddons(Model):
  def __init__(self,config):
    super().__init__(config)
    pass
  def configure_optimizers(self,weight_decay,learning_rate,device_type):
    return torch.optim.Adam(self.parameters(),lr=learning_rate)

  @classmethod
  def from_pretrained(cls,model_type):
    assert model_type in {"gpt2","gpt2-medium","gpt2-large","gpt2-xl"}
    from transformers import GPT2LMHeadModel
    print(f"loading weights for {model_type}")

    config = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]

    config['vocab_size'] = 50257
    config['block_size'] = 1024
    config['bias'] = True

    config = ModelConfig(**config)
    model = Model(config)

    sd = model.state_dict()

    sd_keys = sd.keys()

    sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

    model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    sd_hf = model.state_dict()

    sd_keys_hf = sd_hf.keys()
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]

    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']


    for k in sd_keys_hf:
      if any(k.endswith(w) for w in transposed):
        assert sd_hf[k].shape[::-1] == sd[k].shape
        with torch.no_grad():
          sd[k].copy_(sd_hf[k].t())
      else:
        with torch.no_grad():
          sd[k].copy_(sd_hf[k])
    
    return model

  def init_model(self):
    pass