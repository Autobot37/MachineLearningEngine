context = torch.tensor(enc.encode("dont let me"),dtype=torch.int64)
enc.decode(model.generate(context.unsqueeze(0).to(device),max_tokens=10)[0].tolist())