#@title
# "quantized_model = torch.quantization.quantize_dynamic(
#     model, {torch.nn.Linear}, dtype=torch.qint8
# )
# print(quantized_model)

#@title
# def print_size_of_model(model):
#     torch.save(model.state_dict(), "temp.p")
#     print('Size (MB):', os.path.getsize("temp.p")/1e6)
#     os.remove('temp.p')

# print_size_of_model(model)
# print_size_of_model(quantized_model)

