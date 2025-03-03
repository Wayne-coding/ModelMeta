from pprint import pprint

from src.gpt import GPT_Model, config, GPT

from gpt_torch import GPT_Model as GPT_Model_torch, GPTPyTorch

model = GPT()
model_torch = GPTPyTorch(config)
layer_names_ms = model.cells_and_names()
layer_names_torch = model_torch.named_modules()
layer_names_str_ms = []
layer_names_str_torch = []
for layer in layer_names_ms:
    layer_names_str_ms.append(layer[0])
for layer in layer_names_torch:
    layer_names_str_torch.append(layer[0])
pprint(set(layer_names_str_ms) - set(layer_names_str_torch))
pprint(set(layer_names_str_torch) - set(layer_names_str_ms))
# pprint(layer_names_str_torch)
# for i in range(len(layer_names_str_ms)):
    # if layer_names_str_ms[i] not in layer_names_str_torch:
    #     print(layer_names_str_ms[i])
    #     print("not in torch")
    # if layer_names_str_torch[i] not in layer_names_str_ms:
    #     print(layer_names_str_torch[i])
    #     print("not in ms")
