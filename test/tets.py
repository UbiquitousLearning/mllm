import torch
# import torch.nn as nn
# import torch.nn.functional as F

n = torch.randint(0,1000,(1, 2,))
print(n)
print(n.shape)
y = torch.nn.Embedding(1000,1024)
print(y.weight.data.shape)
y = y(n)
print(y)
print(y.shape)
# # nch
# y = n*torch.rsqrt(n.pow(2).mean(dim=-1, keepdim=True))
# print(y.shape)
# y1 = torch.zeros((1, 2, 3))
# print(n.size(0))
# for n1 in range(n.size(0)):
#     for n2 in range(n.size(2)):
#         sum = 0
#         for n3 in range(n.size(1)):
#             print(n1, n3, n2)
#             sum = n[n1, n3, n2] * n[n1, n3, n2]
#         sum = torch.rsqrt(sum / n.size(1))
#         y1[n1, n3, n2] = n[n1, n3, n2] / sum
# print(y1.shape)
# print(y1 == y)
# print(y1)
# print(y)
# from transformers import LlamaTokenizer
# import torch
# tokenizer = LlamaTokenizer.from_pretrained('decapoda-research/llama-7b-hf')
# ids = tokenizer.encode("hello world!")
# print(ids)