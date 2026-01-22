import torch
import torch.nn as nn
import torch.nn.functional as F

embed = nn.Embedding(num_embeddings= 4, embedding_dim= 2)

print("Lookup Table: ", embed.weight.shape)

input_data = torch.tensor([[1, 0, 3], [2, 3, 1]])


output = embed(input_data)

print("input shape: ", input_data.shape)
print("output shape: ", output.shape)