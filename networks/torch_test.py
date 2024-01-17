
import torch
import torch.nn as nn

# 定义多头注意力模块
embed_dim = 16
num_heads = 1
attention = nn.MultiheadAttention(embed_dim, num_heads)

# 输入数据 (query, key, value)
batch_size = 1
seq_length = 2621440

query = torch.randn(seq_length, batch_size, embed_dim)
key = torch.randn(seq_length, batch_size, embed_dim)
value = torch.randn(seq_length, batch_size, embed_dim)
print(query.shape)
# 调用多头注意力模块
output, attention_weights = attention(query, key, value)

print("Output shape:", output.shape)
print("Attention weights shape:", attention_weights.shape)
