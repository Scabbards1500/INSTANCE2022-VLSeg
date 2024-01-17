import torch
import torch.nn as nn
import math
import unittest
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout=0, max_len: int = 2621440) -> None:
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model) # 所以还是这里创建的全零吗 [L,d_model]
        position = torch.arange(0, max_len).unsqueeze(1) # [L,1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)) # [d_model/2]=2
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # size=(1, L, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        #  output = word_embedding + positional_embedding
        x = x + nn.Parameter(self.pe[:, :x.size(1)], requires_grad=False)  # size = [batch, L, d_model]
        # print(x)
        # print(x.shape)
        return self.dropout(x)  # size = [batch, L, d_model]

# 创建一个示例的PositionalEncoding模块
pos_enc = PositionalEncoding(d_model=16, dropout=0.1, max_len=2621440)

# 创建示例输入张量
input_tensor = torch.rand(1, 16, 32, 320, 256)  # (batch_size, sequence_length, d_model)
image_tensor = torch.transpose(input_tensor,1,4)
new_shape = (1,2621440,16)
image_tensor = image_tensor.reshape(new_shape)

# 使用PositionalEncoding模块处理输入
output_tensor = pos_enc(image_tensor)

# 示例输入用例1
print("Input Tensor 1:")
print(input_tensor[0])  # 打印批次中的第一个序列
print("Output Tensor 1:")
print(output_tensor[0])  # 打印处理后的第一个序列

# 示例输入用例2
print("Input Tensor 2:")
print(input_tensor[1])  # 打印批次中的第二个序列
print("Output Tensor 2:")
print(output_tensor[1])  # 打印处理后的第二个序列

print(torch.zeros(10,4))