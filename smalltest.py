import numpy as np

def positional_encoding(d_model, max_len):
    position = np.arange(0, max_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pos_enc = np.zeros((max_len, d_model))
    pos_enc[:, 0::2] = np.sin(position * div_term)
    pos_enc[:, 1::2] = np.cos(position * div_term)
    return pos_enc

# 输入张量的形状
input_shape = (16, 32, 320, 256)

# 获取位置编码
max_len = input_shape[2]
d_model = input_shape[-1]
position_embedding = positional_encoding(d_model, max_len)

# 在输入张量的最后两维上添加位置嵌入
embedded_tensor = np.zeros(input_shape)
embedded_tensor[:, :, :, :, :d_model] += position_embedding

# 现在，embedded_tensor 包含了位置嵌入信息，其形状与输入张量相同。
