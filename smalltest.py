import numpy as np

# 输入数据的维度
sequence_length = 64  # 序列长度
input_dimension = 128  # 输入维度

# 创建位置编码矩阵
def create_positional_embedding(sequence_length, input_dimension):
    position = np.arange(0, sequence_length)[:, np.newaxis]
    div_term = np.exp(np.arange(0, input_dimension, 2) * -(np.log(10000.0) / input_dimension))
    positional_embedding = np.zeros((sequence_length, input_dimension))

    # 奇数索引使用正弦函数编码
    positional_embedding[:, 0::2] = np.sin(position * div_term)

    # 偶数索引使用余弦函数编码
    positional_embedding[:, 1::2] = np.cos(position * div_term)

    return positional_embedding

# 创建一个示例位置编码
positional_embedding = create_positional_embedding(sequence_length, input_dimension)

print(positional_embedding.shape)
print(positional_embedding)
