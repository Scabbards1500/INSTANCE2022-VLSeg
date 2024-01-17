import torch.nn as nn
import torch
import math

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


class FusLanguageVision(nn.Module):
    def __init__(self, in_channels:int, output_text_len:int, input_text_len:int=24, embed_dim:int=768):
        super(FusLanguageVision, self).__init__()
        self.in_channels = in_channels
        self.self_attn_norm = nn.LayerNorm(in_channels)
        self.cross_attn_norm = nn.LayerNorm(in_channels)
        self.self_attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=1, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=4, batch_first=True)
        self.text_project = nn.Sequential(
            nn.Conv1d(input_text_len, output_text_len, kernel_size=1, stride=1),
            nn.GELU(),
            # nn.Linear(embed_dim, in_channels),
            nn.LeakyReLU(),
        )
        self.vis_pos = PositionalEncoding(d_model=in_channels)
        self.txt_pos = PositionalEncoding(d_model=in_channels)
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        self.scale = nn.Parameter(torch.tensor(0.01), requires_grad=True)

    def forward(self, x, txt):
        print("inchannels", self.in_channels)
        txt = self.text_project(txt)  # Permute to [batch_size, embed_dim, input_text_len]
        # Self-Attention
        vis2 = self.norm1(x)
        print("vis2", vis2.shape)
        q = k = self.vis_pos(vis2)
        vis2, _ = self.self_attn(q, k, value=vis2)
        vis2 = self.self_attn_norm(vis2)
        vis = x + vis2

        # Cross-Attention
        vis2 = self.norm2(vis)
        vis2, _ = self.cross_attn(query=self.vis_pos(vis2),
                                  key=self.txt_pos(txt),
                                  value=txt)
        vis2 = self.cross_attn_norm(vis2)
        vis = vis + self.scale * vis2

        return vis

# 测试代码
import torch

# 创建模型
model = FusLanguageVision(in_channels=16, output_text_len=32, input_text_len=16, embed_dim=768)

# 创建示例输入
image_tensor = torch.randn(1, 16, 32, 320, 256)
image_tensor = torch.transpose(image_tensor,1,4)
new_shape = (1,2621440,16)
image_tensor = image_tensor.reshape(new_shape)

text_tensor = torch.randn(1, 16, 16)

# 运行模型
output = model(image_tensor, text_tensor)
