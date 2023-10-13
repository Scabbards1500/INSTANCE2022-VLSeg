import torch.nn as nn
import torch
import math

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(0.1)

        # 计算位置编码
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim))
        position_encoding = torch.zeros(1, max_seq_len, embed_dim)
        print("position",position.shape)
        print("div_term",div_term.shape)

        position_encoding[:, 0::2] = torch.sin(position * div_term)
        position_encoding[:, 1::2] = torch.cos(position * div_term)

        # 注册位置编码为模型的参数，使其可以与模型一起训练
        self.register_buffer('position_encoding', position_encoding)

    def forward(self, x):
        # 将位置编码添加到输入张量上，并应用dropout
        print("x", x)
        print("x",x.shape)
        print("x_emb",(self.position_encoding[:, :x.size(1), :]).shape)
        x = x + self.position_encoding[:, :x.size(1), :]
        x = self.dropout(x)
        return x


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
        self.vis_pos = PositionalEncoding(in_channels)
        self.txt_pos = PositionalEncoding(in_channels)
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        self.scale = nn.Parameter(torch.tensor(0.01), requires_grad=True)

    def forward(self, x, txt):
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
model = FusLanguageVision(in_channels=256, output_text_len=32, input_text_len=24, embed_dim=768)

# 创建示例输入
image_tensor = torch.randn(16, 32, 320, 256)
text_tensor = torch.randn(1, 24, 32)

# 运行模型
output = model(image_tensor, text_tensor)
