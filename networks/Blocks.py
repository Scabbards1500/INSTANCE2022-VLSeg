import torch

import torch.nn as nn
import math



# PositionalEncoding是融合的时候再加的
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2621440):
        super(PositionalEncoding, self).__init__()
        self.encoding = self.generate_encoding(d_model, max_len)

    def generate_encoding(self, d_model, max_len):
        encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        encoding = encoding.unsqueeze(0)
        print("通过")
        return encoding

    def forward(self, x):
        pos_s = x + self.encoding[:, :x.size(1)].to(x.device)
        return pos_s
    # 目前到这一步都是没问题的



class FusLanguageVision(nn.Module):
    def __init__(self, in_channels:int, output_text_len:int, input_text_len:int, embed_dim:int=768):
        super(FusLanguageVision, self).__init__()
        self.in_channels = in_channels
        self.self_attn_norm = nn.LayerNorm(in_channels)
        self.cross_attn_norm = nn.LayerNorm(in_channels)
        self.self_attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=1)
        self.cross_attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=4)
        self.text_project = nn.Sequential(
            nn.Conv1d(input_text_len, output_text_len, kernel_size=1, stride=1),
            nn.GELU(),
            # nn.Linear(embed_dim, in_channels),
            nn.LeakyReLU(),
        )
        self.vis_pos = PositionalEncoding(d_model=in_channels)
        self.txt_pos = PositionalEncoding(d_model=in_channels, max_len=output_text_len)
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        self.scale = nn.Parameter(torch.tensor(0.01), requires_grad=True)

    def forward(self, x, txt): #x torch.Size([1, 16, 32, 320, 256])
        x = torch.transpose(x,1,4)
        txt = self.text_project(txt) #txt=[1,24,16]
        # Self-Attention
        shape = x.shape
        new_shape = (1, shape[0] * shape[1] * shape[2] * shape[3], 16)
        x = x.reshape(new_shape)
        vis2 = self.norm1(x) #vis2=[1, 2621440, 16]
        q = self.vis_pos(vis2) #q=[1, 2621440, 16]
        k = self.vis_pos(vis2) #k=[1, 2621440, 16]
        output, attentionweights = self.self_attn(q, k, value=vis2) #output=[1, 2621440, 16]
        print("vis2------ok")
        vis2 = self.self_attn_norm(output) #vis2=[1, 2621440, 16]
        vis = x + vis2 #vis=[1, 2621440, 16]
        print("yeahhhhhhhhhh")

        # Cross-Attention
        vis2 = self.norm2(vis) #vis2=[1, 2621440, 16]
        key = self.txt_pos(txt)
        print("key",key.shape)
        vis2,_ = self.cross_attn(query=self.vis_pos(vis2),
                                   key=self.txt_pos(txt),
                                   value=txt)
        vis2 = self.cross_attn_norm(vis2)
        vis = vis + self.scale*vis2

        return vis

