import torch

import torch.nn as nn
import math
from einops import rearrange


# PositionalEncoding是融合的时候再加的
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=81920):
        super(PositionalEncoding, self).__init__()
        self.encoding = self.generate_encoding(d_model, max_len)

    def generate_encoding(self, d_model, max_len):
        d_model = d_model
        encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        encoding = encoding.unsqueeze(0)
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
        self.cross_attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=4, batch_first=True)
        self.text_project = nn.Sequential(
            nn.Conv1d(input_text_len, output_text_len, kernel_size=1, stride=1),
            nn.GELU(),
            # nn.Linear(in_channels,embed_dim),
            nn.LeakyReLU(),
        )
        self.vis_pos = PositionalEncoding(d_model=in_channels)
        self.txt_pos = PositionalEncoding(d_model=in_channels, max_len=output_text_len)
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        self.scale = nn.Parameter(torch.tensor(0.01), requires_grad=True)

    def forward(self, x, txt):
        txt = self.text_project(txt) #txt=[1,24,32] -- [1,512,32]
        txt = txt.transpose(1,2) # txt = [1,32,512]

        vis2 = self.norm1(x) # x = {1,81920,512}
        q = self.vis_pos(vis2)
        k = self.vis_pos(vis2)
        output, attentionweights = self.self_attn(q, k, value=vis2)
        vis2 = self.self_attn_norm(output)
        vis = x + vis2

        # Cross-Attention
        vis2 = self.norm2(vis) # vis2 = {1,81920,512}
        query = self.vis_pos(vis2)
        key = self.txt_pos(txt)

        vis2,_ = self.cross_attn(query,
                                   key,
                                   value=txt)
        vis2 = self.cross_attn_norm(vis2)
        vis = vis + self.scale*vis2

        return vis


class FusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, text_len, embed_dim) -> None:
        super().__init__()
        self.guide_layer = FusLanguageVision(in_channels,out_channels,text_len,embed_dim)  # for skip

    def forward(self, vis, txt):
        originshape = vis.shape
        vis = torch.transpose(vis, 1, 4).reshape(
            (1, vis.shape[1] * vis.shape[2], vis.shape[4] * vis.shape[3])).transpose(1, 2)

        if txt is not None:
            vis = self.guide_layer(vis, txt)

        vis = torch.transpose(vis, 1, 2).reshape(originshape)

        return vis

