import torch.nn as nn
from collections import OrderedDict
import torch
from .BioBert import BERTModel
from .Blocks import FusLanguageVision,FusionBlock



class UNet3d(nn.Module):
    """
    Unet3d implement
    """

    def __init__(self, in_channels, out_channels, bert_type, init_features=16):
        super(UNet3d, self).__init__()
        self. text_encoder = BERTModel(bert_type, 512)
        self.features = init_features
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.text_size = text_size
        self.text_module4 = nn.Conv1d(in_channels=768, out_channels=512, kernel_size=3, padding=1)
        self.text_module3 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.text_module2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.text_module1 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.text_module0 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)

        self.encoder1 = UNet3d._block(self.in_channels, self.features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = UNet3d._block(self.features, self.features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = UNet3d._block(self.features * 2, self.features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = UNet3d._block(self.features * 4, self.features * 8, name="enc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.fusion1 = FusionBlock(self.features *32, self.features*32, 24, self.features)
        self.fusion2 = FusionBlock(self.features * 32, self.features * 32,24,self.features * 2)
        self.fusion3 = FusionBlock(self.features * 32, self.features * 32,24,self.features * 4)
        self.fusion4 = FusionBlock(self.features * 32, self.features * 32, 24,self.features * 8)


        self.bottleneck = UNet3d._block(self.features * 8, self.features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose3d(self.features * 16, self.features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet3d._block((self.features * 8) * 2, self.features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose3d(self.features * 8, self.features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet3d._block((self.features * 4) * 2, self.features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose3d(self.features * 4, self.features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet3d._block((self.features * 2) * 2, self.features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose3d(self.features * 2, self.features, kernel_size=2, stride=2)
        self.decoder1 = UNet3d._block(self.features * 2, self.features, name="dec1")
        self.conv = nn.Conv3d(in_channels=self.features, out_channels=self.out_channels, kernel_size=1)



    def forward(self, x, text):

        text_output = self.text_encoder(text['input_ids'], text['attention_mask'])
        text_embeds, text_project = text_output['feature'], text_output['project']  # [1, 24, 768], [1,256]

        txt_pre = self.text_module4(text_embeds[-1].transpose(1, 2)).transpose(1, 2)  # [1, 24, 512] 其中1是batch_size，24是句子长度，512是词向量维度
        txt4 = self.text_module3(txt_pre.transpose(1, 2)).transpose(1, 2)  # [1, 24, 256]
        txt3 = self.text_module2(txt4.transpose(1, 2)).transpose(1, 2)  # [1, 24, 128]
        txt2 = self.text_module1(txt3.transpose(1, 2)).transpose(1, 2)  # [1, 24, 64]
        txt1 = self.text_module0(txt2.transpose(1, 2)).transpose(1, 2)  # [1, 24, 32]


        enc1 = self.encoder1(x)  # [1, 16, 32, 320, 256] # batchsize，channel，depth，height，width
        enc2 = self.encoder2(self.pool1(enc1))  # [1, 32, 16, 160, 128]
        enc3 = self.encoder3(self.pool2(enc2))  # [1, 64, 8, 80, 64]
        enc4 = self.encoder4(self.pool3(enc3))  # [1, 128, 4, 40, 32]


        #这里文本和图像开始融合，到时候再一起送入decoder中

        fus1 = self.fusion1(enc1, txt1)
        fus2 = self.fusion2(enc2, txt2)
        fus3 = self.fusion3(enc3, txt3)
        fus4 = self.fusion4(enc4, txt4)



        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck) #[1, 128, 4, 40, 32]
        dec4 = torch.cat((dec4, fus4), dim=1)# 通道拼接 [1, 256, 4, 40, 32]
        dec4 = self.decoder4(dec4) # [1, 128, 4, 40, 32]

        dec3 = self.upconv3(dec4) # [1, 64, 8, 80, 64]
        dec3 = torch.cat((dec3, fus3), dim=1)# 通道拼接 [1, 128, 8, 80, 64]
        dec3 = self.decoder3(dec3) # [1, 64, 8, 80, 64]

        dec2 = self.upconv2(dec3) # [1, 32, 16, 160, 128]
        dec2 = torch.cat((dec2, fus2), dim=1)# 通道拼接，这里塞入文本信息
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2) # [1, 16, 32, 320, 256]
        dec1 = torch.cat((dec1, fus1), dim=1)  # 通道拼接 [1, 32, 32, 320, 256]
        dec1 = self.decoder1(dec1) # [1, 16, 32, 320, 256]
        out_logit = self.conv(dec1)

        if self.out_channels == 1:
            output = torch.sigmoid(out_logit)
        if self.out_channels > 1:
            output = torch.softmax(out_logit, dim=1)
        return out_logit, output

    @staticmethod
    def _block(in_channels, features, name, prob=0.2):
        block = nn.Sequential(OrderedDict([
            (name + "conv1", nn.Conv3d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False, ),),
            (name + "norm1", nn.GroupNorm(num_groups=8, num_channels=features)),
            (name + "droupout1", nn.Dropout3d(p=prob, inplace=True)),
            (name + "relu1", nn.ReLU(inplace=True)),
            (name + "conv2", nn.Conv3d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False, ),),
            (name + "norm2", nn.GroupNorm(num_groups=8, num_channels=features)),
            (name + "droupout2", nn.Dropout3d(p=prob, inplace=True)),
            (name + "relu2", nn.ReLU(inplace=True)),
        ]))
        return block


# fusion_net = UNet3d(in_channels=1, out_channels=1)