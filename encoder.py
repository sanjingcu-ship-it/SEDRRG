import torch
from torch import nn
import torchvision.models as models
from einops import rearrange
import torch.nn.functional as F
from vit_pytorch import ViT
from einops import repeat
class Encoder(nn.Module):
    def __init__(self, word_dim, from_x=-1):
        super(Encoder, self).__init__()
        self.from_x = from_x
        resnet = models.resnet34(pretrained=True)
        modules = list(resnet.children())[:from_x]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(512, word_dim)

    def forward(self, x):
        x = self.resnet(x).squeeze()  # (batch_size, enc_dim, enc_img_size, enc_img_size)
        if self.from_x != -1:
            x = rearrange(x, 'b d h w ->b (h w) d')
        # x = x.permute(0, 2, 3, 1)
        x = self.linear(x)
        return x

class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()

        # self.diffusion_pipeline = DiffusionPipeline.from_pretrained("CompVis/ldm-text2im-large-256")

        self.vit = Vito(image_size=224, patch_size=32, num_classes=1000, dim=256, depth=12, heads=12, mlp_dim=2048, dropout=0.1, emb_dropout=0.1)
        self.enc_dim = 768

    def forward(self, x):
        # x = self.diffusion_pipeline(x)["sample"]

        x = self.vit(x)
        return x


class Vito(ViT):
    def forward(self, img):
        # print("1",img.shape)
        x = self.to_patch_embedding(img)
        # print("2", x.shape)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        # print("3", x.shape)

        if n + 1 > self.pos_embedding.shape[1]:
            pos_embed = F.interpolate(
                self.pos_embedding.permute(0, 2, 1),
                size=n + 1,
                mode='linear'
            ).permute(0, 2, 1)
        else:
            pos_embed = self.pos_embedding[:, :(n + 1)]
        # print("4", x.shape)

        x += pos_embed
        # print("5", x.shape)
        x = self.dropout(x)
        # print("6", x.shape)
        return self.transformer(x)
# class EnhancedEncoder(nn.Module):
#     def __init__(self):
#         self.backbone = models.swin_b(pretrained=True)
#         self.proj = nn.Linear(1024, word_dim)
#
#     def forward(self, x):
#         features = self.backbone(x).last_hidden_state
#         return self.proj(features[:, 0])  # CLS token

import torch
import torch.nn as nn
import torchvision.models as models
from einops import rearrange

import torch
import torch.nn as nn
import torchvision.models as models
from einops import rearrange


class ImprovedEncoder(nn.Module):
    def __init__(self, word_dim, from_x=-1, use_pretrained=True):
        super(ImprovedEncoder, self).__init__()
        self.from_x = from_x

        resnet = models.resnet50(pretrained=use_pretrained)
        modules = list(resnet.children())[:from_x]
        self.resnet = nn.Sequential(*modules)

        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2048, 2048 // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(2048 // 16, 2048, kernel_size=1),
            nn.Sigmoid()
        )

        self.conv1x1 = nn.Conv2d(2048, word_dim, kernel_size=1)

        self.linear = nn.Sequential(
            nn.Linear(word_dim, word_dim),
            nn.GELU(),
            nn.Linear(word_dim, word_dim)
        )

    def forward(self, x):
        x = self.resnet(x)  # (batch_size, 2048, h, w)

        se_weights = self.se_block(x)
        x = x * se_weights

        x = self.conv1x1(x)  # (batch_size, word_dim, h, w)

        if self.from_x != -1:
            x = rearrange(x, 'b d h w -> b (h w) d')  # (batch_size, h*w, word_dim)

        x = self.linear(x)  # (batch_size, h*w, word_dim)

        return x



if __name__ == '__main__':
    encoder = Encoder(256, -2)
    img = torch.randn(32, 3, 256, 256)
    out = encoder(img)
    print(out.shape)

