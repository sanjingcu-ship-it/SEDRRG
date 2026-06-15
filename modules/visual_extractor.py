import torch
import torch.nn as nn
import torchvision.models as models
import math
import torch.nn.functional as F

from swin_transformer_v2 import SwinTransformerV2



class VisualExtractor(nn.Module):##############SwinTransformerV2#######
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.use_lesion_mask = getattr(args, "use_lesion_mask", False)
        self.lesion_alpha = getattr(args, "lesion_alpha", 0.5)
        self.mask_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.visual_extractor_name = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained

        if self.visual_extractor_name == 'swim_transformer_v2':
            full_model = SwinTransformerV2(pretrained=self.pretrained)

            self.patch_embed = full_model.patch_embed
            self.layers = full_model.layers
            self.norm = full_model.norm

        else:
            raise NotImplementedError

    def forward(self, images):
        x = self.patch_embed(images)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        patch_feats = x  # [B, N, C]
        B, N, C = patch_feats.shape

        # --------- Optional lesion prior mask M_expand ----------
        if self.use_lesion_mask:
            side = int(math.sqrt(N))
            if side * side == N:
                # 1) token saliency: L2-norm over channels  -> [B, N]
                sal = patch_feats.detach().norm(dim=-1)

                # 2) min-max normalize per sample -> [B, N] in [0,1]
                sal_min = sal.amin(dim=1, keepdim=True)
                sal_max = sal.amax(dim=1, keepdim=True)
                sal = (sal - sal_min) / (sal_max - sal_min + 1e-6)

                # 3) expand by 2D max-pooling on patch grid
                m = sal.view(B, 1, side, side)  # [B,1,H,W]
                m_expand = self.mask_pool(m).view(B, N, 1)  # [B,N,1]

                # 4) lesion-aware modulation (applied on normalized patch tokens)
                patch_feats = patch_feats * (1.0 + self.lesion_alpha * m_expand)

        avg_feats = patch_feats.mean(dim=1)
        return patch_feats, avg_feats

# class VisualExtractor(nn.Module): ###########resnet101###########
#     def __init__(self, args):
#         super(VisualExtractor, self).__init__()
#         self.visual_extractor = args.visual_extractor
#         self.pretrained = args.visual_extractor_pretrained
#         model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
#         modules = list(model.children())[:-2]
#         self.model = nn.Sequential(*modules)
#         self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
#
#     def forward(self, images):
#         patch_feats = self.model(images)
#         avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
#         batch_size, feat_size, _, _ = patch_feats.shape
#         patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
#         return patch_feats, avg_feats
