"""
This script defines the DINOv2 model architecture and downloads the pretrained DINOv2 backbone. 
A custom segmentation head is added to the backbone to perform semantic segmentation on the 
Cityscapes dataset.

Based on the DINOv2 architecture from the original paper:
Maxime Oqua et al. (2021), "DINOv2: Learning Robust Visual Features without Supervision", 
https://arxiv.org/abs/2304.07193

Additional inspiration from the following blog post:
Sovit Ranjan Rath (2025) "DINOv2 Segmentation – Fine-Tuning and Transfer Learning Experiments"
https://debuggercafe.com/dinov2-segmentation-fine-tuning-and-transfer-learning-experiments/
"""

import torch
import torch.nn as nn

from functools import partial

## CHANGE ACCORDING TO THE ViT MODEL YOU ARE USING
# ------------------------------------------------
from config_vits14 import model as model_dict
# from config_vitb14 import model as model_dict 
# from config_vitl14 import model as model_dict
# from config_vitg14 import model as model_dict

def load_backbone(backbone_size="small"): # "small", "base", "large", "giant"
    """
    Load the DINOv2 backbone model from Facebook Research's repository."
    """
    backbone_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }

    backbone_arch = backbone_archs[backbone_size]
    backbone_name = f"dinov2_{backbone_arch}"

    backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    # backbone_model.cuda()

    backbone_model.forward = partial(
        backbone_model.get_intermediate_layers,
        n=model_dict['backbone']['out_indices'],
        reshape=True,
    )

    return backbone_model


class ASPPHead(nn.Module):
    def __init__(self, in_channels, num_classes, hidden_dim=256):
        super().__init__()
        self.H = 46
        self.W = 46
        
        self.aspp = nn.ModuleList([
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 3, padding=6, dilation=6),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 3, padding=12, dilation=12),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, hidden_dim, 1),
                nn.ReLU()
            )
        ])
        
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim * 4, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(hidden_dim, num_classes, 1)
        )

    def forward(self, x):
        x = x.reshape(-1, x.shape[1], self.H, self.W)
        res = []
        for aspp_module in self.aspp:
            feat = aspp_module(x)
            if feat.size() != x.size():
                feat = nn.functional.interpolate(
                    feat, size=x.size()[2:], mode='bilinear', align_corners=False
                )
            res.append(feat)
        
        x = torch.cat(res, dim=1)
        x = self.project(x)
        
        return nn.functional.interpolate(
            x, size=(self.H*14, self.W*14), 
            mode='bilinear', 
            align_corners=False
        )

class LinearClassifierToken(torch.nn.Module):
    def __init__(self, in_channels, nc=1, tokenW=32, tokenH=32):
        super(LinearClassifierToken, self).__init__()
        self.in_channels = in_channels
        self.W = tokenW
        self.H = tokenH
        self.nc = nc
        self.conv = torch.nn.Conv2d(in_channels, nc, (1, 1))

    def forward(self,x):
        outputs =  self.conv(
            x.reshape(-1, self.in_channels, self.H, self.W)
        )
        
        upsampled_logits = nn.functional.interpolate(
            outputs, size=(self.H*14, self.W*14), 
            mode='bilinear', 
            align_corners=False
        )
        
        return upsampled_logits


class DINOv2Segmentation(nn.Module):
    def __init__(self, num_classes=19, fine_tune=False):
        super(DINOv2Segmentation, self).__init__()

        self.backbone_model = load_backbone()
        print("Backbone model loaded successfully.")

        if fine_tune:
            for name, param in self.backbone_model.named_parameters():
                param.requires_grad = True
        else:
            for name, param in self.backbone_model.named_parameters():
                param.requires_grad = False

        # self.decode_head = LinearClassifierToken(in_channels=1536, nc=num_classes, tokenW=46, tokenH=46)
        self.decode_head = ASPPHead(in_channels=1536, num_classes=num_classes)

    def forward(self, x):
        features = self.backbone_model(x)

        # `features` is a tuple.
        concatenated_features = torch.cat(features, 1)

        classifier_out = self.decode_head(concatenated_features)

        return classifier_out
    
# if __name__ == '__main__':
#     model = DINOv2Segmentation()
#     from torchinfo import summary
#     summary(
#         model, 
#         (1, 3, 644, 644),
#         col_names=('input_size', 'output_size', 'num_params'),
#         row_settings=['var_names']
#     )