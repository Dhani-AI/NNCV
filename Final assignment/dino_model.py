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
from collections import OrderedDict

from config_vits14 import model as model_dict

def load_backbone():
    BACKBONE_SIZE = "small" # in ("small", "base", "large" or "giant")

    backbone_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }
    backbone_arch = backbone_archs[BACKBONE_SIZE]
    backbone_name = f"dinov2_{backbone_arch}"


    backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    backbone_model.cuda()

    backbone_model.forward = partial(
        backbone_model.get_intermediate_layers,
        n=model_dict['backbone']['out_indices'],
        reshape=True,
    )

    return backbone_model


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
        return outputs


class DINOv2Segmentation(nn.Module):
    def __init__(self, fine_tune=False):
        super(DINOv2Segmentation, self).__init__()

        self.backbone_model = load_backbone()
        print("Backbone model loaded successfully.")
        if fine_tune:
            for name, param in self.backbone_model.named_parameters():
                param.requires_grad = True
        else:
            for name, param in self.backbone_model.named_parameters():
                param.requires_grad = False

        self.decode_head = LinearClassifierToken(in_channels=1536, nc=19, tokenW=46, tokenH=46)

    def forward(self, x):
        features = self.backbone_model(x)

        # `features` is a tuple.
        concatenated_features = torch.cat(features, 1)

        classifier_out = self.decode_head(concatenated_features)

        return classifier_out