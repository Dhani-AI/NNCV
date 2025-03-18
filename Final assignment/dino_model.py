"""
This script defines the DINOv2 model architecture and downloads the pretrained DINOv2 backbone. 
A custom segmentation head is added to the backbone to perform semantic segmentation on the 
Cityscapes dataset.

Based on the DINOv2 architecture from the original paper:
Maxime Oqua et al. (2021), "DINOv2: Learning Robust Visual Features without Supervision", 
https://arxiv.org/abs/2304.07193
"""

import torch
import urllib
import warnings
import torch.nn.functional as F
import torch.nn as nn

from functools import partial
from collections import OrderedDict

BACKBONE_SIZE = "small" # in ("small", "base", "large" or "giant")

backbone_archs = {
    "small": "vits14",
    "base": "vitb14",
    "large": "vitl14",
    "giant": "vitg14",
}
backbone_arch = backbone_archs[BACKBONE_SIZE]
backbone_name = f"dinov2_{backbone_arch}"
    
HEAD_SCALE_COUNT = 3 # more scales: slower but better results, in (1,2,3,4,5)
HEAD_DATASET = "voc2012" # in ("ade20k", "voc2012")
HEAD_TYPE = "ms" # in ("ms, "linear")

DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
head_config_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_config.py"
head_checkpoint_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_head.pth"

backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
backbone_model.cuda()

# TODO: This part needs cleaning #
def load_config_from_url(url: str) -> str:
    with urllib.request.urlopen(url) as f:
        return f.read().decode()
    
cfg_str = load_config_from_url(head_config_url)
with open('temp.py', 'w') as f:
    f.write(cfg_str)

from temp import model as model_dict
##################################

backbone_model.forward = partial(
    backbone_model.get_intermediate_layers,
    n=model_dict['backbone']['out_indices'],
    reshape=True,
)

for name, param in backbone_model.named_parameters():
    param.requires_grad = False

def resize(input_data,
       size=None,
       scale_factor=None,
       mode='nearest',
       align_corners=None,
       warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > input_w:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input_data, size, scale_factor, mode, align_corners)

class BNHead(nn.Module):
    """Just a batchnorm."""

    def __init__(self, resize_factors=None, **kwargs):
        super().__init__(**kwargs)
        # HARDCODED IN_CHANNELS FOR NOW.
        self.in_channels = 1536
        self.bn = nn.SyncBatchNorm(self.in_channels)
        self.resize_factors = resize_factors
        self.in_index = [0, 1, 2, 3]
        self.input_transform = 'resize_concat'
        self.align_corners = False

        self.conv_seg = nn.Conv2d(self.in_channels, 21, kernel_size=1)

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        x = self._transform_inputs(inputs)
        feats = self.bn(x)
        return feats

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == "resize_concat":
            # accept lists (for cls token)
            input_list = []
            for x in inputs:
                if isinstance(x, list):
                    input_list.extend(x)
                else:
                    input_list.append(x)
            inputs = input_list
            # an image descriptor can be a local descriptor with resolution 1x1
            for i, x in enumerate(inputs):
                if len(x.shape) == 2:
                    inputs[i] = x[:, :, None, None]
            # select indices
            inputs = [inputs[i] for i in self.in_index]
            # Resizing shenanigans
            # print("before", *(x.shape for x in inputs))
            if self.resize_factors is not None:
                assert len(self.resize_factors) == len(inputs), (len(self.resize_factors), len(inputs))
                inputs = [
                    resize(input=x, scale_factor=f, mode="bilinear" if f >= 1 else "area")
                    for x, f in zip(inputs, self.resize_factors)
                ]
                # print("after", *(x.shape for x in inputs))
            upsampled_inputs = [
                resize(input_data=x, size=inputs[0].shape[2:], mode="bilinear", align_corners=self.align_corners)
                for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == "multiple_select":
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def cls_seg(self, feat):
        """Classify each pixel."""
        output = self.conv_seg(feat)
        return output
        
    
    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output
    
class DINOv2Segmentation(nn.Module):
    def __init__(self):
        super(DINOv2Segmentation, self).__init__()

        self.backbone_model = backbone_model
        self.decode_head = BNHead()

        self.model = nn.Sequential(OrderedDict([
            ('backbone', self.backbone_model),
            ('decode_head', self.decode_head)
        ]))

    def forward(self, x):
        outputs = self.model(x)
        return outputs
    
if __name__ == '__main__':
    from torchinfo import summary
    model = DINOv2Segmentation()
    summary(
        model, 
        (1, 3, 644, 644),
        col_names=('input_size', 'output_size', 'num_params'),
        row_settings=['var_names']
    )