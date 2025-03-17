import torch
import torch.nn as nn
from torch.nn import functional as F

class DinoSegmentationModel(nn.Module):
    def __init__(self, n_classes=19, backbone_size='small'):
        super(DinoSegmentationModel, self).__init__()
        
        # Load pre-trained DINOv2 backbone
        backbone_archs = {
            "small": "vits14",
            "base": "vitb14",
            "large": "vitl14",
            "giant": "vitg14",
        }

        backbone_arch = backbone_archs[backbone_size]
        backbone_name = f"dinov2_{backbone_arch}"
        
        # Load DINOv2 backbone
        self.backbone = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
        
        # Freeze backbone weights
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Get embedding dimension based on model type
        embed_dims = {
            'small': 384,
            'base': 768,
            'large': 1024,
            'giant': 1536
        }
        embed_dim = embed_dims[backbone_size]
        
        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, n_classes, kernel_size=1)
        )
        
    def forward(self, x):
        # Get input dimensions
        B, C, H, W = x.shape
        
        # Get DINO features
        features = self.backbone.forward_features(x)  # [B, N, D]
        
        # Remove CLS token and reshape to 2D
        features = features[:, 1:, :]  # Remove CLS token
        
        # Reshape patches back to image-like format
        P = int((H // 16) * (W // 16))  # Number of patches
        D = features.shape[-1]  # Embedding dimension
        features = features.reshape(B, H//16, W//16, D)
        features = features.permute(0, 3, 1, 2)  # [B, D, H//16, W//16]
        
        # Upsample to original resolution
        features = F.interpolate(
            features, 
            size=(H, W), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Apply segmentation head
        logits = self.seg_head(features)
        
        return logits

def create_dino_segmentation_model(n_classes=19, backbone_size='small'):
    """
    Create a DINOv2-based segmentation model
    
    Args:
        n_classes (int): Number of output classes
        model_type (str): Size of DINOv2 model ('small', 'base', 'large', 'giant')
    """
    model = DinoSegmentationModel(n_classes=n_classes, backbone_size=backbone_size)
    return model
