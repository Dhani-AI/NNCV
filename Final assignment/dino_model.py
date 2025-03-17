import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F

class DinoSegmentationModel(nn.Module):
    def __init__(self, n_classes=19):
        super(DinoSegmentationModel, self).__init__()
        
        # Load pre-trained DINO model
        self.backbone = models.vit_s16(weights='DINO-s16')
        
        # Freeze backbone weights
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Get embedding dimension from backbone
        embed_dim = self.backbone.hidden_dim  # typically 384 for ViT-S/16
        
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

def create_dino_segmentation_model(n_classes=19):
    model = DinoSegmentationModel(n_classes=n_classes)
    return model
