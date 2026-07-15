import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SiameseNet(nn.Module):
    """
    Siamese Network using ResNet18 backbone.
    Outputs L2-normalized embeddings for contrastive learning.
    """
    def __init__(self, embedding_dim=128):
        super().__init__()
        # ResNet18 backbone
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        base.fc = nn.Identity()  # Remove classification layer
        self.backbone = base

        # Projection head
        self.proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),  # optional but stabilizes training
            nn.Linear(256, embedding_dim)
        )

    def forward_once(self, x):
        feat = self.backbone(x)
        emb = self.proj(feat)
        emb = F.normalize(emb, p=2, dim=1)  # L2-normalize embeddings
        return emb

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)
