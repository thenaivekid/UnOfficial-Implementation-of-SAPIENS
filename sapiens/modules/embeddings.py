import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """Vision Transformer Patch Embedding layer.
    
    This layer converts an image into a sequence of patch embeddings as used in Vision Transformers.
    
    Attributes:
        img_size: Input image size
        patch_size: Patch size for dividing the image
        grid_size: Number of patches along one dimension
        num_patches: Total number of patches
        projection: Convolutional projection layer
        dropout: Dropout layer for regularization
    """
    def __init__(self, img_size=1024, patch_size=16, in_chans=3, embed_dim=1024, dropout_rate=0.1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.projection = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.projection(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)  # (B, embed_dim, N)
        x = x.transpose(1, 2)  # (B, N, embed_dim)
        x = self.dropout(x)
        return x
