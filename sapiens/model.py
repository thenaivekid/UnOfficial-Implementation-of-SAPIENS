import torch
import torch.nn as nn
from .modules import PatchEmbedding, TransformerBlock

class SapiensEncoder(nn.Module):
    """SAPIENS Vision Transformer Encoder.
    
    Implements the full encoder part of the SAPIENS model as described in the paper.
    
    Attributes:
        patch_embed: Patch embedding layer
        cls_token: Learnable classification token
        pos_embed: Positional embedding
        pos_drop: Dropout for positional embedding
        layers: Transformer blocks
        ln1: Final layer normalization
    """
    def __init__(self, img_size=1024, patch_size=16, in_chans=3, embed_dim=1024, depth=24, num_heads=16, dropout_rate=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim, dropout_rate)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(dropout_rate)

        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, 4.0, dropout_rate) for _ in range(depth)
        ])
        self.ln1 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.layers:
            x = block(x)

        x = self.ln1(x)
        return x

def vit_base_patch16_1024(dropout_rate=0.1):
    """Factory function to create a SAPIENS Vision Transformer model.
    
    Args:
        dropout_rate: Dropout rate for all dropout layers in the model
        
    Returns:
        A SapiensEncoder model instance
    """
    model = SapiensEncoder(
        img_size=1024,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        dropout_rate=dropout_rate,
    )
    return model
