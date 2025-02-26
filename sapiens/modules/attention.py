import torch
import torch.nn as nn

class Attention(nn.Module):
    """Multi-head Self-Attention module.
    
    Implements the multi-headed self-attention mechanism from the Transformer architecture.
    
    Attributes:
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head
        scale: Scaling factor for dot-product attention
        qkv: Linear projection for query, key, and value
        proj: Output projection
        attn_drop: Dropout applied to attention weights
        proj_drop: Dropout applied to output projections
    """
    def __init__(self, embed_dim, num_heads, dropout_rate=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
