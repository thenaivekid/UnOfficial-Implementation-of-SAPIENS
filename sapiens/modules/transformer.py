import torch
import torch.nn as nn
from .attention import Attention
from .feedforward import FFN

class TransformerBlock(nn.Module):
    """Transformer Block.
    
    A single block of the Transformer architecture, consisting of self-attention and feedforward network.
    
    Attributes:
        ln1, ln2: Layer normalization for attention and feedforward
        attn: Self-attention module
        ffn: Feed forward network
        drop_path: Dropout for residual connection
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout_rate=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads, dropout_rate)
        self.ln2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.ffn = FFN(embed_dim, hidden_dim, dropout_rate)
        self.drop_path = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.ln1(x)))
        x = x + self.drop_path(self.ffn(self.ln2(x)))
        return x
