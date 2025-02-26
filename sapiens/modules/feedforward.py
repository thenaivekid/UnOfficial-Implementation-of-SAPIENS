import torch
import torch.nn as nn

class FFN(nn.Module):
    """Feed Forward Network module.
    
    Implements the feedforward network from the Transformer architecture.
    
    Attributes:
        layers: ModuleList containing the linear layers
        act: Activation function (GELU)
        drop1, drop2: Dropout layers for regularization
    """
    def __init__(self, in_dim, hidden_dim, dropout_rate=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(in_dim, hidden_dim)  # This will be layers.0.0
            ]),
            nn.Linear(hidden_dim, in_dim)      # This will be layers.1
        ])
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout_rate)
        self.drop2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.layers[0][0](x)  # Apply first linear layer (layers.0.0)
        x = self.act(x)
        x = self.drop1(x)
        x = self.layers[1](x)     # Apply second linear layer (layers.1)
        x = self.drop2(x)
        return x
