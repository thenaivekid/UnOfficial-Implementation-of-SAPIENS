"""
SAPIENS: Foundation for Human Vision Models

This is an unofficial implementation of the SAPIENS model from the paper:
"Sapiens: Foundation for Human Vision Models" (https://arxiv.org/abs/2408.12569)
"""

from .model import vit_base_patch16_1024, SapiensEncoder
from .utils import load_torchscript_model, load_pretrained_model, load_config

__version__ = "0.1.0"
