import torch
import yaml

def load_torchscript_model(path, device="cuda"):
    """Load a TorchScript model.
    
    Args:
        path: Path to the TorchScript model file
        device: Device to load the model on ('cuda' or 'cpu')
        
    Returns:
        Loaded TorchScript model on the specified device
    """
    ts_model = torch.jit.load(path)
    ts_model.to(device)
    return ts_model

def load_pretrained_model(model, checkpoint_path, device="cuda"):
    """Load pretrained weights into a model.
    
    Args:
        model: Model instance to load weights into
        checkpoint_path: Path to the checkpoint file
        device: Device to load the weights on ('cuda' or 'cpu')
        
    Returns:
        Model with loaded weights
    """
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    return model

def load_config(config_path):
    """Load a YAML configuration file.
    
    Args:
        config_path: Path to the YAML config file
        
    Returns:
        Dictionary containing the configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
