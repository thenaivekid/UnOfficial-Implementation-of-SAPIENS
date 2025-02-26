import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from sapiens import vit_base_patch16_1024, load_torchscript_model, load_pretrained_model

def preprocess_image(image_path, img_size=1024):
    """Preprocess image for inference."""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    return img_tensor

def main():
    parser = argparse.ArgumentParser(description="Run inference using SAPIENS model")
    parser.add_argument('--model_type', choices=['native', 'torchscript'], default='native', 
                        help='Type of model to use: native PyTorch or TorchScript')
    parser.add_argument('--model_path', required=True,
                        help='Path to model weights (native) or TorchScript model')
    parser.add_argument('--image_path', required=True,
                        help='Path to input image')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                        help='Device to run inference on')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    
    # Process input image
    img_tensor = preprocess_image(args.image_path).to(device)
    print(f"Image shape: {img_tensor.shape}")
    
    if args.model_type == 'native':
        # Load native PyTorch model
        print(f"Loading native PyTorch model from {args.model_path}")
        model = vit_base_patch16_1024()
        model = load_pretrained_model(model, args.model_path)
        model.eval()
        model.to(device)
        
        with torch.no_grad():
            
            outputs = model(img_tensor)
            print(f"Native model output shape: {outputs.shape}")
            print(f"CLS token features (first 5): {outputs[0, 0, :5]}")
            
    else:  # TorchScript model
        # Load TorchScript model
        print(f"Loading TorchScript model from {args.model_path}")
        ts_model = load_torchscript_model(args.model_path)
        ts_model.eval()
        ts_model.to(device)
        with torch.no_grad():
            outputs = ts_model(img_tensor)
            print(f"TorchScript model output shape: {outputs.shape}")
            print(f"CLS token features (first 5): {outputs[0, 0, :5]}")
    
    print("Inference completed successfully!")

if __name__ == "__main__":
    main()
