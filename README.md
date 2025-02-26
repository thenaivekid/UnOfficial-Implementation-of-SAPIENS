# UnOfficial Implementation of SAPIENS
Implementation of paper: Sapiens: Foundation for Human Vision Models  
https://arxiv.org/abs/2408.12569

## Overview
This repository demonstrates:
- Pytorch implementation of SAPIENS model and shows how to load the pretrained weights
- Loading and running a pretrained TorchScript model.
- A training loop for fine-tuning.


## Installation
```bash
pip install -r requirements.txt
```

## Usage
### Inference
Download the pretrained weights:
```bash
wget https://huggingface.co/facebook/sapiens-pretrain-0.3b-torchscript/resolve/main/sapiens_0.3b_epoch_1600_torchscript.pt2
wget https://huggingface.co/facebook/sapiens-pretrain-0.3b/resolve/main/sapiens_0.3b_epoch_1600_clean.pth
```
Run inference with either model type:
```bash
# Using native PyTorch model
python run_inference.py --model_type native --model_path sapiens_0.3b_epoch_1600_clean.pth --image_path path/to/image.jpg

# Using TorchScript model
python run_inference.py --model_type torchscript --model_path sapiens_0.3b_epoch_1600_torchscript.pt2 --image_path path/to/image.jpg
```

### Fine-Tuning
To fine-tune the model on your own dataset:

1. Organize your dataset in the following structure:
```
data/
├── train/
│   ├── class1/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── class2/
│       ├── img3.jpg
│       └── img4.jpg
└── val/
    ├── class1/
    │   └── img5.jpg
    └── class2/
        └── img6.jpg
```

2. Create or modify a config file:
```bash
# Use the default config
cp configs/default_config.yaml configs/my_config.yaml
# Edit the config to match your dataset and requirements
```

3. Run the fine-tuning script:
```bash
python train.py --config configs/my_config.yaml --pretrained sapiens_0.3b_epoch_1600_clean.pth
```

4. Resume training from a checkpoint:
```bash
python train.py --config configs/my_config.yaml --resume checkpoints/sapiens_finetuned/checkpoint_epoch_10.pt
```

## Model Modifications
The training script automatically adds a classification head to the pretrained SAPIENS model, which extracts features from the CLS token and projects them to the number of classes in your dataset.

## Monitoring Training
During training, you'll see updates on:
- Training loss and accuracy for each epoch
- Validation loss and accuracy after each epoch
- Best model checkpoint saving
- Learning rate scheduling

## Demo
[Watch the demo](https://video.fktm6-1.fna.fbcdn.net/o1/v/t2/f2/m69/AQPw9mck5zNWnfEut5KjZg5PU97Vqo7xXVcWBYRM7Ht26JYn568GgLKLE6wuFrwU3a5sxHNxKPj8rLQiULMJ-erk.mp4?efg=eyJ4cHZfYXNzZXRfaWQiOjEyMzM3MTkzMTQ3NDYyODgsInZlbmNvZGVfdGFnIjoieHB2X3Byb2dyZXNzaXZlLkZBQ0VCT09LLi5DMy4xOTIwLmRhc2hfaDI2NC1iYXNpYy1nZW4yXzEwODBwIn0&_nc_ht=video.fktm6-1.fna.fbcdn.net&_nc_cat=102&strext=1&vs=2a2a6c83ce12d24c&_nc_vs=HBkcFQIYOnBhc3N0aHJvdWdoX2V2ZXJzdG9yZS9HSkYwS3h2aEoxdGdFbUVnQUlEZXpFbHJnWlF3YnY0R0FBQUYVAALIAQAoABgAGwKIB3VzZV9vaWwBMRJwcm9ncmVzc2l2ZV9yZWNpcGUBMRUAACbgrpCY9YOxBBUCKAJDMywXQF47tkWhysEYGmRhc2hfaDI2NC1iYXNpYy1nZW4yXzEwODBwEQB1AgA&ccb=9-4&oh=00_AYB8pkVMe4mwo40JP9ZHcPj57HIGLiEflj3RcfDnINTV8g&oe=67C13CB4&_nc_sid=1d576d)
