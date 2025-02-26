import argparse
import os
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sapiens import vit_base_patch16_1024, load_pretrained_model

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

def create_data_loaders(config):
    """Create training and validation data loaders."""
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config['img_size']),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Transform for validation
    val_transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = datasets.ImageFolder(
        os.path.join(config['data_dir'], 'train'),
        transform=train_transform
    )
    
    val_dataset = datasets.ImageFolder(
        os.path.join(config['data_dir'], 'val'),
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader, len(train_dataset.classes)

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    start_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # If model outputs features, use only the CLS token for classification
        if len(outputs.shape) > 2:
            outputs = outputs[:, 0]  # Use CLS token (first token)
            
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(train_loader):
            print(f'Epoch: {epoch} | Batch: {batch_idx+1}/{len(train_loader)} | Loss: {running_loss/(batch_idx+1):.4f} | Acc: {100.*correct/total:.2f}%')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    epoch_time = time.time() - start_time
    
    return epoch_loss, epoch_acc, epoch_time

def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            
            # If model outputs features, use only the CLS token for classification
            if len(outputs.shape) > 2:
                outputs = outputs[:, 0]  # Use CLS token (first token)
                
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc

def add_classification_head(model, num_classes):
    """Add a classification head to the SAPIENS model."""
    # Get feature dimension from the model
    feature_dim = model.embed_dim
    
    # Create a simple classifier head
    classifier = nn.Linear(feature_dim, num_classes)
    
    # Return the modified model
    return nn.Sequential(model, classifier)

def main():
    parser = argparse.ArgumentParser(description="Fine-tune SAPIENS model")
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--pretrained', required=True, help='Path to pretrained model weights')
    parser.add_argument('--resume', default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    train_loader, val_loader, num_classes = create_data_loaders(config)
    print(f"Dataset loaded with {num_classes} classes")
    
    # Initialize model
    if args.resume:
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model = checkpoint['model']
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        print(f"Resuming from epoch {start_epoch}, best accuracy: {best_acc:.2f}%")
    else:
        print(f"Initializing model with pretrained weights from {args.pretrained}")
        base_model = vit_base_patch16_1024()
        base_model = load_pretrained_model(base_model, args.pretrained, device)
        
        # Modify the model for fine-tuning (add classification head)
        model = add_classification_head(base_model, num_classes)
        model = model.to(device)
        start_epoch = 0
        best_acc = 0
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
        eta_min=config['min_lr']
    )
    
    # Resume optimizer and scheduler if needed
    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    # Create directory for checkpoints
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Training loop
    print(f"Starting training for {config['epochs']} epochs")
    for epoch in range(start_epoch, config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        
        # Train
        train_loss, train_acc, epoch_time = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch + 1
        )
        print(f"Training completed in {epoch_time:.1f}s | Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Validation | Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
        
        # Update scheduler
        scheduler.step()
        
        # Save checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        
        checkpoint_path = os.path.join(config['output_dir'], f"checkpoint_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'model': model,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_acc': best_acc,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        
        if is_best:
            best_model_path = os.path.join(config['output_dir'], "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model': model,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_acc': best_acc,
            }, best_model_path)
            print(f"New best model saved with accuracy: {val_acc:.2f}%")
    
    print(f"Training completed. Best validation accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()
