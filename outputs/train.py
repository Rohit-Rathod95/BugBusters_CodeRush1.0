"""
Training Script for EEG Classification Model
"""

import argparse
import os
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib
from models import ResNetMetaClassifier, SimpleEEGClassifier, EEGDataset

warnings.filterwarnings('ignore')
plt.switch_backend('Agg')


class EEGTrainer:
    """
    Trainer class for EEG classification models.
    """
    
    def __init__(self, model, device, criterion, optimizer, scheduler=None):
        """
        Initialize trainer.
        
        Args:
            model (nn.Module): PyTorch model
            device (torch.device): Device to train on
            criterion (nn.Module): Loss function
            optimizer (torch.optim): Optimizer
            scheduler: Learning rate scheduler (optional)
        """
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        predictions = []
        targets = []
        
        for batch in tqdm(train_loader, desc="Training", leave=False):
            # Move data to device
            images = batch['image'].to(self.device)
            metadata = batch['metadata'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            if isinstance(self.model, ResNetMetaClassifier):
                outputs = self.model(images, metadata)
            else:  # SimpleEEGClassifier
                outputs = self.model(metadata)
            
            # Compute loss
            loss = self.criterion(outputs.squeeze(), labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            predictions.extend(torch.sigmoid(outputs.squeeze()).detach().cpu().numpy())
            targets.extend(labels.detach().cpu().numpy())
        
        # Compute metrics
        epoch_loss = running_loss / len(train_loader)
        predictions_binary = (np.array(predictions) > 0.5).astype(int)
        epoch_accuracy = accuracy_score(targets, predictions_binary)
        
        return epoch_loss, epoch_accuracy
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch."""
        self.model.eval()
        running_loss = 0.0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                # Move data to device
                images = batch['image'].to(self.device)
                metadata = batch['metadata'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                if isinstance(self.model, ResNetMetaClassifier):
                    outputs = self.model(images, metadata)
                else:  # SimpleEEGClassifier
                    outputs = self.model(metadata)
                
                # Compute loss
                loss = self.criterion(outputs.squeeze(), labels)
                
                # Statistics
                running_loss += loss.item()
                predictions.extend(torch.sigmoid(outputs.squeeze()).detach().cpu().numpy())
                targets.extend(labels.detach().cpu().numpy())
        
        # Compute metrics
        epoch_loss = running_loss / len(val_loader)
        predictions_binary = (np.array(predictions) > 0.5).astype(int)
        epoch_accuracy = accuracy_score(targets, predictions_binary)
        
        # Additional metrics
        try:
            auc_score = roc_auc_score(targets, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                targets, predictions_binary, average='binary'
            )
        except:
            auc_score = 0.0
            precision = recall = f1 = 0.0
        
        metrics = {
            'loss': epoch_loss,
            'accuracy': epoch_accuracy,
            'auc': auc_score,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        return metrics
    
    def train(self, train_loader, val_loader, num_epochs, save_dir):
        """
        Train the model.
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            num_epochs (int): Number of epochs to train
            save_dir (str): Directory to save model and plots
        """
        print(f"Training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model: {self.model.__class__.__name__}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 30)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validate
            val_metrics = self.validate_epoch(val_loader)
            self.val_losses.append(val_metrics['loss'])
            self.val_accuracies.append(val_metrics['accuracy'])
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"Val AUC: {val_metrics['auc']:.4f}, Val F1: {val_metrics['f1']:.4f}")
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_model_state = self.model.state_dict().copy()
                
                # Save model
                model_path = os.path.join(save_dir, 'best_model.pth')
                torch.save({
                    'model_state_dict': self.best_model_state,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy'],
                    'model_class': self.model.__class__.__name__,
                    'model_config': getattr(self.model, 'config', {})
                }, model_path)
                
                print(f"New best model saved! Val Loss: {val_metrics['loss']:.4f}")
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step(val_metrics['loss'])
        
        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        # Save training plots
        self.save_training_plots(save_dir)
        
        print(f"\nTraining complete! Best validation loss: {self.best_val_loss:.4f}")
    
    def save_training_plots(self, save_dir):
        """Save training curves."""
        fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss curves
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(epochs, self.train_accuracies, 'b-', label='Training Accuracy')
        ax2.plot(epochs, self.val_accuracies, 'r-', label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()


def create_data_loaders(features_csv, train_ratio=0.8, batch_size=32, num_workers=4):
    """
    Create training and validation data loaders.
    
    Args:
        features_csv (str): Path to features CSV file
        train_ratio (float): Ratio of data to use for training
        batch_size (int): Batch size
        num_workers (int): Number of workers for data loading
    
    Returns:
        tuple: (train_loader, val_loader, dataset)
    """
    # Load features
    features_df = pd.read_csv(features_csv)
    
    # Filter out unknown labels
    features_df = features_df[features_df['label'].isin([0, 1])]
    
    print(f"Dataset statistics:")
    print(f"  Total samples: {len(features_df)}")
    print(f"  Healthy (0): {sum(features_df['label'] == 0)}")
    print(f"  Affected (1): {sum(features_df['label'] == 1)}")
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    full_dataset = EEGDataset(features_df, transform=None)  # Transform will be applied per split
    
    # Split dataset
    train_size = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Apply transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, full_dataset


def main():
    parser = argparse.ArgumentParser(description='Train EEG classification model')
    parser.add_argument('--features_csv', '-f', type=str, required=True,
                       help='Path to features CSV file')
    parser.add_argument('--output_dir', '-o', type=str, required=True,
                       help='Output directory for model and results')
    parser.add_argument('--model_type', '-m', type=str, default='resnet_meta',
                       choices=['resnet_meta', 'simple'],
                       help='Model type to train')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                       help='Dropout rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay for regularization')
    parser.add_argument('--freeze_backbone', action='store_true',
                       help='Freeze ResNet backbone (for ResNet model)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers for data loading')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Loading dataset...")
    train_loader, val_loader, dataset = create_data_loaders(
        args.features_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Get feature dimensions
    sample_batch = next(iter(train_loader))
    n_metadata_features = sample_batch['metadata'].shape[1]
    
    print(f"Metadata features: {n_metadata_features}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Create model
    if args.model_type == 'resnet_meta':
        model = ResNetMetaClassifier(
            n_metadata_features=n_metadata_features,
            dropout_rate=args.dropout_rate,
            freeze_backbone=args.freeze_backbone
        )
    else:  # simple
        model = SimpleEEGClassifier(
            n_features=n_metadata_features,
            dropout_rate=args.dropout_rate
        )
    
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=10, 
        verbose=True
    )
    
    # Save training configuration
    config = {
        'model_type': args.model_type,
        'n_metadata_features': n_metadata_features,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'dropout_rate': args.dropout_rate,
        'weight_decay': args.weight_decay,
        'freeze_backbone': args.freeze_backbone,
        'train_samples': len(train_loader.dataset),
        'val_samples': len(val_loader.dataset)
    }
    
    config_path = os.path.join(args.output_dir, 'training_config.pkl')
    joblib.dump(config, config_path)
    
    # Create trainer and train
    trainer = EEGTrainer(model, device, criterion, optimizer, scheduler)
    trainer.train(train_loader, val_loader, args.epochs, args.output_dir)
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, 'final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'model_class': model.__class__.__name__
    }, final_model_path)
    
    print(f"\nTraining complete!")
    print(f"Models saved to: {args.output_dir}")
    print(f"Best model: {os.path.join(args.output_dir, 'best_model.pth')}")
    print(f"Training config: {config_path}")


if __name__ == "__main__":
    main()