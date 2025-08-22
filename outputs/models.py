"""
Neural Network Models for EEG Classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np


class ResNetMetaClassifier(nn.Module):
    """
    Combined ResNet + Metadata classifier for EEG analysis.
    Uses ResNet18 for scalogram images and MLP for metadata features.
    """
    
    def __init__(self, n_metadata_features=10, dropout_rate=0.5, freeze_backbone=False):
        """
        Initialize the model.
        
        Args:
            n_metadata_features (int): Number of metadata features (PSD features)
            dropout_rate (float): Dropout rate for regularization
            freeze_backbone (bool): Whether to freeze ResNet backbone
        """
        super(ResNetMetaClassifier, self).__init__()
        
        self.n_metadata_features = n_metadata_features
        
        # ResNet backbone for scalogram images
        self.backbone = models.resnet18(pretrained=True)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get the number of features from ResNet
        backbone_features = self.backbone.fc.in_features
        
        # Remove the final classification layer from ResNet
        self.backbone.fc = nn.Identity()
        
        # Metadata MLP
        self.metadata_mlp = nn.Sequential(
            nn.Linear(n_metadata_features, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout_rate),
            
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        
        # Combined classifier
        combined_features = backbone_features + 16  # 512 from ResNet + 16 from metadata MLP
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, 1)  # Single output for binary classification
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for custom layers."""
        for m in [self.metadata_mlp, self.classifier]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
                elif isinstance(layer, nn.BatchNorm1d):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)
    
    def forward(self, image, metadata):
        """
        Forward pass.
        
        Args:
            image (torch.Tensor): Scalogram images (batch_size, 3, 224, 224)
            metadata (torch.Tensor): Metadata features (batch_size, n_metadata_features)
        
        Returns:
            torch.Tensor: Binary classification logits (batch_size, 1)
        """
        # Extract features from scalogram using ResNet
        image_features = self.backbone(image)  # (batch_size, 512)
        
        # Process metadata through MLP
        meta_features = self.metadata_mlp(metadata)  # (batch_size, 16)
        
        # Concatenate features
        combined_features = torch.cat([image_features, meta_features], dim=1)  # (batch_size, 528)
        
        # Final classification
        output = self.classifier(combined_features)  # (batch_size, 1)
        
        return output
    
    def get_feature_importance(self, image, metadata):
        """
        Get feature importance for interpretability.
        
        Args:
            image (torch.Tensor): Scalogram images
            metadata (torch.Tensor): Metadata features
        
        Returns:
            dict: Feature importance scores
        """
        self.eval()
        
        # Enable gradient computation for input
        image.requires_grad_(True)
        metadata.requires_grad_(True)
        
        # Forward pass
        output = self.forward(image, metadata)
        
        # Compute gradients
        output.backward()
        
        # Get gradients as importance scores
        image_importance = torch.abs(image.grad).mean(dim=(2, 3))  # Average over spatial dimensions
        metadata_importance = torch.abs(metadata.grad)
        
        return {
            'image_importance': image_importance.detach(),
            'metadata_importance': metadata_importance.detach()
        }


class SimpleEEGClassifier(nn.Module):
    """
    Simplified classifier using only metadata features.
    Useful for comparison and lightweight deployment.
    """
    
    def __init__(self, n_features, hidden_sizes=[64, 32, 16], dropout_rate=0.5):
        """
        Initialize simple classifier.
        
        Args:
            n_features (int): Number of input features
            hidden_sizes (list): List of hidden layer sizes
            dropout_rate (float): Dropout rate
        """
        super(SimpleEEGClassifier, self).__init__()
        
        layers = []
        input_size = n_features
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout_rate)
            ])
            input_size = hidden_size
        
        # Final classification layer
        layers.append(nn.Linear(input_size, 1))
        
        self.classifier = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights."""
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input features (batch_size, n_features)
        
        Returns:
            torch.Tensor: Classification logits (batch_size, 1)
        """
        return self.classifier(x)


class EEGDataset(torch.utils.data.Dataset):
    """
    Custom dataset for EEG data loading.
    """
    
    def __init__(self, features_df, transform=None, metadata_cols=None):
        """
        Initialize dataset.
        
        Args:
            features_df (pd.DataFrame): Features dataframe
            transform (callable): Optional transform for images
            metadata_cols (list): List of metadata column names
        """
        self.features_df = features_df.reset_index(drop=True)
        self.transform = transform
        
        # Default metadata columns (PSD features)
        if metadata_cols is None:
            self.metadata_cols = [col for col in features_df.columns 
                                if '_power' in col or '_rel_power' in col]
        else:
            self.metadata_cols = metadata_cols
        
        # Normalize metadata features
        self.metadata_mean = features_df[self.metadata_cols].mean()
        self.metadata_std = features_df[self.metadata_cols].std()
        self.metadata_std = self.metadata_std.replace(0, 1)  # Avoid division by zero
    
    def __len__(self):
        return len(self.features_df)
    
    def __getitem__(self, idx):
        row = self.features_df.iloc[idx]
        
        # Load image
        image_path = row['image_path']
        try:
            from PIL import Image
            image = Image.open(image_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            else:
                # Default transform: convert to tensor and normalize
                import torchvision.transforms as transforms
                default_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
                image = default_transform(image)
        
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return dummy image
            image = torch.zeros(3, 224, 224)
        
        # Get metadata features
        metadata = row[self.metadata_cols].values.astype(np.float32)
        
        # Normalize metadata
        metadata = (metadata - self.metadata_mean.values) / self.metadata_std.values
        metadata = torch.tensor(metadata, dtype=torch.float32)
        
        # Get label
        label = torch.tensor(row['label'], dtype=torch.float32)
        
        return {
            'image': image,
            'metadata': metadata,
            'label': label,
            'image_path': image_path
        }


def get_model_summary(model, input_shapes):
    """
    Print model summary.
    
    Args:
        model (nn.Module): PyTorch model
        input_shapes (tuple): Tuple of input shapes (image_shape, metadata_shape)
    """
    try:
        from torchsummary import summary
        print("Model Summary:")
        print("=" * 50)
        
        # Create dummy inputs
        image_shape, metadata_shape = input_shapes
        dummy_image = torch.randn(1, *image_shape)
        dummy_metadata = torch.randn(1, metadata_shape)
        
        model.eval()
        with torch.no_grad():
            output = model(dummy_image, dummy_metadata)
            print(f"Output shape: {output.shape}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print("=" * 50)
        
    except ImportError:
        print("torchsummary not available. Install with: pip install torchsummary")
        
        # Basic parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")


if __name__ == "__main__":
    # Test the models
    print("Testing ResNetMetaClassifier...")
    
    # Create model
    model = ResNetMetaClassifier(n_metadata_features=10, dropout_rate=0.5)
    
    # Test forward pass
    batch_size = 4
    dummy_image = torch.randn(batch_size, 3, 224, 224)
    dummy_metadata = torch.randn(batch_size, 10)
    
    output = model(dummy_image, dummy_metadata)
    print(f"Input shapes: Image {dummy_image.shape}, Metadata {dummy_metadata.shape}")
    print(f"Output shape: {output.shape}")
    
    # Model summary
    get_model_summary(model, ((3, 224, 224), 10))
    
    print("\nTesting SimpleEEGClassifier...")
    simple_model = SimpleEEGClassifier(n_features=10)
    simple_output = simple_model(dummy_metadata)
    print(f"Simple model output shape: {simple_output.shape}")