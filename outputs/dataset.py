"""
Dataset handling utilities for EEG classification system.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

class EEGDataset(Dataset):
    """
    Custom Dataset class for EEG data combining scalogram images and metadata features.
    """
    
    def __init__(self, csv_file, image_dir, transform=None, scaler=None, fit_scaler=False):
        """
        Initialize the EEG dataset.
        
        Args:
            csv_file (str): Path to the features CSV file
            image_dir (str): Directory containing scalogram images
            transform: Image transformations
            scaler: Fitted scaler for metadata features
            fit_scaler (bool): Whether to fit the scaler on this data
        """
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        
        # Prepare metadata features
        self.metadata_columns = [
            'delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power',
            'delta_rel_power', 'theta_rel_power', 'alpha_rel_power', 
            'beta_rel_power', 'gamma_rel_power'
        ]
        
        # Handle missing metadata columns
        for col in self.metadata_columns:
            if col not in self.data.columns:
                print(f"Warning: {col} not found in dataset. Setting to 0.")
                self.data[col] = 0.0
        
        # Prepare metadata features
        self.metadata_features = self.data[self.metadata_columns].values.astype(np.float32)
        
        # Handle scaling
        if fit_scaler and scaler is None:
            self.scaler = StandardScaler()
            self.metadata_features = self.scaler.fit_transform(self.metadata_features)
        elif scaler is not None:
            self.scaler = scaler
            self.metadata_features = self.scaler.transform(self.metadata_features)
        else:
            self.scaler = None
        
        # Labels
        self.labels = self.data['label'].values.astype(np.float32)
        
        print(f"Dataset loaded: {len(self)} samples")
        print(f"Label distribution: {np.bincount(self.labels.astype(int))}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a single sample from the dataset."""
        
        # Load image
        image_path = os.path.join(self.image_dir, self.data.iloc[idx]['image_path'])
        
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            image = torch.zeros(3, 224, 224) if self.transform else Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Get metadata features
        metadata = torch.tensor(self.metadata_features[idx], dtype=torch.float32)
        
        # Get label
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return image, metadata, label

def get_data_transforms():
    """
    Get data transformations for training and validation.
    
    Returns:
        tuple: (train_transform, val_transform)
    """
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_data_loaders(csv_file, image_dir, batch_size=32, test_size=0.2, random_state=42):
    """
    Create training and validation data loaders.
    
    Args:
        csv_file (str): Path to features CSV
        image_dir (str): Path to images directory
        batch_size (int): Batch size for data loaders
        test_size (float): Proportion of data for validation
        random_state (int): Random seed
    
    Returns:
        tuple: (train_loader, val_loader, scaler)
    """
    
    # Load data to get splits
    data = pd.read_csv(csv_file)
    
    # Split data
    train_idx, val_idx = train_test_split(
        range(len(data)), 
        test_size=test_size, 
        random_state=random_state,
        stratify=data['label']
    )
    
    # Create temporary CSV files for splits
    train_data = data.iloc[train_idx].reset_index(drop=True)
    val_data = data.iloc[val_idx].reset_index(drop=True)
    
    train_csv = csv_file.replace('.csv', '_train_temp.csv')
    val_csv = csv_file.replace('.csv', '_val_temp.csv')
    
    train_data.to_csv(train_csv, index=False)
    val_data.to_csv(val_csv, index=False)
    
    # Get transforms
    train_transform, val_transform = get_data_transforms()
    
    # Create datasets
    train_dataset = EEGDataset(
        train_csv, image_dir, 
        transform=train_transform, 
        fit_scaler=True
    )
    
    val_dataset = EEGDataset(
        val_csv, image_dir, 
        transform=val_transform, 
        scaler=train_dataset.scaler
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # Clean up temporary files
    os.remove(train_csv)
    os.remove(val_csv)
    
    return train_loader, val_loader, train_dataset.scaler

def save_scaler(scaler, path):
    """Save the fitted scaler."""
    joblib.dump(scaler, path)
    print(f"Scaler saved to {path}")

def load_scaler(path):
    """Load a fitted scaler."""
    return joblib.load(path)

class InferenceDataset(Dataset):
    """
    Dataset class for inference on single files.
    """
    
    def __init__(self, images, metadata_features, transform=None):
        """
        Initialize inference dataset.
        
        Args:
            images (list): List of PIL images
            metadata_features (list): List of metadata feature arrays
            transform: Image transformations
        """
        self.images = images
        self.metadata_features = metadata_features
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        
        if self.transform:
            image = self.transform(image)
        
        metadata = torch.tensor(self.metadata_features[idx], dtype=torch.float32)
        
        return image, metadata

def calculate_class_weights(labels):
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        labels (array): Array of labels
    
    Returns:
        torch.Tensor: Class weights
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    class_weights = len(labels) / (len(unique_labels) * counts)
    
    return torch.FloatTensor(class_weights)

def validate_dataset(csv_file, image_dir):
    """
    Validate dataset consistency.
    
    Args:
        csv_file (str): Path to features CSV
        image_dir (str): Path to images directory
    
    Returns:
        dict: Validation results
    """
    
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    try:
        # Load CSV
        data = pd.read_csv(csv_file)
        results['stats']['total_samples'] = len(data)
        
        # Check required columns
        required_columns = ['image_path', 'label']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            results['valid'] = False
            results['errors'].append(f"Missing required columns: {missing_columns}")
        
        # Check labels
        unique_labels = data['label'].unique()
        results['stats']['unique_labels'] = unique_labels.tolist()
        results['stats']['label_distribution'] = data['label'].value_counts().to_dict()
        
        # Check images exist
        missing_images = []
        for idx, row in data.iterrows():
            image_path = os.path.join(image_dir, row['image_path'])
            if not os.path.exists(image_path):
                missing_images.append(row['image_path'])
        
        if missing_images:
            results['warnings'].append(f"Missing {len(missing_images)} images")
            if len(missing_images) > len(data) * 0.1:  # More than 10% missing
                results['valid'] = False
                results['errors'].append("Too many missing images")
        
        results['stats']['missing_images'] = len(missing_images)
        
    except Exception as e:
        results['valid'] = False
        results['errors'].append(f"Error validating dataset: {e}")
    
    return results