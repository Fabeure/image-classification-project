import numpy as np
import scipy.io
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class HyperspectralDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

def apply_pca(data, n_components=2):
    """Apply PCA to reduce spectral dimensions"""
    # Reshape for PCA
    h, w, bands = data.shape
    reshaped_data = data.reshape(-1, bands)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(reshaped_data)
    
    # Reshape back to original spatial dimensions
    return reduced_data.reshape(h, w, n_components)

def prepare_patches(data, labels, patch_size=11):
    """Prepare patches for CNN input"""
    padding = patch_size // 2
    padded_data = np.pad(data, ((padding, padding), (padding, padding), (0, 0)), mode='reflect')
    
    patches = []
    valid_labels = []
    
    for i in range(padding, padded_data.shape[0] - padding):
        for j in range(padding, padded_data.shape[1] - padding):
            if labels[i-padding, j-padding] != 0:  # Skip background pixels
                patch = padded_data[i-padding:i+padding+1, j-padding:j+padding+1, :]
                patches.append(patch)
                valid_labels.append(labels[i-padding, j-padding] - 1)  # Subtract 1 to make labels 0-based
                
    return np.array(patches), np.array(valid_labels)

def load_indian_pines():
    """Load and preprocess Indian Pines dataset"""
    data = scipy.io.loadmat('/Users/skandertebourbi/Downloads/Indian_pines_corrected.mat')['indian_pines_corrected']
    labels = scipy.io.loadmat('/Users/skandertebourbi/Downloads/Indian_pines_gt.mat')['indian_pines_gt']
    
    # Normalize the data
    scaler = StandardScaler()
    shaped_data = data.reshape(-1, data.shape[-1])
    shaped_data = scaler.fit_transform(shaped_data)
    data = shaped_data.reshape(data.shape)
    
    # Apply PCA
    data = apply_pca(data, n_components=2)
    
    return data, labels

def load_pavia():
    """Load and preprocess Pavia University dataset"""
    data = scipy.io.loadmat('/Users/skandertebourbi/Downloads/Pavia.mat')['pavia']
    labels = scipy.io.loadmat('/Users/skandertebourbi/Downloads/Pavia_gt.mat')['pavia_gt']
    
    # Normalize the data
    scaler = StandardScaler()
    shaped_data = data.reshape(-1, data.shape[-1])
    shaped_data = scaler.fit_transform(shaped_data)
    data = shaped_data.reshape(data.shape)
    
    # Apply PCA
    data = apply_pca(data, n_components=2)
    
    return data, labels

def get_data_loaders(data, labels, patch_size=11, batch_size=32, test_size=0.3):
    """Create train and validation data loaders"""
    # Prepare patches
    patches, valid_labels = prepare_patches(data, labels, patch_size)
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        patches, valid_labels, test_size=test_size, random_state=42, stratify=valid_labels
    )
    
    # Create datasets
    train_dataset = HyperspectralDataset(
        X_train, y_train,
        transform=lambda x: x.permute(2, 0, 1)  # Convert to channels-first format using PyTorch permute
    )
    val_dataset = HyperspectralDataset(
        X_val, y_val,
        transform=lambda x: x.permute(2, 0, 1)  # Convert to channels-first format using PyTorch permute
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, len(np.unique(valid_labels)) 