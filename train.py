import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix

from dataset import get_data_loaders, load_indian_pines, load_pavia
from model import HyperspectralCNN


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, device='cuda'):
    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # Validation phase
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(all_labels, all_preds)
        val_accuracies.append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    return train_losses, val_accuracies

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Training parameters
    patch_size = 11
    batch_size = 32
    num_epochs = 100
    learning_rate = 0.001
    
    # Train on Indian Pines
    print("\nTraining on Indian Pines dataset...")
    data, labels = load_indian_pines()
    train_loader, val_loader, num_classes = get_data_loaders(
        data, labels, patch_size=patch_size, batch_size=batch_size
    )
    
    # Create model with 2 input channels (after PCA)
    model_indian = HyperspectralCNN(in_channels=2, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_indian.parameters(), lr=learning_rate)
    
    train_losses_indian, val_accuracies_indian = train_model(
        model_indian, train_loader, val_loader, criterion, optimizer, 
        num_epochs=num_epochs, device=device
    )
    
    # Train on Pavia University
    print("\nTraining on Pavia University dataset...")
    data, labels = load_pavia()
    train_loader, val_loader, num_classes = get_data_loaders(
        data, labels, patch_size=patch_size, batch_size=batch_size
    )
    
    # Create model with 2 input channels (after PCA)
    model_pavia = HyperspectralCNN(in_channels=2, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_pavia.parameters(), lr=learning_rate)
    
    train_losses_pavia, val_accuracies_pavia = train_model(
        model_pavia, train_loader, val_loader, criterion, optimizer, 
        num_epochs=num_epochs, device=device
    )
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(121)
    plt.plot(train_losses_indian, label='Indian Pines')
    plt.plot(train_losses_pavia, label='Pavia')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(122)
    plt.plot(val_accuracies_indian, label='Indian Pines')
    plt.plot(val_accuracies_pavia, label='Pavia')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.close()

if __name__ == '__main__':
    main() 