import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from model import HyperspectralCNN
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import os
from dataset import load_indian_pines, load_pavia

def predict_entire_image(model, data, labels, patch_size=11, batch_size=32, device='cuda'):
    """Predict class labels for every pixel in the image"""
    padding = patch_size // 2
    padded_data = np.pad(data, ((padding, padding), (padding, padding), (0, 0)), mode='reflect')
    
    height, width = labels.shape
    all_patches = []
    positions = []
    
    # Prepare patches for all pixels
    for i in range(height):
        for j in range(width):
            patch = padded_data[i:i+patch_size, j:j+patch_size, :]
            all_patches.append(patch)
            positions.append((i, j))
    
    # Convert to tensor and create DataLoader
    patches_tensor = torch.FloatTensor(np.array(all_patches)).permute(0, 3, 1, 2)  # NCHW format
    dataset = TensorDataset(patches_tensor, torch.zeros(len(positions)))  # Dummy labels
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Make predictions
    model.eval()
    predictions = np.zeros_like(labels)
    
    with torch.no_grad():
        batch_start = 0
        for batch in loader:
            inputs = batch[0].to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Assign predictions to their positions
            batch_size = inputs.size(0)
            for k in range(batch_size):
                i, j = positions[batch_start + k]
                predictions[i, j] = preds[k].item() + 1  # +1 to match original labeling
            
            batch_start += batch_size
    
    return predictions

def evaluate_predictions(true_labels, pred_labels):
    """Evaluate predictions against ground truth"""
    # Flatten arrays
    true_flat = true_labels.flatten()
    pred_flat = pred_labels.flatten()
    
    # Only evaluate pixels that have ground truth labels (including class 0 if present)
    eval_mask = true_flat != -1  # Assuming -1 means no ground truth
    
    print("\nClassification Report (All Labeled Pixels):")
    print(classification_report(true_flat[eval_mask], pred_flat[eval_mask], zero_division=0))
    
    print(f"\nOverall Accuracy: {accuracy_score(true_flat[eval_mask], pred_flat[eval_mask]):.4f}")
    
    # Confusion matrix
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(true_flat[eval_mask], pred_flat[eval_mask])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()
    
    return cm

def visualize_results(data, true_labels, pred_labels, dataset_name):
    """Visualize input data, ground truth and predictions"""
    plt.figure(figsize=(18, 6))
    
    # First PCA component
    plt.subplot(131)
    plt.imshow(data[:, :, 0], cmap='viridis')
    plt.title(f'{dataset_name} - First PCA Component')
    plt.colorbar()
    
    # Ground truth
    plt.subplot(132)
    plt.imshow(true_labels, cmap='nipy_spectral')
    plt.title('Ground Truth Labels')
    plt.colorbar()
    
    # Predictions
    plt.subplot(133)
    plt.imshow(pred_labels, cmap='nipy_spectral')
    plt.title('Predicted Labels')
    plt.colorbar()
    
    plt.suptitle(f'{dataset_name} Classification Results', y=1.02)
    plt.tight_layout()
    plt.show()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    patch_size = 11
    batch_size = 32
    
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Process Indian Pines
    print("\n=== Indian Pines Dataset ===")
    data, labels = load_indian_pines()
    num_classes = len(np.unique(labels)) - 1  # Exclude background
    
    model = HyperspectralCNN(in_channels=2, num_classes=num_classes).to(device)
    model_path = 'models/best_model_indian.pth'
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Model loaded successfully")
    else:
        print(f"Error: Model not found at {model_path}")
        return
    
    # Predict all pixels
    predictions = predict_entire_image(model, data, labels, patch_size, batch_size, device)
    
    # Evaluate and visualize
    cm = evaluate_predictions(labels, predictions)
    visualize_results(data, labels, predictions, "Indian Pines")
    
    # Process Pavia University
    print("\n=== Pavia University Dataset ===")
    data, labels = load_pavia()
    num_classes = len(np.unique(labels)) - 1  # Exclude background
    
    model = HyperspectralCNN(in_channels=2, num_classes=num_classes).to(device)
    model_path = 'models/best_model_pavia.pth'
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Model loaded successfully")
    else:
        print(f"Error: Model not found at {model_path}")
        return
    
    # Predict all pixels
    predictions = predict_entire_image(model, data, labels, patch_size, batch_size, device)
    
    # Evaluate and visualize
    cm = evaluate_predictions(labels, predictions)
    visualize_results(data, labels, predictions, "Pavia University")

if __name__ == '__main__':
    main()