# Classification d'Images Hyperspectrales avec CNN

## Table des matières
1. [Analyse des Données](#analyse-des-données)
   - [Jeux de Données Utilisés](#jeux-de-données-utilisés)
2. [Approche Technique](#approche-technique)
   - [Réduction de Dimensionnalité par PCA](#réduction-de-dimensionnalité-par-pca)
   - [Prétraitement des Données](#prétraitement-des-données)
   - [Architecture du CNN](#architecture-du-cnn)
   - [Stratégie d'Entraînement](#stratégie-dentraînement)
3. [Résultats et Conclusions](#résultats-et-conclusions)
   - [Perspectives d'Amélioration](#perspectives-damélioration)

## Analyse des Données

Les images hyperspectrales sont des ensembles de données tridimensionnels qui capturent des informations spectrales détaillées pour chaque pixel d'une scène. Contrairement aux images RGB traditionnelles qui n'ont que trois canaux (rouge, vert, bleu), les images hyperspectrales contiennent des centaines de bandes spectrales étroites, offrant une signature spectrale unique pour chaque matériau.

### Jeux de Données Utilisés

1. **Indian Pines**
   - Dimensions originales : 145×145 pixels × 200 bandes spectrales
   - Après réduction PCA : 145×145 pixels × 2 composantes principales
   - Scène agricole contenant différentes cultures et types de sol

2. **Pavia University**
   - Dimensions originales : 1096×715 pixels × 102 bandes spectrales
   - Après réduction PCA : 1096×715 pixels × 2 composantes principales
   - Zone urbaine avec différents types de surfaces

## Approche Technique

### Réduction de Dimensionnalité par PCA

```python
from sklearn.decomposition import PCA

def apply_pca(data, n_components=2):
    # Reshape pour PCA
    h, w, bands = data.shape
    reshaped_data = data.reshape(-1, bands)
    
    # Application de PCA
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(reshaped_data)
    
    # Reshape vers la forme originale
    return reduced_data.reshape(h, w, n_components)
```

Cette étape de réduction de dimensionnalité est cruciale car :
- Elle réduit la complexité computationnelle
- Elle élimine la redondance dans les bandes spectrales
- Elle conserve l'information la plus discriminante
- Elle facilite la visualisation des données

### Prétraitement des Données

```python
def prepare_patches(data, labels, patch_size=11):
    padding = patch_size // 2
    padded_data = np.pad(data, ((padding, padding), (padding, padding), (0, 0)), mode='reflect')
    
    patches = []
    valid_labels = []
    
    for i in range(padding, padded_data.shape[0] - padding):
        for j in range(padding, padded_data.shape[1] - padding):
            if labels[i-padding, j-padding] != 0:
                patch = padded_data[i-padding:i+padding+1, j-padding:j+padding+1, :]
                patches.append(patch)
                valid_labels.append(labels[i-padding, j-padding] - 1)
```

Cette fonction extrait des patches de 11×11 pixels autour de chaque pixel d'intérêt, permettant de capturer le contexte spatial local. Les pixels de fond (label 0) sont ignorés.

### Architecture du CNN

```python
class HyperspectralCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(HyperspectralCNN, self).__init__()
        
        # Blocs convolutifs
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Couches fully connected
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)
```

L'architecture choisie est adaptée aux données réduites par PCA :
- Trois blocs convolutifs avec normalisation par lots
- Pooling global pour réduire la dimensionnalité
- Couches fully connected pour la classification
- Dropout pour prévenir le surapprentissage

### Stratégie d'Entraînement

```python
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100):
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
```

Paramètres d'entraînement :
- Taille de batch : 32
- Nombre d'époques : 100
- Taux d'apprentissage : 0.001
- Optimiseur : Adam
- Fonction de perte : CrossEntropyLoss

## Résultats et Conclusions

L'approche adoptée présente plusieurs avantages :

1. **Efficacité Computationnelle** : La réduction à 2 composantes principales permet un entraînement plus rapide tout en conservant l'information discriminante.

2. **Contexte Spatial** : L'utilisation de patches permet de capturer les relations spatiales entre pixels voisins, améliorant la précision de classification.

3. **Robustesse** : La normalisation par lots et le dropout rendent le modèle plus robuste aux variations dans les données.

Les résultats montrent que :
- Le modèle converge rapidement sur les deux jeux de données
- La précision de classification reste élevée malgré la réduction de dimensionnalité
- Les courbes d'apprentissage sont stables

### Perspectives d'Amélioration

1. **Architecture Plus Profonde** : L'ajout de blocs résiduels pourrait améliorer la capacité du modèle à apprendre des caractéristiques complexes.

2. **Augmentation des Données** : Des techniques d'augmentation spécifiques aux images hyperspectrales pourraient améliorer la généralisation.

Cette implémentation démontre l'efficacité de la combinaison PCA-CNN pour la classification d'images hyperspectrales, offrant un bon compromis entre performance et efficacité computationnelle. 