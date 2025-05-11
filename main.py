import matplotlib.pyplot as plt
import numpy as np
import scipy.io

indian_pines_path = '/Users/skandertebourbi/Downloads/Indian_pines_corrected.mat'
pavia_path = '/Users/skandertebourbi/Downloads/Pavia.mat'

# Load the data
indian_pines = scipy.io.loadmat(indian_pines_path)
indian_pines_data = indian_pines['indian_pines_corrected']

# Create a figure to show different spectral bands
plt.figure(figsize=(15, 8))

# Show the same spatial region at different wavelengths (spectral bands)
bands_to_show = [0, 50, 100, 150]  # Different spectral bands
for idx, band in enumerate(bands_to_show):
    plt.subplot(2, 2, idx + 1)
    plt.imshow(indian_pines_data[:, :, band])
    plt.title(f'Spectral Band {band}')
    plt.colorbar()

plt.suptitle('Indian Pines - Same Region at Different Wavelengths', y=1.02)
plt.tight_layout()
plt.show()

# Print explanation of the data structure
print("\nHyperspectral Data Explanation:")
print(f"Indian Pines data shape: {indian_pines_data.shape}")
print("- First two dimensions ({0} x {1}) represent spatial coordinates (like a regular image)".format(
    indian_pines_data.shape[0], indian_pines_data.shape[1]))
print(f"- Third dimension ({indian_pines_data.shape[2]}) represents different wavelengths/spectral bands")
print("\nUnlike a regular RGB image with 3 color channels, hyperspectral images capture")
print("many narrow bands across the electromagnetic spectrum, providing detailed")
print("spectral information for each pixel.")
