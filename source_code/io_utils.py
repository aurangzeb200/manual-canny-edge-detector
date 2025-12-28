import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def ensure_folder(path):
    os.makedirs(path, exist_ok=True)


def list_input_images(input_folder, input_ext):
    ext = input_ext.lower().lstrip('.')
    files = []
    for entry in os.listdir(input_folder):
        if entry.lower().endswith('.' + ext):
            files.append(os.path.join(input_folder, entry))
    return sorted(files)


def read_gray_image(path):
    im = Image.open(path).convert('L')
    return np.array(im, dtype=np.uint8)


def save_u8_image(arr, path):
    if arr.dtype != np.uint8:
        a = np.clip(np.rint(arr.astype(np.float64)), 0, 255).astype(np.uint8)
    else:
        a = arr
    Image.fromarray(a).save(path)


def normalize_to_u8(arr):
    a = arr.astype(np.float64)
    amin, amax = a.min(), a.max()
    if amax == amin:
        return np.zeros_like(a, dtype=np.uint8)
    norm = (a - amin) / (amax - amin)
    return (norm * 255.0).astype(np.uint8)


def save_quantized_plot_with_colorbar(quantized_data, magnitude_data, output_path,
                                      mask_threshold=1e-3, cmap='jet', dpi=150):
    if quantized_data.shape != magnitude_data.shape:
        raise ValueError("quantized_data and magnitude_data must have same shape")

    mask = magnitude_data < mask_threshold
    masked = np.ma.array(quantized_data, mask=mask)

    direction_labels = {
        0: '0° (Horizontal)',
        1: '45° (Diagonal /)',
        2: '90° (Vertical)',
        3: '135° (Diagonal \\)'
    }

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(masked, cmap=cmap, interpolation='nearest', vmin=0, vmax=3)
    ax.set_title('Quantized Gradient Directions')
    ax.axis('off')

    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3], fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels([direction_labels[i] for i in range(4)])
    cbar.set_label('Gradient direction', rotation=270, labelpad=12)

    plt.savefig(output_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)
