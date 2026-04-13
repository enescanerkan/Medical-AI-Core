# test_advanced_prep.py
"""
Visualizes Wavelet Edge Boosting and Laterality-Aware Smart Cropping.
"""

import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Dynamically append the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ProjectConfig
from preprocessing.cv_transforms import AdvancedImageProcessor


def normalize_for_display(image_array: np.ndarray) -> np.ndarray:
    """
    Normalizes a floating-point frequency band to 0-255 for proper visualization.
    Eliminates the 'solid gray' Matplotlib issue by stretching the contrast.

    Args:
        image_array (np.ndarray): The raw wavelet coefficient matrix.

    Returns:
        np.ndarray: An 8-bit unsigned integer matrix ready for imshow.
    """
    abs_array = np.abs(image_array)
    # Stretch values strictly between 0 and 255
    normalized = cv2.normalize(abs_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return normalized


def main():
    cfg = ProjectConfig()
    processed_dir = os.path.join(cfg.dataset_path, "processed")

    image_files = [f for f in os.listdir(processed_dir) if f.endswith(".png")]
    if not image_files:
        print("Error: No images found in the processed directory. Please run create_dataset.py first.")
        return

    test_file = image_files[0]
    test_path = os.path.join(processed_dir, test_file)
    print(f"Testing advanced preprocessing on: {test_file}")

    # Load original image in grayscale
    raw_img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)

    # Step 1: Apply Wavelet Edge Boost (Factor set to 10.0 for high visibility)
    boosted_img = AdvancedImageProcessor.wavelet_edge_boost(raw_img, boost_factor=10.0)

    # Step 2: Extract Wavelet Decomposition bands from the BOOSTED image
    ll_band, (lh_band, hl_band, hh_band) = AdvancedImageProcessor.get_discrete_wavelet(boosted_img)

    # Step 3: Apply Min-Max Normalization to high-frequency bands for visualization
    lh_vis = normalize_for_display(lh_band)
    hl_vis = normalize_for_display(hl_band)
    hh_vis = normalize_for_display(hh_band)

    # Step 4: Calculate the mathematical difference (Proof of Concept)
    diff_img = cv2.absdiff(raw_img, boosted_img)
    diff_visible = np.clip(diff_img * 10, 0, 255).astype(np.uint8)

    # Step 5: Apply Laterality-Aware Smart Cropping and Padding
    final_tensor_ready_img, laterality = AdvancedImageProcessor.smart_crop_and_pad(boosted_img, target_size=(512, 512))

    # --- Visualization Dashboard ---
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle(f"Advanced Preprocessing & Frequency Analysis - {test_file}", fontsize=18, fontweight='bold')

    # Top Row: Preprocessing Pipeline
    axes[0, 0].imshow(raw_img, cmap='gray')
    axes[0, 0].set_title("1. Original Image")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(boosted_img, cmap='gray')
    axes[0, 1].set_title("2. Boosted Image")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(diff_visible, cmap='hot')
    axes[0, 2].set_title("3. Absolute Difference (Hot)")
    axes[0, 2].axis('off')

    axes[0, 3].imshow(final_tensor_ready_img, cmap='gray')
    axes[0, 3].set_title(f"4. Cropped (512x512) | Pos: {laterality}")
    axes[0, 3].axis('off')

    # Bottom Row: Wavelet Frequencies (Now visible!)
    axes[1, 0].imshow(ll_band, cmap='gray')
    axes[1, 0].set_title("LL Band (Background)")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(lh_vis, cmap='gray')
    axes[1, 1].set_title("LH Band (Horizontal Details)")
    axes[1, 1].axis('off')

    axes[1, 2].imshow(hl_vis, cmap='gray')
    axes[1, 2].set_title("HL Band (Vertical Details)")
    axes[1, 2].axis('off')

    axes[1, 3].imshow(hh_vis, cmap='gray')
    axes[1, 3].set_title("HH Band (Diagonal Details)")
    axes[1, 3].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()