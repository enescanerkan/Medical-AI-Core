# test_advanced_prep.py
"""
Visualizes Wavelet Edge Boosting and Laterality-Aware Smart Cropping.
"""

import sys
import os
import cv2
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ProjectConfig
from preprocessing.cv_transforms import AdvancedImageProcessor


def main():
    cfg = ProjectConfig()
    processed_dir = os.path.join(cfg.dataset_path, "processed")

    image_files = [f for f in os.listdir(processed_dir) if f.endswith(".png")]
    if not image_files:
        print("No images found in the processed directory.")
        return

    test_file = image_files[0]
    test_path = os.path.join(processed_dir, test_file)
    print(f"Testing advanced prep on: {test_file}")

    raw_img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)

    # 1. Wavelet Edge Boost
    boosted_img = AdvancedImageProcessor.wavelet_edge_boost(raw_img, boost_factor=16.0)

    # 2. Smart Crop & Pad  (512x512)
    final_tensor_ready_img, laterality = AdvancedImageProcessor.smart_crop_and_pad(boosted_img, target_size=(512, 512))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Advanced Preprocessing Pipeline - {test_file}", fontsize=16)

    axes[0].imshow(raw_img, cmap='gray')
    axes[0].set_title(f"1. Original ({raw_img.shape[1]}x{raw_img.shape[0]})")
    axes[0].axis('off')

    axes[1].imshow(boosted_img, cmap='gray')
    axes[1].set_title("2. Wavelet Edge Boosted (Factor: 3.0)")
    axes[1].axis('off')

    axes[2].imshow(final_tensor_ready_img, cmap='gray')
    axes[2].set_title(f"3. Smart Cropped & Padded (512x512) | Pos: {laterality}")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()