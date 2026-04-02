# test_preprocessing.py
"""
Validates and visualizes the image preprocessing pipeline.
This script applies CLAHE, binary thresholding, and contour extraction to a
sample medical image to evaluate the feature enhancement quality before model training.
"""

import sys
import os

# Dynamically append the project root directory to the Python path
# to enable absolute imports across different environments.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import matplotlib.pyplot as plt
from config import ProjectConfig
from preprocessing.cv_transforms import AdvancedImageProcessor

def main():
    cfg = ProjectConfig()
    processed_dir = os.path.join(cfg.dataset_path, "processed")

    # Fetch all available PNG images from the processed dataset
    image_files = [f for f in os.listdir(processed_dir) if f.endswith(".png")]

    if not image_files:
        print("Error: No PNG files found in the 'processed' directory. Please run dicom_processor.py first.")
        return

    # Select the first available image for pipeline validation
    test_file = image_files[0]
    test_path = os.path.join(processed_dir, test_file)
    print(f"Testing preprocessing pipeline on: {test_file}")

    # Load the target image in grayscale mode
    raw_img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)

    # Step 1: Contrast Enhancement
    # Apply CLAHE to amplify local contrast and highlight potential microcalcifications.
    clahe_img = AdvancedImageProcessor.apply_clahe(raw_img, clip_limit=3.0)

    # Step 2: Region of Interest (ROI) Extraction
    # Apply binary thresholding to filter out background noise, followed by contour detection.
    _, binary_img = cv2.threshold(clahe_img, 50, 255, cv2.THRESH_BINARY)
    contours = AdvancedImageProcessor.find_contours(clahe_img, threshold_val=50)

    # Step 3: Overlay Annotation
    # Convert the enhanced grayscale image to RGB format specifically to draw colored annotations (red).
    contour_img = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(contour_img, contours, -1, (255, 0, 0), 2)

    # Step 4: Qualitative Visualization
    # Display the transformation stages side-by-side for visual inspection.
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"Preprocessing Pipeline Evaluation - {test_file}", fontsize=16)

    axes[0].imshow(raw_img, cmap='gray')
    axes[0].set_title("1. Original Image")
    axes[0].axis('off')

    axes[1].imshow(clahe_img, cmap='gray')
    axes[1].set_title("2. CLAHE Enhanced")
    axes[1].axis('off')

    axes[2].imshow(binary_img, cmap='gray')
    axes[2].set_title("3. Binary Thresholding")
    axes[2].axis('off')

    # Note: cmap='gray' is intentionally omitted here because contour_img is an RGB matrix.
    axes[3].imshow(contour_img)
    axes[3].set_title(f"4. Detected Contours (n={len(contours)})")
    axes[3].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()