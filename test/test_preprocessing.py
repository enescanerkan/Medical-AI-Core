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

    # Step4: Calculate Histogram
    raw_hist = AdvancedImageProcessor.calculate_histogram(raw_img)
    clahe_hist = AdvancedImageProcessor.calculate_histogram(clahe_img)

    # Step 5: Qualitative Visualization
    # Display the transformation stages side-by-side for visual inspection.
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Image Enhancement Analysis - {test_file}", fontsize=16)

    axes[0, 0].imshow(raw_img, cmap='gray')
    axes[0, 0].set_title("Original Mammogram")
    axes[0, 0].axis('off')

    axes[0, 1].plot(raw_hist, color='black')
    axes[0, 1].fill_between(range(256), raw_hist, color='gray', alpha=0.5)
    axes[0, 1].set_title("Original Pixel Distribution")
    axes[0, 1].set_xlim([0, 256])
    axes[0, 1].grid(True, alpha=0.3)

    # Note: cmap='gray' is intentionally omitted here because contour_img is an RGB matrix.
    axes[1, 0].imshow(clahe_img, cmap='gray')
    axes[1, 0].set_title(f"CLAHE Enhanced (Clip Limit: 3.0)")
    axes[1, 0].axis('off')

    axes[1, 1].plot(clahe_hist, color='blue')
    axes[1, 1].fill_between(range(256), clahe_hist, color='blue', alpha=0.3)
    axes[1, 1].set_title("CLAHE Pixel Distribution")
    axes[1, 1].set_xlim([0, 256])
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()