# visualizer.py
"""
Provides visualization tools for medical images, histograms, and model outputs.
"""

import matplotlib.pyplot as plt
import numpy as np

class MedicalVisualizer:
    """
    Utility class for plotting medical image comparisons.
    """

    @staticmethod
    def plot_before_after(original: np.ndarray, processed: np.ndarray, title: str = "Image Comparison"):
        """
        Plots the original and processed images side-by-side using Matplotlib.

        Args:
            original (np.ndarray): The raw input image.
            processed (np.ndarray): The processed image (e.g., after CLAHE).
            title (str): Main title for the plot.
        """
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(title, fontsize=14)

        axes[0].imshow(original, cmap='gray')
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        axes[1].imshow(processed, cmap='gray')
        axes[1].set_title("Processed Image")
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()