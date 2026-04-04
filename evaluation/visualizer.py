"""
Provides visualization tools for medical images, histograms, and model outputs.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class MedicalVisualizer:
    """
    Utility class for plotting medical image comparisons and explainability maps.
    """

    @staticmethod
    def plot_before_after(original: np.ndarray, processed: np.ndarray, title: str = "Image Comparison") -> None:
        """
        Plots the original and processed images side-by-side using Matplotlib.
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

    @staticmethod
    def plot_gradcam(model: torch.nn.Module, target_layer, input_tensor: torch.Tensor,
                     original_image: np.ndarray, prediction_score: float, true_label: int,
                     save_path: str = None) -> None:  # Added save_path parameter
        """
        Generates and displays (or saves) a Grad-CAM heatmap over the original image.
        """
        import os
        model.eval()

        cam = GradCAM(model=model, target_layers=[target_layer])
        targets = [ClassifierOutputTarget(0)]

        # noinspection PyTypeChecker
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        cam_image = show_cam_on_image(original_image, grayscale_cam, use_rgb=True)

        plt.figure(figsize=(6, 6))
        plt.imshow(cam_image)

        pred_label = 1 if prediction_score >= 0.5 else 0
        title_color = "green" if pred_label == true_label else "red"

        plt.title(f"True: {true_label} | Pred: {pred_label} (Score: {prediction_score:.2f})",
                  color=title_color, fontsize=12)
        plt.axis('off')
        plt.tight_layout()

        # Save to disk instead of showing if save_path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
        else:
            plt.show()
            plt.close()