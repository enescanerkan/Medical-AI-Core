import os
import sys

import torch
import numpy as np
from tqdm import tqdm

from config import ProjectConfig
from training.dataset_loader import get_data_loaders
from training.model import MammographyResNet
from evaluation.metrics import MedicalMetrics
from evaluation.visualizer import MedicalVisualizer


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    Reverts the ImageNet normalization to display the image properly.
    """
    img = tensor.cpu().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img


def evaluate_model():
    cfg = ProjectConfig()
    csv_path = cfg.labels_csv_path
    img_dir = cfg.processed_data_path
    model_weights = os.path.join(cfg.result_path, "mammography_resnet.pth")
    batch_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(csv_path):
        print(f"Evaluation Error: labels CSV not found at {csv_path}.")
        sys.exit(1)

    if not os.path.exists(model_weights):
        print(f"Evaluation Error: model weights not found at {model_weights}.")
        sys.exit(1)

    print("Loading test dataset...")
    try:
        test_loader = get_data_loaders(csv_path=csv_path, img_dir=img_dir, split='test', batch_size=batch_size)
    except Exception as exc:
        print(f"Evaluation Error: {exc}")
        sys.exit(1)

    print("Loading trained model...")
    model = MammographyResNet(num_classes=1)
    model.load_state_dict(torch.load(model_weights, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    # Select the target layer for Grad-CAM (last convolutional layer of ResNet50)
    target_layer = model.backbone.layer4[-1]

    print("Running evaluation...")
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(test_loader)):
            images = images.to(device)
            labels = labels.to(device).view(-1, 1)

            # Get raw logits and apply sigmoid
            outputs = model(images)
            probabilities = torch.sigmoid(outputs)

            all_preds.append(probabilities)
            all_targets.append(labels)

            # Generate Grad-CAM only for the first 3 test images to keep evaluation fast.
            if i < 3:
                with torch.enable_grad():
                    original_image_np = denormalize(images[0])
                    # Define a save path for each test image
                    save_path = os.path.join(
                        cfg.result_path,
                        "gradcam",
                        f"test_image_{i}_true{int(labels[0].item())}.png",
                    )

                    MedicalVisualizer.plot_gradcam(
                        model=model,
                        target_layer=target_layer,
                        input_tensor=images,
                        original_image=original_image_np,
                        prediction_score=probabilities[0].item(),
                        true_label=int(labels[0].item()),
                        save_path=save_path  # Pass the save path
                    )

    if not all_preds:
        print("Evaluation Error: no samples found in test split.")
        sys.exit(1)

    # Compute Final Metrics
    final_preds = torch.cat(all_preds)
    final_targets = torch.cat(all_targets)

    accuracy = MedicalMetrics.calculate_accuracy(final_preds, final_targets)
    sensitivity = MedicalMetrics.calculate_sensitivity(final_preds, final_targets)
    specificity = MedicalMetrics.calculate_specificity(final_preds, final_targets)

    print("\n" + "=" * 30)
    print("FINAL TEST METRICS")
    print("=" * 30)
    print(f"Accuracy    : {accuracy:.4f}")
    print(f"Sensitivity : {sensitivity:.4f}")
    print(f"Specificity : {specificity:.4f}")
    print("=" * 30)


if __name__ == "__main__":
    evaluate_model()