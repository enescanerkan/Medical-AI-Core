#example.py
"""
A test script to demonstrate the pipeline from data loading to preprocessing and modeling.
"""

import numpy as np
from config import ProjectConfig
from preprocessing.cv_transforms import AdvancedImageProcessor
from models.tensor_bridge import TensorBridge
from evaluation.visualizer import MedicalVisualizer
from models.architectures import SimpleMedicalCNN

def main():
    print("--- Medical AI Core Pipeline Test ---")

    cfg = ProjectConfig()
    dummy_image = np.random.randint(0, 256, (500, 500), dtype=np.uint8)

    clahe_image = AdvancedImageProcessor.apply_clahe(dummy_image, clip_limit=cfg.clahe_clip_limit)

    MedicalVisualizer.plot_before_after(dummy_image, clahe_image, "Synthetic Mammogram - CLAHE Test")

    bridge = TensorBridge(target_size=cfg.target_image_size)
    tensor_image = bridge.to_pytorch(clahe_image)
    tensor_input = tensor_image.unsqueeze(0)

    model = SimpleMedicalCNN(in_channels=1, num_classes=2)
    output = model(tensor_input)

    print(f"Model Output (Logits): {output.detach().numpy()}")
    print("Pipeline executed successfully!")

if __name__ == "__main__":
    main()