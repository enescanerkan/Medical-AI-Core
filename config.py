# config.py

from dataclasses import dataclass
import os


@dataclass
class ProjectConfig:
    """
    Configuration class storing all hardcoded variables and paths.
    Using a dataclass ensures clean structure and easy access (SOLID principles)
    """

    # Dataset Paths
    base_dir: str = os.path.dirname(os.path.abspath(__file__))
    dataset_path: str = os.path.join(base_dir, 'dataset')

    # Image Processing Parameters ( e.g., for Mammography or CT)
    target_image_size: tuple = (224,224)
    clahe_clip_limit: float= 2.0

    # Deep Learning Hyperparameters
    batch_size: int = 16
    learning_rate: float = 0.001
    epochs: int = 50

