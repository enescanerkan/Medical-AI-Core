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
    raw_data_path: str = os.path.join(base_dir, 'data', 'raw')
    raw_test_data_path: str = os.path.join(base_dir, 'data', 'raw_test')
    dataset_path: str = os.path.join(base_dir, 'dataset')
    processed_data_path: str = os.path.join(dataset_path, 'processed')
    labels_csv_path: str = os.path.join(dataset_path, 'labels.csv')
    result_path: str = os.path.join(base_dir, 'result')

    # Image Processing Parameters ( e.g., for Mammography or CT)
    target_image_size: tuple = (224,224) # Type Hinting
    clahe_clip_limit: float= 2.0

    # Feautre Flags
    process_test_images: bool = False

    # Deep Learning Hyperparameters
    # TODO: Add optimizer configuration support (e.g., Adam, SGD)
    batch_size: int = 16
    learning_rate: float = 0.001
    epochs: int = 50
