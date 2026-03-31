import os
import cv2
import numpy as np


class MedicalDataLoader:
    """
    A data loader class responsible for handling medical image files.

    This class adheres to the Single Responsibility Principle (SRP) by
    focusing solely on reading, validating, and returning image data
    from a specified directory.
    """

    def __init__(self, data_path: str) -> None:
        """
        Initializes the MedicalDataLoader with the root data directory.

        Args:
            data_path (str): The absolute or relative path to the directory
                             containing the medical image dataset.
        """
        self.data_path = data_path

    def load_image(self, file_name: str) -> np.ndarray:
        """
        Loads an image file from the specified data path.

        Args:
            file_name (str): The exact name of the file to load (e.g., 'scan_001.png').

        Returns:
            np.ndarray: The loaded image represented as a NumPy array.

        Raises:
            FileNotFoundError: If the specified file does not exist in the data path.
            ValueError: If the image cannot be read or is corrupted.
        """
        path = os.path.join(self.data_path, file_name)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Image file not found at: {path}")

        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Failed to load image. Ensure it is a valid format: {path}")

        return image