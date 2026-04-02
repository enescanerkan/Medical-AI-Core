# augmentations.py
"""
Defines data augmentation pipelines for medical images.
"""

import torchvision.transforms as T


class MedicalDataAugmenter:
    # Data Augmentation combats Overfitting in small medical datasets.

    def __init__(self, target_size: tuple = (224, 224)):
        """
         Initializes the augmentation pipelines.

                Args:
                    target_size (tuple): Desired image dimensions (H, W).
        """
        self.train_transforms = T.Compose([
            T.ToPILImage(),
            T.RandomRotation(degrees=15),
            T.RandomHorizontalFlip(p=0.5),
            T.Resize(target_size),
            T.ToTensor(),
            # Normalization speeds up Gradient Descent convergence.
            T.Normalize(mean=[0.5], std=[0.5])
        ])

        self.val_transforms = T.Compose([
            T.ToPILImage(),
            T.Resize(target_size),
            T.ToTensor(),
            # Normalization standardizes pixel values to have a mean of 0.5
            # and standard deviation of 0.5. This speeds up Gradient Descent convergence.
            T.Normalize(mean=[0.5], std=[0.5])
        ])

    def augment_for_training(self, image_array):
        """Applies the training transformation pipeline to a single image array."""
        return self.train_transforms(image_array)

