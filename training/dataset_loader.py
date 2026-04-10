import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


REQUIRED_COLUMNS = {"filename", "label", "split"}


class MammographyDataset(Dataset):
    """
    Custom PyTorch Dataset for loading mammography images and their corresponding labels.

    Attributes:
        annotations (pd.DataFrame): DataFrame containing filenames and labels.
        img_dir (str): Directory containing the preprocessed PNG images.
        transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(self, csv_file: str, img_dir: str, split: str, transform=None):
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"labels CSV not found: {csv_file}")
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"processed image directory not found: {img_dir}")

        df = pd.read_csv(csv_file)
        missing_columns = REQUIRED_COLUMNS.difference(df.columns)
        if missing_columns:
            raise ValueError(
                f"labels CSV must contain columns {sorted(REQUIRED_COLUMNS)}; missing {sorted(missing_columns)}"
            )

        if split not in set(df["split"].dropna().unique()):
            raise ValueError(f"split='{split}' not found in {csv_file}")

        self.annotations = df[df['split'] == split].reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.annotations)

    def __getitem__(self, index: int) -> tuple:
        """
        Generates one sample of data.

        Args:
            index (int): Index of the sample to fetch.

        Returns:
            tuple: (image_tensor, label_tensor)
        """
        row = self.annotations.iloc[index]
        img_name = row["filename"]
        img_path = os.path.join(self.img_dir, img_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Processed image not found: {img_path}")

        # Open image and convert to RGB (ResNet expects 3 channels)
        image = Image.open(img_path).convert("RGB")

        # Extract label and convert to float tensor for binary classification (BCE Loss)
        y_label = torch.tensor(float(row["label"]))

        if self.transform:
            image = self.transform(image)

        return image, y_label


def get_data_loaders(csv_path: str, img_dir: str, split: str, batch_size: int = 4) -> DataLoader:
    """
    Constructs and returns a PyTorch DataLoader for the mammography dataset.

    Args:
        csv_path (str): Path to the labels CSV file.
        img_dir (str): Path to the processed images directory.
        batch_size (int): Number of samples per batch. Defaults to 4.
        split(str): CSV file split.

    Returns:
        DataLoader: Iterable PyTorch DataLoader.
    """
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of: 'train', 'val', 'test'")

    # Standard ImageNet normalization values required by pre-trained ResNet
    transform_pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = MammographyDataset(csv_file=csv_path, img_dir=img_dir, split=split, transform=transform_pipeline)

    # Shuffle is set to True to ensure random distribution of MLO and CC during training
    is_train = (split == 'train')
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=is_train)

    return loader