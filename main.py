import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from config import ProjectConfig
from training.dataset_loader import get_data_loaders
from training.model import MammographyResNet
from training.trainer import MedicalModelTrainer


def parse_args() -> argparse.Namespace:
    cfg = ProjectConfig()
    default_save_path = os.path.join(cfg.result_path, "mammography_resnet.pth")

    parser = argparse.ArgumentParser(description="Train mammography classifier.")
    parser.add_argument("--epochs", type=int, default=cfg.epochs, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=cfg.learning_rate, help="AdamW learning rate.")
    parser.add_argument("--csv", type=str, default=cfg.labels_csv_path, help="Path to labels CSV file.")
    parser.add_argument("--img-dir", type=str, default=cfg.processed_data_path, help="Path to processed image directory.")
    parser.add_argument("--save-path", type=str, default=default_save_path, help="Path to save trained model weights.")
    return parser.parse_args()


if __name__ == "__main__":
    cfg = ProjectConfig()
    args = parse_args()

    csv_path = args.csv
    img_dir = args.img_dir
    model_save_path = args.save_path

    if not os.path.exists(csv_path):
        print(
            f"Dataset Error: labels CSV not found at {csv_path}. "
            "First run `create_dataset.py` to prepare `dataset/processed/`, then provide `dataset/labels.csv`."
        )
        sys.exit(1)

    if not os.path.isdir(img_dir):
        print(f"Dataset Error: processed image directory not found at {img_dir}. Run `create_dataset.py` first.")
        sys.exit(1)

    try:
        train_dataloader = get_data_loaders(
            csv_path=csv_path,
            img_dir=img_dir,
            split='train',
            batch_size=args.batch_size,
        )
    except Exception as e:
        print(f"Dataset Error: {e}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet_model = MammographyResNet(num_classes=1)
    loss_function = nn.BCEWithLogitsLoss()
    optimizer_function = optim.AdamW(resnet_model.parameters(), lr=args.learning_rate)

    trainer = MedicalModelTrainer(
        model=resnet_model,
        dataloader=train_dataloader,
        criterion=loss_function,
        optimizer=optimizer_function,
        device=device,
    )

    trainer.run(num_epochs=args.epochs, save_path=model_save_path)
