import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from training.dataset_loader import get_data_loaders
from training.model import MammographyResNet

class ModelTrainer:
    """
    Orchestrates the training loop for the deep learning model.
    """
    def __init__(self, model: nn.Module, train_loader, criterion, optimizer, device: torch.device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self) -> float:
        self.model.train()
        running_loss = 0.0
        loop = tqdm(self.train_loader, leave=True)

        for images, labels in loop:
            images = images.to(self.device)
            labels = labels.to(self.device).view(-1, 1)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        return running_loss / len(self.train_loader)

    def run(self, num_epochs: int, save_path: str) -> None:
        print(f"Starting training on device: {self.device} for {num_epochs} epochs.")

        for epoch in range(num_epochs):
            print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
            avg_loss = self.train_epoch()
            print(f"Average Training Loss: {avg_loss:.4f}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        print(f"\nTraining Complete! Model saved to {save_path}")

if __name__ == "__main__":
    # Configuration
    CSV_PATH = "dataset/labels.csv"
    IMG_DIR = "dataset/processed"
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 10
    MODEL_SAVE_PATH = "result/mammography_resnet.pth"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # FIX: Added split='train' argument to match the new dataset_loader signature
    try:
        train_dataloader = get_data_loaders(
            csv_path=CSV_PATH,
            img_dir=IMG_DIR,
            split='train',
            batch_size=BATCH_SIZE
        )
    except Exception as e:
        print(f"Dataset Error: {e}")
        exit(1)

    resnet_model = MammographyResNet(num_classes=1)
    loss_function = nn.BCEWithLogitsLoss()
    optimizer_function = optim.AdamW(resnet_model.parameters(), lr=LEARNING_RATE)

    trainer = ModelTrainer(
        model=resnet_model,
        train_loader=train_dataloader,
        criterion=loss_function,
        optimizer=optimizer_function,
        device=DEVICE
    )

    trainer.run(num_epochs=NUM_EPOCHS, save_path=MODEL_SAVE_PATH)