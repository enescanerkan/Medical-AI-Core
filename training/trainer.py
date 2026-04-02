# trainer.py
"""
Description: Encapsulates the main PyTorch training loop engine.
"""

import torch
import torch.nn as nn
import torch.optim as optim


class MedicalModelTrainer:
    """
    Handles the epoch iterations, forward/backward passes, and weight updates.

    IMPORTANT CONCEPT: Encapsulating the training loop isolates logic and adheres
    to the Single Responsibility Principle.
    """

    def __init__(self, model: nn.Module, dataloader, criterion: nn.Module, optimizer: optim.Optimizer,
                 device: str = 'cpu'):
        """
        Args:
            model (nn.Module): The PyTorch neural network.
            dataloader: Iterable providing batches of images and labels.
            criterion (nn.Module): The loss function (e.g., DiceLoss).
            optimizer (optim.Optimizer): The optimization algorithm (e.g., Adam).
            device (str): Compute device ('cpu' or 'cuda').
        """
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

        # Device matching. Explicitly move model to target device.
        self.model.to(self.device)

    def train_one_epoch(self) -> float:
        """
        Executes a single training epoch across the entire dataloader.

        Returns:
            float: The average loss for this epoch.
        """
        # Calling .train() enables dynamic layers like Dropout.
        self.model.train()
        running_loss = 0.0

        for images, labels in self.dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Zero gradients to prevent accumulation from previous passes.
            self.optimizer.zero_grad()

            # Forward Pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backpropagation computes partial derivatives.
            loss.backward()

            # Weight Update based on computed gradients.
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(self.dataloader)