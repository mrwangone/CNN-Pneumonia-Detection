import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt

from model import PneumoniaClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
            self,
            data_dir: str,
            batch_size: int = 32,
            num_epochs: int = 20,
            learning_rate: float = 0.001,
            image_size: int = 224
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.image_size = image_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Initialize history for plotting
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

        self._setup_transforms()
        self._setup_data_loaders()

    def _setup_transforms(self):
        """Set up data transforms for training and validation without augmentation."""
        self.train_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.4826, 0.4826, 0.4826], [0.2367, 0.2367, 0.2367])
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.4826, 0.4826, 0.4826], [0.2367, 0.2367, 0.2367])
        ])

    def _setup_data_loaders(self):
        """Set up data loaders for training and validation."""
        train_dataset = datasets.ImageFolder(
            os.path.join(self.data_dir, 'train'),
            transform=self.train_transform
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        val_dataset = datasets.ImageFolder(
            os.path.join(self.data_dir, 'val'),
            transform=self.val_transform
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")

    def save_checkpoint(self, model, optimizer, epoch, val_loss, is_best=False):
        """Save model checkpoint and best model state."""
        checkpoint_dir = Path('checkpoints')
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'history': self.history
        }

        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)

        # If this is the best model, save it separately
        if is_best:
            torch.save(model.state_dict(), 'model1.pth')
            torch.save(checkpoint, checkpoint_dir / 'best_checkpoint.pt')
            logger.info(f'Saved best model with validation loss: {val_loss:.4f}')

    def plot_training_progress(self):
        """Plot the training and validation metrics."""
        plt.figure(figsize=(15, 5))

        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], 'b-', label='Training Loss')
        plt.plot(self.history['val_loss'], 'r-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], 'b-', label='Training Accuracy')
        plt.plot(self.history['val_acc'], 'r-', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('training_progress1.png', dpi=300, bbox_inches='tight')
        plt.close()

    def train(self):
        """Main training loop."""
        model = PneumoniaClassifier().to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0

        for epoch in range(self.num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{self.num_epochs}')
            for inputs, labels in pbar:
                inputs = inputs.to(self.device)
                labels = labels.float().to(self.device)

                optimizer.zero_grad()
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                predicted = (outputs > 0.5).float()
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.float().to(self.device)

                    outputs = model(inputs).squeeze()
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    predicted = (outputs > 0.5).float()
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            # Calculate epoch statistics
            avg_train_loss = train_loss / len(self.train_loader)
            avg_val_loss = val_loss / len(self.val_loader)
            train_accuracy = 100 * train_correct / train_total
            val_accuracy = 100 * val_correct / val_total

            # Update history
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(avg_val_loss)
            self.history['train_acc'].append(train_accuracy)
            self.history['val_acc'].append(val_accuracy)

            logger.info(
                f'Epoch {epoch + 1}/{self.num_epochs} - '
                f'Train Loss: {avg_train_loss:.4f}, '
                f'Train Acc: {train_accuracy:.2f}%, '
                f'Val Loss: {avg_val_loss:.4f}, '
                f'Val Acc: {val_accuracy:.2f}%'
            )

            is_best = avg_val_loss < best_val_loss
            if is_best:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            self.save_checkpoint(model, optimizer, epoch, avg_val_loss, is_best)

            # Early stopping
            if patience_counter >= patience:
                logger.info(f'Early stopping triggered after {epoch + 1} epochs')
                break

        # Plot training progress at the end
        self.plot_training_progress()


def main():
    trainer = Trainer(
        data_dir='./data/dataset',
        batch_size=32,
        num_epochs=10,
        learning_rate=0.01
    )
    trainer.train()


if __name__ == '__main__':
    main()
