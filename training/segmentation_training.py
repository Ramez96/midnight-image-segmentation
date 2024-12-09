# segmentation_training.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.transforms.functional import resize
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

# ==============================
# Dataset Class
# ==============================
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, target_size=(240, 320)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_size = target_size
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace('.jpg', '.regions.txt'))

        # Load the image
        image = Image.open(img_path).convert("RGB")

        # Load the mask as a NumPy array
        mask = np.loadtxt(mask_path, dtype=np.int64)
        # Replace -1 with class 8 (valid class)
        mask[mask == -1] = 8
        # Convert the mask to a tensor
        mask = torch.as_tensor(mask, dtype=torch.long)

        # Resize both the image and mask to the target size
        image = resize(image, self.target_size, interpolation=Image.BILINEAR)
        mask = resize(mask.unsqueeze(0), self.target_size, interpolation=Image.NEAREST).squeeze(0)

        # Apply transformations to the image
        if self.transform:
            image = self.transform(image)

        return image, mask

# ==============================
# Model Creation Function
# ==============================
def create_model(num_classes=9, pretrained=True):
    """Creates and returns a DeepLabV3 training with a ResNet-101 backbone."""
    print(f"Creating DeepLabV3 training with {num_classes} classes...")
    model = models.segmentation.deeplabv3_resnet50(pretrained=pretrained)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)  # Adjust for num_classes
    return model

# ==============================
# Training Function
# ==============================
def train_segmentation_model(
    model, train_loader, val_loader, num_epochs=20, learning_rate=1e-4, device=None
):
    """Trains the given segmentation training."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Starting training loop...")

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        for step, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Print loss every 50 steps (you can adjust this)
            if step % 50 == 0:
                print(f"  Step {step}, Loss: {loss.item():.4f}")

        print(f"Epoch {epoch + 1} completed. Total Loss: {epoch_loss:.4f}")

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_step, (images, masks) in enumerate(val_loader):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)['out']
                val_loss += criterion(outputs, masks).item()

                # Print validation loss every 50 steps
                if val_step % 50 == 0:
                    print(f"  Validation Step {val_step}, Validation Loss: {val_loss:.4f}")

        print(f"Validation Loss after Epoch {epoch + 1}: {val_loss:.4f}")

    print("Training complete.")
    return model

# ==============================
# Utility Functions
# ==============================
def get_data_loaders(
    train_image_dir, train_mask_dir, val_image_dir, val_mask_dir, batch_size=16
):
    """Creates and returns training and validation DataLoaders."""
    print("Creating data loaders...")
    transform = transforms.Compose([
        transforms.Resize((240, 320)),  # Resize images to match training input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = SegmentationDataset(train_image_dir, train_mask_dir, transform=transform)
    val_dataset = SegmentationDataset(val_image_dir, val_mask_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    print(f"Data loaders created with batch size {batch_size}.")
    return train_loader, val_loader
