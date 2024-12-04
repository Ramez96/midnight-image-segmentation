# segmentation_training.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

# ==============================
# Dataset Class
# ==============================
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace('.jpg', '.png'))
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        
        if self.transform:
            image = self.transform(image)
            mask = transforms.ToTensor()(mask)  # Convert mask to tensor
            
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)  # Ensure mask is of type long
        return image, mask

# ==============================
# Model Creation Function
# ==============================
def create_model(num_classes=9, pretrained=True):
    """Creates and returns a DeepLabV3 model with a ResNet-101 backbone."""
    model = models.segmentation.deeplabv3_resnet101(pretrained=pretrained)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)  # Adjust for num_classes
    return model

# ==============================
# Training Function
# ==============================
def train_segmentation_model(
    model, train_loader, val_loader, num_epochs=20, learning_rate=1e-4, device=None
):
    """Trains the given segmentation model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)['out']
                val_loss += criterion(outputs, masks).item()
        print(f"Validation Loss: {val_loss:.4f}")

    return model

# ==============================
# Utility Functions
# ==============================
def get_data_loaders(
    train_image_dir, train_mask_dir, val_image_dir, val_mask_dir, batch_size=16
):
    """Creates and returns training and validation DataLoaders."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to match model input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = SegmentationDataset(train_image_dir, train_mask_dir, transform=transform)
    val_dataset = SegmentationDataset(val_image_dir, val_mask_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader
