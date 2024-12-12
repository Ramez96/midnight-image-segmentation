# main_script.py

import torch
from segmentation_training import create_model, train_segmentation_model, get_data_loaders

# Set dataset paths
train_image_dir = "../dataset/train/preprocessing5"
train_mask_dir = "../dataset/train/labels"
val_image_dir = "../dataset/val/preprocessing5"
val_mask_dir = "../dataset/val/labels"

# Get data loaders
train_loader, val_loader = get_data_loaders(
    train_image_dir, train_mask_dir, val_image_dir, val_mask_dir, batch_size=16
)

# Create the training
model = create_model(num_classes=9, pretrained=True)

# Train the training
trained_model = train_segmentation_model(model, train_loader, val_loader, num_epochs=20)

# Save the trained training
torch.save(trained_model.state_dict(), "../models/sharpend.pth")
