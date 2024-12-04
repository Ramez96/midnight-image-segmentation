# main_script.py

import torch
from segmentation_training import create_model, train_segmentation_model, get_data_loaders

# Set dataset paths
train_image_dir = "dataset/train/images"
train_mask_dir = "dataset/train/masks"
val_image_dir = "dataset/val/images"
val_mask_dir = "dataset/val/masks"

# Get data loaders
train_loader, val_loader = get_data_loaders(
    train_image_dir, train_mask_dir, val_image_dir, val_mask_dir, batch_size=16
)

# Create the model
model = create_model(num_classes=9, pretrained=True)

# Train the model
trained_model = train_segmentation_model(model, train_loader, val_loader, num_epochs=20)

# Save the trained model
torch.save(trained_model.state_dict(), "segmentation_model.pth")
