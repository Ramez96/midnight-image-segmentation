# Import required libraries
import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
import random


class YourModel(nn.Module):
    def __init__(self, num_classes=9, pretrained=True):
        super(YourModel, self).__init__()
        # Load the DeepLabV3 training
        deeplab = models.segmentation.deeplabv3_resnet50(pretrained=pretrained)

        # Extract the backbone and classifier
        self.backbone = deeplab.backbone  # Matches the key names in your .pth file
        self.classifier = deeplab.classifier

        # Modify the classifier to match the number of classes
        self.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        # Use the backbone and classifier directly
        features = self.backbone(x)
        output = self.classifier(features['out'])
        return output

    def load_weights(self, model_path):
        """Load pre-trained weights into the training."""
        # Load the state_dict directly into the training
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        self.load_state_dict(state_dict, strict=False)

    def evaluate(self):
        """Set the training to evaluation mode."""
        self.eval()

# ==============================
# Define preprocessing pipeline
# ==============================
preprocess = transforms.Compose([
    transforms.Resize((240, 320)),
    transforms.ToTensor()
])

# Define a function to map classes to colors
def decode_segmap(segmentation, num_classes=21):
    label_colors = np.array([
        [0, 0, 0],        # Background
        [128, 0, 0],      # Aeroplane
        [0, 128, 0],      # Bicycle
        [128, 128, 0],    # Bird
        [0, 0, 128],      # Boat
        [128, 0, 128],    # Bottle
        [0, 128, 128],    # Bus
        [128, 128, 128],  # Car
        [64, 0, 0],       # Cat
        [192, 0, 0],      # Chair
        [64, 128, 0],     # Cow
        [192, 128, 0],    # Dining Table
        [64, 0, 128],     # Dog
        [192, 0, 128],    # Horse
        [64, 128, 128],   # Motorbike
        [192, 128, 128],  # Person
        [0, 64, 0],       # Potted Plant
        [128, 64, 0],     # Sheep
        [0, 192, 0],      # Sofa
        [128, 192, 0],    # Train
        [0, 64, 128]      # TV/Monitor
    ])
    r = np.zeros_like(segmentation).astype(np.uint8)
    g = np.zeros_like(segmentation).astype(np.uint8)
    b = np.zeros_like(segmentation).astype(np.uint8)
    for l in range(0, num_classes):
        idx = segmentation == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
    return np.stack([r, g, b], axis=2)

# ==============================
# Load Models and Test Images
# ==============================
image_dir = "../dataset/test/noisy"
output_base_dir = "../output_labels/"
os.makedirs(output_base_dir, exist_ok=True)  # Ensure the base output directory exists

# List of .pth files to test
pth_files = ["../models/noisy2.pth", "../models/blurred.pth", "../models/sharpened.pth"]

# Select 10 random images from the directory
all_images = [img for img in os.listdir(image_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
selected_images = random.sample(all_images, min(len(all_images), 10))

for pth_file in pth_files:
    # Extract the model name without the .pth extension
    model_name = os.path.splitext(os.path.basename(pth_file))[0]

    # Create output directory for this model
    model_output_dir = os.path.join(output_base_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    # Load the model
    model = YourModel(num_classes=9)
    model.load_weights(pth_file)
    model.eval()  # Set to evaluation mode

    for image_name in selected_images:
        image_path = os.path.join(image_dir, image_name)

        # Load and preprocess the image
        input_image = Image.open(image_path).convert("RGB")
        original_size = input_image.size  # Save the original size for overlay
        input_tensor = preprocess(input_image).unsqueeze(0)  # Add batch dimension

        # Perform segmentation
        with torch.no_grad():
            output = model(input_tensor)
            output_resized = F.interpolate(output, size=(240, 320), mode="bilinear", align_corners=False)
            predicted_labels = torch.argmax(output_resized, dim=1).squeeze(0).cpu().numpy()

        # Save the segmentation labels to a text file
        label_file = os.path.join(model_output_dir, f"{os.path.splitext(image_name)[0]}_labels.txt")
        np.savetxt(label_file, predicted_labels, fmt='%d')

        # Create an overlay of the segmentation map on the original image
        segmentation_map = decode_segmap(predicted_labels)
        segmentation_map_resized = Image.fromarray(segmentation_map).resize(original_size, Image.BILINEAR)
        segmentation_overlay = (0.5 * np.array(input_image) + 0.5 * np.array(segmentation_map_resized)).astype(np.uint8)

        # Save the overlay image
        overlay_path = os.path.join(model_output_dir, f"{os.path.splitext(image_name)[0]}_overlay.png")
        Image.fromarray(segmentation_overlay).save(overlay_path)

print("Processing completed. Outputs saved in respective directories.")
