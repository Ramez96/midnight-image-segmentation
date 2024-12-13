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

        self.backbone = deeplab.backbone  
        self.classifier = deeplab.classifier

        # Modify the classifier to match the number of classes
        self.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features['out'])
        return output

    def load_weights(self, model_path):
        """Load pre-trained weights into the training."""
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        self.load_state_dict(state_dict, strict=False)

    def evaluate(self):
        """Set the training to evaluation mode."""
        self.eval()

preprocess = transforms.Compose([
    transforms.Resize((240, 320)),
    transforms.ToTensor()
])

def decode_segmap(segmentation, num_classes=9):
    label_colors = np.array([
        [0, 0, 0],        # Background
        [128, 0, 0],      # Sky
        [0, 128, 0],      # Tree
        [128, 128, 0],    # Road
        [0, 0, 128],      # Grass
        [128, 0, 128],    # Water
        [0, 128, 128],    # Building
        [128, 128, 128],  # Mountain
        [64, 0, 0],       # Foreground Objects
        
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
input_base_dir = "../dataset/test/"
output_base_dir = "../output_labels/"
os.makedirs(output_base_dir, exist_ok=True)

pth_files = ["../models/noisy.pth", "../models/blurred.pth", "../models/sharpened.pth","../models/baseline.pth"]

model_name_1 = os.path.splitext(os.path.basename(pth_files[0]))[0]
image_dir_1 = os.path.join(input_base_dir, model_name_1)
all_images_1 = [img for img in os.listdir(image_dir_1) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
selected_images = random.sample(all_images_1, min(len(all_images_1), 10))

for pth_file in pth_files:
    # Extract the model name without the .pth extension
    model_name = os.path.splitext(os.path.basename(pth_file))[0]

    # Determine the corresponding input directory
    image_dir = os.path.join(input_base_dir, model_name)

    if not os.path.exists(image_dir):
        print(f"Warning: Input directory for {model_name} does not exist. Skipping.")
        continue

    # Create output directory for this model
    model_output_dir = os.path.join(output_base_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    # Load the model
    model = YourModel(num_classes=9)
    model.load_weights(pth_file)
    model.eval()  # Set to evaluation mode
    for image_name in selected_images:
        image_path = os.path.join(image_dir, image_name)

        if not os.path.exists(image_path):
            print(f"Warning: {image_name} does not exist in {image_dir}. Skipping.")
            continue

        # Load and preprocess the image
        input_image = Image.open(image_path).convert("RGB")
        original_size = input_image.size  
        input_tensor = preprocess(input_image).unsqueeze(0)  

        with torch.no_grad():
            output = model(input_tensor)
            output_resized = F.interpolate(output, size=(240, 320), mode="bilinear", align_corners=False)
            predicted_labels = torch.argmax(output_resized, dim=1).squeeze(0).cpu().numpy()

        label_file = os.path.join(model_output_dir, f"{os.path.splitext(image_name)[0]}.txt")
        np.savetxt(label_file, predicted_labels, fmt='%d')

        segmentation_map = decode_segmap(predicted_labels)
        segmentation_map_resized = Image.fromarray(segmentation_map).resize(original_size, Image.BILINEAR)
        segmentation_overlay = (0.5 * np.array(input_image) + 0.5 * np.array(segmentation_map_resized)).astype(np.uint8) # Create an overlay of the segmentation map on the original image

        # Save the overlay image
        overlay_path = os.path.join(model_output_dir, f"{os.path.splitext(image_name)[0]}_overlay.png")
        Image.fromarray(segmentation_overlay).save(overlay_path)
print("Processing completed. Outputs saved in respective directories.")
