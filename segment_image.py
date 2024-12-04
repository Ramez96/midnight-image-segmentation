# Import required libraries
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

class YourModel(nn.Module):
    def __init__(self, num_classes=9, pretrained=True):
        super(YourModel, self).__init__()
        # Load the pretrained DeepLabV3 with ResNet101 backbone
        self.model = models.segmentation.deeplabv3_resnet101(pretrained=pretrained)
        
        # Modify the classifier to match the number of classes
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)  # 9 classes

    def forward(self, x):
        return self.model(x)['out']

    def load_weights(self, model_path):
        """Load pre-trained weights into the model."""
        self.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    def evaluate(self):
        """Set the model to evaluation mode."""
        self.eval()

# ==============================
# Step 1: Load the Pre-trained DeepLabV3 Model
# ==============================
model_path = "your_model.pth"  # Path to your model
num_classes = 9  # Number of classes your model was trained for

# Load your model (make sure your model definition matches)
model = YourModel(num_classes=num_classes)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Load weights
model.eval()  # Set to evaluation mode
# ==============================
# Step 2: Preprocess the Input Image
# ==============================
# Define preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to model's input size
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Load an image
image_path = "./test_image.jpg"  # Replace with your image path
input_image = Image.open(image_path).convert("RGB")  # Ensure the image is RGB
input_tensor = preprocess(input_image).unsqueeze(0)  # Add batch dimension

# ==============================
# Step 3: Perform Segmentation
# ==============================
with torch.no_grad():
    output = model(input_tensor)["out"][0]  # Get output predictions

# Convert to segmentation map
output_predictions = torch.argmax(output, dim=0).numpy()  # Class IDs

# ==============================
# Step 4: Visualize the Results
# ==============================
# Map the classes to colors
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

# Decode segmentation map
segmentation_map = decode_segmap(output_predictions)

# Show the original and segmented images
plt.figure(figsize=(10, 5))

# Original image
plt.subplot(1, 2, 1)
plt.imshow(input_image)
plt.title("Original Image")

# Segmented image
plt.subplot(1, 2, 2)
plt.imshow(segmentation_map)
plt.title("Segmented Image")

plt.show()
