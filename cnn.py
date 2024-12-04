import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision import transforms
from torch import nn, optim
from PIL import Image
import matplotlib.pyplot as plt


# -------------------------------
# Custom Dataset Class
# -------------------------------
class StanfordBackgroundDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        self.labels = sorted(os.listdir(label_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image and label
        img_path = os.path.join(self.image_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.labels[idx])

        # Open image
        image = Image.open(img_path).convert("RGB")
        # Open label as numpy array and convert to a PIL image
        label = np.loadtxt(label_path, dtype=np.int32)
        label = Image.fromarray(label)

        # Apply transformations
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label


# -------------------------------
# Data Preparation
# -------------------------------
def prepare_data(image_dir, label_dir):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    dataset = StanfordBackgroundDataset(image_dir, label_dir, transform=transform)
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    return train_loader, val_loader


# -------------------------------
# Define the DeepLabv3 Model
# -------------------------------
def get_deeplabv3_model(num_classes):
    model = deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)  # Adjust for custom classes
    return model


# -------------------------------
# Training Loop
# -------------------------------
def train_model(model, train_loader, val_loader, num_epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.long().to(device)
            optimizer.zero_grad()

            outputs = model(images)['out']
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

        # Validation
        validate_model(model, val_loader, device)


# -------------------------------
# Validation Loop
# -------------------------------
def validate_model(model, val_loader, device):
    model.eval()
    accuracy = 0.0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.long().to(device)
            outputs = model(images)['out']
            predictions = outputs.argmax(dim=1)

            accuracy += (predictions == labels).sum().item()
            total += labels.numel()

    print(f"Validation Accuracy: {accuracy / total:.2%}")


# -------------------------------
# Visualization
# -------------------------------
def visualize_sample(model, dataset, device):
    model.eval()
    image, label = dataset[0]
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)['out']
        predicted = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

    # Display
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(label.squeeze(), cmap="gray")
    plt.title("Ground Truth")

    plt.subplot(1, 2, 2)
    plt.imshow(predicted, cmap="gray")
    plt.title("Prediction")
    plt.show()


# -------------------------------
# Main Function
# -------------------------------
if __name__ == "__main__":
    # Paths to dataset
    image_dir = "./stanford_background_dataset/images"
    label_dir = "./stanford_background_dataset/labels"

    # Prepare data
    train_loader, val_loader = prepare_data(image_dir, label_dir)

    # Load model
    num_classes = 10  # Replace with the actual number of classes in your dataset
    model = get_deeplabv3_model(num_classes)

    # Train model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 20
    train_model(model, train_loader, val_loader, num_epochs, device)

    # Visualize results
    dataset = StanfordBackgroundDataset(image_dir, label_dir, transform=transforms.ToTensor())
    visualize_sample(model, dataset, device)
