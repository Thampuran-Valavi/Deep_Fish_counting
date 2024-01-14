import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define data transformations (you can customize these based on your needs)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to a standard size
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize pixel values
])

data_path = 'C:\\Users\\thamp\\Final Project\\Pytorch-Deepfish\\Deep fish'

# Correct the path by adding the separator between 'Deep fish' and 'train'
train_dataset = datasets.ImageFolder(root=os.path.join(data_path, 'train'), transform=transform)

# Load the DeepFish dataset
train_dataset = datasets.ImageFolder(root=os.path.join(data_path, 'train'), transform=transform)
test_dataset = datasets.ImageFolder(root=os.path.join(data_path, 'test'), transform=transform)

# Create DataLoader instances
batch_size = 32
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Explore sample images and labels
for images, labels in train_loader:
    # Access a batch of images and labels
    print("Image shape:", images.shape)
    print("Labels:", labels)
    break
