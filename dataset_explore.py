

import torch
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

data_path = 'C:\\Users\\thamp\\Final Project\\Pytorch-Deepfish\\Deep fish'

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load the DeepFish dataset
# Correct the path by adding the separator between 'Deep fish' and 'train'
train_dataset = datasets.ImageFolder(root=os.path.join(data_path, 'train'), transform=transform)
# Correct the path by adding the separator between 'Deep fish' and 'test'
test_dataset = datasets.ImageFolder(root=os.path.join(data_path, 'test'), transform=transform)


# Create DataLoader instances
batch_size = 32
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Explore sample images and labels
sample_images, sample_labels = next(iter(train_loader))
print("Image shape:", sample_images.shape)
print("Labels:", sample_labels)

# Visualize sample images
plt.figure(figsize=(12, 6))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(sample_images[i].permute(1, 2, 0))  # permute to (H, W, C) for proper display
    plt.title(f"Class: {sample_labels[i]}")
    plt.axis("off")

plt.show()

import matplotlib.pyplot as plt
import numpy as np

def show_images(images, labels, title="Sample Images", num_images=8):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))

    for i in range(num_images):
        image = images[i].numpy().transpose((1, 2, 0))
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        image = std * image + mean
        image = np.clip(image, 0, 1)
        axes[i].imshow(image)
        axes[i].set_title(f"Label: {labels[i].item()}")
        axes[i].axis("off")

    plt.suptitle(title)
    plt.show()

# Rest of your code...


# Explore sample images and labels
sample_images, sample_labels = next(iter(train_loader))
show_images(sample_images, sample_labels, title="Sample Images from the Dataset")
plt.show(block=True)
input("Press Enter to close the program.")
