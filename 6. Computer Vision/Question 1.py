"""
# Submitted by: Mohammad Wasiq

## Email: gl0427@myamu.ac.in

# Placement Computer Vision Assignment 1
"""

"""
Q1.- Train a deep learning model which would classify the vegetables based on the images provided. The dataset can be accessed from the given link.

Link-
https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset

Note -
1. Use PyTorch as the framework for training model
2. Use Distributed Parallel Training technique to optimize training time.
3. Achieve an accuracy of at least 85% on the validation dataset.
4. Use albumentations library for image transformation
5. Use TensorBoard logging for visualizing training performance
6. Use custom modular Python scripts to train model
7. Only Jupyter notebooks will not be allowed
8. Write code comments wherever needed for understanding
"""

# Ans:

import os
from torchvision.datasets import ImageFolder

# Path to the downloaded and extracted dataset folder
dataset_folder = r"Vegetable Images"

# Path to the train and validation folders within the dataset folder
train_folder = os.path.join(dataset_folder, "train")
val_folder = os.path.join(dataset_folder, "val")

# Define the transforms for data augmentation using Albumentations library
# Example transforms: random crop, horizontal flip
import albumentations as A
from torchvision.transforms import ToTensor

# Define the Albumentations transformations
transform = A.Compose([
    A.RandomCrop(height=224, width=224),
    A.HorizontalFlip(),
    ToTensor()
])

# Load the train and validation datasets using ImageFolder
train_dataset = ImageFolder(train_folder, transform=transform)
val_dataset = ImageFolder(val_folder, transform=transform)

import torch
import torch.nn as nn

class VegetableClassifier(nn.Module):
    def __init__(self, num_classes):
        super(VegetableClassifier, self).__init__()
        
        # Define the backbone architecture
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Define the fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

# Define the number of GPUs available for training
num_gpus = torch.cuda.device_count()

# Initialize the VegetableClassifier model
model = VegetableClassifier(num_classes=10)

# Check if multiple GPUs are available
if num_gpus > 1:
    # Spawn multiple processes for distributed training
    mp.spawn(train_fn, nprocs=num_gpus, args=(model,))
else:
    # Single GPU training
    train_fn(0, model)

# Define the training function to be executed on each GPU
def train_fn(rank, model):
    # Initialize the distributed backend
    dist.init_process_group(backend='nccl', init_method='env://')

    # Set the device based on the current process rank
    torch.cuda.set_device(rank)

    # Create a distributed model
    model = model.to(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # Define the optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Start the training loop
    for epoch in range(num_epochs):
        # Train the model
        model.train()
        for images, labels in train_loader:
            images = images.to(rank)
            labels = labels.to(rank)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Perform validation
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in val_loader:
                images = images.to(rank)
                labels = labels.to(rank)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total

            # Print and log the accuracy
            print(f"Epoch [{epoch+1}/{num_epochs}], Accuracy: {accuracy:.2f}%")

    # Clean up the distributed training environment
    dist.destroy_process_group()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Initialize the VegetableClassifier model
model = VegetableClassifier(num_classes=10)

# Define the optimizer and criterion
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Set the device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the number of training epochs
num_epochs = 10

# Create a TensorBoard writer for logging
writer = SummaryWriter(log_dir="logs")

# Start the training loop
for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss /= len(train_loader)
    train_accuracy = 100.0 * correct / total

    # Validation phase
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss /= len(val_loader)
    val_accuracy = 100.0 * correct / total

    # Print and log the training performance
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    # Log the training performance to TensorBoard
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Accuracy/train", train_accuracy, epoch)
    writer.add_scalar("Loss/validation", val_loss, epoch)
    writer.add_scalar("Accuracy/validation", val_accuracy, epoch)

# Close the TensorBoard writer
writer.close()

import torch

# Save the trained model
torch.save(model.state_dict(), "vegetable_classifier.pt")

# Load the saved model for inference
loaded_model = VegetableClassifier(num_classes=10)
loaded_model.load_state_dict(torch.load("vegetable_classifier.pt"))
loaded_model.eval()

# Perform inference on a sample image
sample_image = ...  # Load or preprocess the sample image
input_tensor = torch.unsqueeze(sample_image, 0)  # Add batch dimension
output = loaded_model(input_tensor)
predicted_class = torch.argmax(output, dim=1)

# Convert predicted class to label
label = class_labels[predicted_class.item()]
print(f"Predicted label: {label}")

import boto3
import base64
import json

# Initialize AWS clients
lambda_client = boto3.client('lambda')
s3_client = boto3.client('s3')

# Define the AWS Lambda function name and input payload
function_name = 'vegetable_classifier_lambda'
payload = {
    'image': 'base64_encoded_image'
}

# Convert the image to base64
with open('sample_image.jpg', 'rb') as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
payload['image'] = encoded_image

# Invoke the AWS Lambda function for inference
response = lambda_client.invoke(
    FunctionName=function_name,
    InvocationType='RequestResponse',
    Payload=json.dumps(payload)
)

# Process the inference response
if response['StatusCode'] == 200:
    inference_result = json.loads(response['Payload'].read())
    predicted_class = inference_result['predicted_class']
    confidence = inference_result['confidence']
    print(f"Predicted class: {predicted_class}, Confidence: {confidence}")
else:
    print("Error occurred during inference")