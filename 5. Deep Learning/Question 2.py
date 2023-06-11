"""
# Submitted by: Mohammad Wasiq

## Email: gl0427@myamu.ac.in

# Placement Deep Learning Assignment 2
"""

"""
Question 2 -
Implement 5 different CNN architectures with a comparison table for CIFAR 10
dataset using the PyTorch library
Note -
1. The model parameters for each architecture should not be more than 10000
parameters
2 Code comments should be given for proper code understanding
"""

# Ans:

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 10
batch_size = 100
learning_rate = 0.001

# CIFAR-10 dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=False)

# Function to count the number of parameters in a model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Define a CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(8 * 8 * 8, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# Define the model architectures
architectures = [
    ('Architecture 1', Net()),
    ('Architecture 2', nn.Sequential(
        nn.Conv2d(3, 2, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(2, 4, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(4 * 8 * 8, 10)
    )),
    ('Architecture 3', nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(4 * 8 * 8, 10)
    )),
    ('Architecture 4', nn.Sequential(
        nn.Conv2d(3, 2, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(2 * 8 * 8, 10)
    )),
    ('Architecture 5', nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(4, 2, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(2 * 8 * 8, 10)
    ))
]

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()

# Compare architectures
comparison_table = []
for name, model in architectures:
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    total_params = count_parameters(model)

if total_params <= 10000:  # Check if total parameters are less than or equal to 10,000
        # Training loop
        for epoch in range(num_epochs):
            total_loss = 0
            correct = 0
            total = 0
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Print epoch-wise accuracy and loss
            print(f'Architecture: {name}, Epoch: {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {100*correct/total:.2f}%')

# Test the model
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        comparison_table.append((name, total_params, accuracy))

# Print the comparison table
print("\nComparison Table:")
print("-------------------------------------------------------------")
print("| Architecture    | Parameters    | Accuracy (%) |")
print("-------------------------------------------------------------")
for name, params, accuracy in comparison_table:
    print(f"| {name:<15} | {params:<13} | {accuracy:<12.2f} |")
print("-------------------------------------------------------------")