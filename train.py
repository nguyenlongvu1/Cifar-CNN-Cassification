import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from model import MyCNN
from visualize import plot_loss

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load data
    full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_size = int(0.8 * len(full_trainset))
    val_size = len(full_trainset) - train_size
    trainset, valset = random_split(full_trainset, [train_size, val_size])

    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    valloader = DataLoader(valset, batch_size=64, shuffle=False, num_workers=2)

    net = MyCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    epochs = 50
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(trainloader)
        train_losses.append(avg_train_loss)

        # Validation
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in valloader:
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(valloader)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    print('Finished Training')
    torch.save(net.state_dict(), "cnn_model.pth")

    # Plot
    plot_loss(train_losses, val_losses)

    print('Finished Training')
    torch.save(net.state_dict(), "cnn_model.pth")


    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in valloader:
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    print(f"Validation Accuracy: {val_accuracy:.2f}%")
