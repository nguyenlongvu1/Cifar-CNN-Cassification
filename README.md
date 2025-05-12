# CIFAR-10 Image Classification with CNN (PyTorch)

This project trains a Convolutional Neural Network (CNN) on the CIFAR-10 dataset using PyTorch. It includes features like training-validation splitting, loss visualization, learning rate scheduling, and test evaluation with a confusion matrix.

## 📁 Project Structure
```
├── train.py # Training script
├── evaluate.py # Evaluate model on test set
├── model.py # CNN model definition
├── visualize.py # Functions for plotting loss and confusion matrix
├── cnn_model.pth # Saved trained model
├── README.md # This file
└── data/ # CIFAR-10 dataset will be downloaded here
```

## 🧠 Model Architecture
```
Conv2D(3 → 32) + ReLU + BatchNorm
Conv2D(32 → 64) + ReLU + BatchNorm + MaxPool
Conv2D(64 → 128) + ReLU + BatchNorm
Conv2D(128 → 128) + ReLU + BatchNorm + MaxPool
AdaptiveAvgPool2d → Flatten
Dropout → Linear(128 → 128) → ReLU
Dropout → Linear(128 → 10)
```
## ✅ Example Results

- **Test Accuracy**: ~82.55%

