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

## 📊 Visualizations

- Training and validation loss per epoch
  ![Training Loss](./loss.png)
- Confusion matrix on test set
  ![Training Loss](./confusion_matrix.png)
## ✅ Example Results

- **Test Accuracy (Custom CNN)**: ~79.95%  

---

## 🔧 Model Improvement Suggestions (for Custom CNN)

To improve the performance of the custom CNN, consider the following enhancements:

- ✅ **Data Augmentation**: Add transforms such as `RandomCrop`, `RandomHorizontalFlip`, or `ColorJitter` to improve generalization.
- ✅ **Increase Depth or Width**: Add more convolutional layers or increase the number of filters.
- ✅ **Use Residual Connections**: Incorporate skip connections inspired by ResNet to help training deeper networks.
- ✅ **Replace Fully Connected Layers with Global Pooling**: Use `GlobalAvgPool2d` + `Linear(128 → 10)` for more efficient representation.
- ✅ **Label Smoothing**: A regularization technique that can help reduce overconfidence of the model.
- ✅ **Advanced Optimizers**: Try optimizers like `SGD with momentum`, `RAdam`, or `AdamW`.
- ✅ **Learning Rate Scheduling**: Already used `ReduceLROnPlateau`, but other schedulers like `CosineAnnealingLR` may improve convergence.

---

## 🏗️ Transfer Learning with ResNet-18 (Pretrained)

To further improve performance with fewer epochs, we also experimented with a pretrained **ResNet-18** model from `torchvision.models`.

### Modifications:
- Resized input images to 224×224.
- Replaced the final FC layer with `nn.Linear(512, 10)` for CIFAR-10.
- Frozen all convolutional layers and fine-tuned only the fully connected layer.

### Results:
- **Test Accuracy**: ~82.55% after only 6 epochs of training.
- Demonstrates the power of transfer learning for small datasets.

---



