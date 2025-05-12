# CIFAR-10 Image Classification with CNN (PyTorch)

This project trains a Convolutional Neural Network (CNN) on the CIFAR-10 dataset using PyTorch. It includes features like training-validation splitting, loss visualization, learning rate scheduling, and test evaluation with a confusion matrix.

## 📁 Project Structure

├── train.py # Training script

├── evaluate.py # Evaluate model on test set

├── model.py # CNN model definition

├── visualize.py # Functions for plotting loss and confusion matrix

├── cnn_model.pth # Saved trained model

├── README.md # This file

└── data/ # CIFAR-10 dataset will be downloaded here

## 🧠 Model Architecture

```text
Conv2D(3 → 32) + ReLU + BN
Conv2D(32 → 64) + ReLU + BN + MaxPool
Conv2D(64 → 128) + ReLU + BN
Conv2D(128 → 128) + ReLU + BN + MaxPool
AdaptiveAvgPool → Flatten
Dropout → Linear(128→128) → ReLU
Dropout → Linear(128→10)
```text

## ✅ Example Result

- ✅ **Validation Accuracy**: ~82.55%
- ✅ **Test Accuracy**: *(shown after running `evaluate.py`)*
- ✅ **Confusion Matrix**: Displayed using seaborn heatmap

## 📈 Training/Validation Loss Example

> *(You can add a screenshot of the training/validation loss plot here)*

## 📌 Notes

- 📉 Learning rate is automatically reduced when validation loss plateaus using `ReduceLROnPlateau`
- 🔀 Dataset is split into 80% training and 20% validation
- 💾 You can continue training from a saved model (`cnn_model.pth`) if needed

