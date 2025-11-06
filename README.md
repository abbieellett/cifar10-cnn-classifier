# CIFAR-10 CNN Classifier

A convolutional neural network (CNN) implemented in PyTorch to classify images from the CIFAR-10 dataset.

## Project Overview
This project demonstrates a complete CNN workflow:
- **Model Architecture**: 4 convolutional layers with batch normalization, max-pooling, dropout, and 2 fully-connected layers.
- **Dataset**: CIFAR-10 (60,000 32x32 color images in 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Training**: Uses data augmentation, Adam optimizer, and a learning rate scheduler.
- **Evaluation**: Test accuracy measured on CIFAR-10 test set.

## Tech Stack
- Python 3  
- PyTorch  
- Torchvision  
- NumPy  

## File Structure
- `model.py` → defines the CNN model  
- `train.py` → training script  
- `test.py` → testing and evaluation script  
- `.gitignore` → ignored files  
- `model.pth` → trained model weights (optional)  

## Usage

### 1. Install dependencies
```bash
pip install torch torchvision numpy
```

### 2. Train the model
```bash
python train.py
```

### 3. Test the model
```bash
python test.py
```

### 4. Expected Output
Example test accuracy:
```bash
Test Accuracy: 74.35%
```

## Results
- The model achieves ~78% test accuracy on CIFAR-10 using CPU training.
- Training on GPU is recommended for faster results.

## Potential Improvements
- Add random rotation, color jitter, cutout or mixup to improve generalisation.
- Plot training/test loss curves, confusion matrix or sample predictions to better understand performance.
- Save the model with the best validation accuracy and implement early stopping to prevent overfitting.
