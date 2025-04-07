# Jax-FlAX-Starter-demo  
# 🚀 JAX/Flax CIFAR-10 Classifier

A minimal implementation of a **Convolutional Neural Network (CNN)** for CIFAR-10 classification using **JAX and Flax**, designed to demonstrate:
- How JAX's automatic differentiation (`grad`) and Just-In-Time compilation (`jit`) work.
- Flax's neural network API (vs. TensorFlow/PyTorch).
- Best practices for training/evaluating models in JAX.

**Use Case**:  
This project serves as a **reference implementation** for researchers/developers exploring JAX/Flax, and as a starting point for my GSoC proposal on LLM implementations.

---

## 🛠️ Installation
1. **Install dependencies** (JAX + Flax + data tools):
   ```bash
   pip install jax flax optax torch torchvision scikit-learn 

2. Clone this repository:
   ```
   git clone https://github.com/bheemeshpujari63/cifar10-classifier.git
   cd cifar10-classifier
   ```

## Overview

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck), with 6,000 images per class. This project builds a simple CNN model to classify these images using Jax and Flax.

## Features

- CNN architecture implemented with Flax Linen
- Complete training pipeline with JAX
- Validation metrics tracked during training
- Visualization tools for training curves and model predictions
- Classification report for model evaluation

## Requirements

- Python 3.8+
- JAX and Flaxlib
- Optax for optimization
- Matplotlib for visualization
- Scikit-learn for evaluation metrics
- PyTorch/TorchVision for data loading (CIFAR-10)


## Project Structure

```
cifar10_classifier/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── model.py       # Model definition
│   ├── data.py        # Data loading and preprocessing
│   ├── train.py       # Training logic
│   └── visualization.py  # Visualization utilities
└── main.py            # Entry point
```

## Usage

To train the model and evaluate on the test set:

```
python main.py
```

## Model Architecture

The model is a simple CNN with the following structure:
- Conv layer (32 filters, 3x3 kernel) + ReLU + MaxPool
- Conv layer (64 filters, 3x3 kernel) + ReLU + MaxPool
- Flatten layer
- Dense layer (256 units) + ReLU
- Dense layer (10 units) for classification

## Results

The model achieves approximately 65-70% accuracy on the CIFAR-10 test set with just 5 epochs of training.

output:
```
Epoch 5/5: Train Loss: 0.8943, Val Loss: 0.9021
Train Acc: 68.20%, Val Acc: 67.80%

Classification Report:
              precision    recall  f1-score   support
   airplane       0.68      0.67      0.67       100
 automobile       0.72      0.76      0.74       100
       bird       0.59      0.53      0.56       100
        cat       0.52      0.49      0.51       100
       deer       0.65      0.63      0.64       100
        dog       0.59      0.57      0.58       100
       frog       0.73      0.78      0.75       100
      horse       0.68      0.71      0.69       100
       ship       0.75      0.79      0.77       100
      truck       0.72      0.73      0.72       100
```

## Future Improvements

- Implement data augmentation
- Try more complex model architectures
- Add learning rate scheduling
- Support for hyperparameter tuning
- Add model checkpointing

## License

MIT

## Acknowledgments

- JAX team for their excellent library
- Flax team for the neural network library
- The creators of the CIFAR-10 dataset
