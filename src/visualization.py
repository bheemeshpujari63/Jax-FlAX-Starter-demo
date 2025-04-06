import matplotlib.pyplot as plt
import jax.numpy as jnp
from sklearn.metrics import classification_report
from src.data import CLASSES


def visualize_results(model, params, X_test, y_test, num_samples=5):
    """
    Show test images with predictions.
    
    Args:
        model: The model to evaluate
        params: Model parameters
        X_test: Test images
        y_test: Test labels
        num_samples: Number of samples to visualize
    """
    # Get predictions
    logits = model.apply({"params": params}, X_test[:num_samples])
    preds = jnp.argmax(logits, axis=1)
    
    plt.figure(figsize=(15, 5))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(X_test[i])
        plt.title(f"True: {CLASSES[y_test[i]]}\nPred: {CLASSES[preds[i]]}")
        plt.axis('off')
    plt.show()

def plot_training_curves(history):
    """
    Plot loss curves from training history.
    
    Args:
        history: Dictionary containing training metrics
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

def print_classification_metrics(model, params, X_test, y_test, max_samples=1000):
    """
    Print classification report for the model.
    
    Args:
        model: The model to evaluate
        params: Model parameters
        X_test: Test images
        y_test: Test labels
        max_samples: Maximum number of samples to evaluate
    """
    test_logits = model.apply({"params": params}, X_test[:max_samples])
    test_preds = jnp.argmax(test_logits, axis=1)
    
    print("\nClassification Report:")
    print(classification_report(y_test[:max_samples], test_preds, target_names=CLASSES))