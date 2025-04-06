import jax

from src.model import CIFAR10_CNN
from src.data import get_data
from src.train import train_model
from src.visualization import (
    visualize_results, 
    plot_training_curves, 
    print_classification_metrics
)

def main():
    # Check if JAX is using GPU
    print(f"JAX is running on: {jax.devices()[0]}")
    
    # Load model
    model = CIFAR10_CNN()
    
    # Load data
    print("Loading CIFAR-10 dataset...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_data()
    print(f"Data shapes: X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
    
    # Train model
    state, history = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=5
    )
    
    # Visualize results
    plot_training_curves(history)
    visualize_results(model, state.params, X_test, y_test)
    print_classification_metrics(model, state.params, X_test, y_test)
    
    print("Training and evaluation complete!")

if __name__ == "__main__":
    main()