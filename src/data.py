import jax.numpy as jnp
import numpy as np
from torchvision.datasets import CIFAR10
from sklearn.model_selection import train_test_split

def get_data():
    """
    Load and preprocess CIFAR-10 dataset.
    
    Returns:
        tuple: ((X_train, y_train), (X_val, y_val), (X_test, y_test))
               Data in correct shape: (batch, height, width, channels)
    """
    train = CIFAR10(root="./data", train=True, download=True)
    test = CIFAR10(root="./data", train=False, download=True)
    
    # Convert to JAX arrays and normalize (NHWC format)
    X_train = jnp.array(train.data) / 255.0  # Shape: (50000, 32, 32, 3)
    y_train = jnp.array(train.targets)
    X_test = jnp.array(test.data) / 255.0    # Shape: (10000, 32, 32, 3)
    y_test = jnp.array(test.targets)
    
    # Split train into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        np.array(X_train), np.array(y_train), test_size=0.2, random_state=42
    )
    
    return (jnp.array(X_train), jnp.array(y_train)), (jnp.array(X_val), jnp.array(y_val)), (X_test, y_test)

# CIFAR-10 class names
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']