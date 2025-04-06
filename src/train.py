import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

def train_model(model, X_train, y_train, X_val, y_val, epochs=5, batch_size=1000, learning_rate=0.001):
    """
    Train the model on CIFAR-10 data.
    
    Args:
        model: The model to train
        X_train: Training images
        y_train: Training labels
        X_val: Validation images
        y_val: Validation labels
        epochs: Number of training epochs
        batch_size: Number of samples per batch
        learning_rate: Learning rate for optimizer
        
    Returns:
        tuple: (trained state, training history)
    """
    rng = jax.random.PRNGKey(0)
    
    # Initialize parameters with correct input shape
    params = model.init(rng, jnp.ones([1, 32, 32, 3]))["params"]
    
    # Create optimizer
    optimizer = optax.adam(learning_rate=learning_rate)
    
    # Create training state
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer
    )
    
    # Track metrics
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    # Define metrics function
    @jax.jit
    def compute_metrics(params, X, y):
        logits = model.apply({"params": params}, X)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        accuracy = jnp.mean(jnp.argmax(logits, -1) == y)
        return loss, accuracy
    
    # Training loop
    print("Starting training...")
    for epoch in range(epochs):
        # Compute gradients and update parameters
        grad_fn = jax.grad(lambda p, x, y: compute_metrics(p, x, y)[0], has_aux=False)
        grads = grad_fn(state.params, X_train[:batch_size], y_train[:batch_size])
        state = state.apply_gradients(grads=grads)
        
        # Compute metrics
        train_loss, train_acc = compute_metrics(state.params, X_train[:batch_size], y_train[:batch_size])
        val_loss, val_acc = compute_metrics(state.params, X_val[:batch_size], y_val[:batch_size])
        
        # Store metrics
        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['train_acc'].append(float(train_acc))
        history['val_acc'].append(float(val_acc))
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train Acc: {train_acc:.2%}, Val Acc: {val_acc:.2%}\n")
    
    return state, history