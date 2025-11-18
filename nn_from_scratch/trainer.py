# Dummy trainer module for testing purposes

import numpy as np
from lib.tinygradish import Tensor as tsr

# Trains the given model using the provided data, optimizer, and loss function.
def train(
    model,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    batch_size: int,
    optimizer,
    loss_fn,
):
    """
    Simple training loop for FNN.

    model:    callable, e.g., FeedForward or Sequential-wrapped model
    X, y:     numpy arrays (data and targets)
    epochs:   number of passes over the data
    batch_size: mini-batch size
    optimizer: instance of SGD (or compatible) with step() and zero_grad()
    loss_fn:  function taking (pred: Tensor, target: Tensor) -> Tensor (scalar loss)
    """
    
    # Number of samples
    n = X.shape[0]
    
    # To store loss history
    history = []
    
    for epoch in range(1, epochs + 1):
        # Shuffle indices for each epoch
        indices = np.random.permutation(n)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # Ephoch loss accumulator
        epoch_loss = 0.0
        # Number of batches processed
        num_batches = 0
        
        # Mini-batch training
        for start in range(0, n, batch_size):
            end = start + batch_size
            # NP arrays
            xb_np = X_shuffled[start:end]
            yb_np = y_shuffled[start:end]
            
            # Wrap batches in Tensors
            xb = tsr(xb_np, requires_grad=False)
            yb = tsr(yb_np, requires_grad=False)
            
            # Forward pass
            preds = model(xb)
            loss = loss_fn(preds, yb)
            
            # Backward pass + update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update epoch loss
            epoch_loss += float(loss.data)
            num_batches += 1
            
        # Average loss for the epoch
        avg_loss = epoch_loss / num_batches
        # Log progress
        history.append(avg_loss)
        print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.6f}")
        
    return history