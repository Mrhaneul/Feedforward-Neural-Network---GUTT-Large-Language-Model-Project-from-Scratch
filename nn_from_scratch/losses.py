from lib.tinygradish import Tensor as tsr

# Mean Squared Error Loss
def mse_loss(predictions: tsr, targets: tsr) -> tsr:
    """
    Mean Squared Error:
        loss = mean( (pred - target)^2 )
    pred and target are Tensors of the same shape.
    Returns a scalar Tensor (requires_grad=True).
    """
    
    diff = predictions - targets
    squared_diff = diff * diff
    
    return squared_diff.mean()

