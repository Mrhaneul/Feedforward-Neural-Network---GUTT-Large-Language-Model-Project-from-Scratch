import numpy as np
from lib.tinygradish import Tensor as tsr

# Linear Layer
# Implements a fully connected layer: y = xW + b
class Linear:
    def __init__(self, in_features: int, out_features: int):
        # Weights initialization
        # Using small random values for weights
        W_data = 0.01 * np.random.randn(in_features, out_features).astype(np.float32)
        self.W = tsr(W_data, requires_grad=True)
        
        # Bias initialization
        # Initialized to zeros
        b_data = np.zeros((1, out_features), dtype=np.float32)
        self.b = tsr(b_data, requires_grad=True)

    # Make the layer callable
    def __call__(self, x: tsr) -> tsr:
        return self.forward(x)
    
    # Define forward method
    def forward(self, x: tsr) -> tsr:
        # output = xW + b
        # Note: @ operator is used for matrix multiplication
        out = x @ self.W
        if self.b is not None:
            out = out + self.b
        return out
    
    # Return parameters as a list
    @property
    def parameters(self):
        return [self.W, self.b]

    

# ReLU Activation Layer
def relu(x: tsr) -> tsr:
    return x.relu()

# GELU Activation Layer
def gelu(x: tsr) -> tsr:
    return x.gelu()