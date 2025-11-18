from typing import List, Callable, Union
from lib.tinygradish import Tensor as tsr
from nn_from_scratch.layers import Linear, relu, gelu

# LayerLike: Define a type alias for layers
LayerLike = Union[Callable,[[tsr], tsr], Linear]

class Sequential:
    """
    Simple container that applies a list of layers/functions in order.
    Example usage:
        model = Sequential([Linear(2, 4), relu, Linear(4, 1)])
        y = model(x)
    """
    
    # Initialize with a list of layers
    def __init__(self, layers: List[LayerLike]):
        self.layers = layers

    # Make the model callable
    def __call__(self, x: tsr) -> tsr:
        return self.forward(x)

    # Define forward method
    def forward(self, x: tsr) -> tsr:
        for layer in self.layers:
            x = layer(x)    # For both Linear layers and activation functions
        return x

    def parameters(self) -> List[tsr]:
        """
        Collect parameters (weights/biases) from all sub-layers that have .parameters
        Activation functions (relu/gelu) have no parameters, so they’re skipped.
        """
        params = []
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                # Linear.parameters is a @property returning a list [W, b]
                params.extend(layer.parameters)
        return params
    
class FeedForward:
    """
    A standard 2-layer feed-forward network:
        x -> Linear(d_in, d_hidden) -> act -> Linear(d_hidden, d_out)
    """
    
    # Initialize with input, hidden, output dimensions and activation function
    # Default activation is GELU
    def __init__(self, d_in: int, d_hidden: int, d_out: int, activation: Callable = gelu):
        if activation == "gelu":
            activation = gelu
        elif activation == "relu":
            activation = relu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Create the sequential model
        self.net = Sequential([
            # x -> Linear(d_in, d_hidden) -> act -> Linear(d_hidden, d_out)
            Linear(d_in, d_hidden),
            activation,
            Linear(d_hidden, d_out)
        ])
    
    # Make the model callable
    def __call__(self, x: tsr) -> tsr:
        return self.forward(x)
    
    # Define forward method
    def forward(self, x: tsr) -> tsr:
        return self.net(x)
    
    # Get all parameters of the network
    def parameters(self) -> List[tsr]:
        return self.net.parameters()