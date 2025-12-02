# fnn.py
from typing import List, Union, Callable

import numpy as np
import matplotlib.pyplot as plt

from lib.tinygradish import Tensor, Module, Linear, GELU, ReLU, Sequential, SGD


# ---------- Model definition (FeedForward from model.py) ----------

class FeedForward(Module):
    """
    A standard 2-layer feed-forward network:
        x -> Linear(d_in, d_hidden) -> act -> Linear(d_hidden, d_out)
    """

    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        d_out: int,
        activation: Union[str, Callable[[Tensor], Tensor]] = "gelu",
    ):
        super().__init__()

        # First and second linear layers
        lin1 = self.add_module("lin1", Linear(d_in, d_hidden))
        lin2 = self.add_module("lin2", Linear(d_hidden, d_out))

        # Activation as a Module using Tensor methods (x.gelu(), x.relu())
        if isinstance(activation, str):
            if activation == "gelu":
                act = self.add_module("act", GELU())
            elif activation == "relu":
                act = self.add_module("act", ReLU())
            else:
                raise ValueError(f"Unsupported activation: {activation}")
        else:
            # If user passes a custom callable: wrap it in a tiny Module
            class LambdaAct(Module):
                def forward(self, x: Tensor) -> Tensor:
                    return activation(x)

            act = self.add_module("act", LambdaAct())

        # Compose into a Sequential module so parameters() works automatically
        self.net = self.add_module("net", Sequential(lin1, act, lin2))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

# ---------- Mean Squared Error Loss ----------

def mse_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    """
    Mean Squared Error:
        mean( (pred - target)^2 )
    """
    diff = predictions - targets
    return (diff * diff).mean()


# ---------- Training loop ----------

def train(
    model: Module,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    batch_size: int,
    optimizer: SGD,
    loss_fn: Callable[[Tensor, Tensor], Tensor] = mse_loss,
):
    """
    Simple training loop.

    model:    Module taking Tensor -> Tensor
    X, y:     numpy arrays
    epochs:   number of passes over the data
    batch_size: mini-batch size
    optimizer: SGD (or Adam) with zero_grad() and step()
    loss_fn:  function(pred: Tensor, target: Tensor) -> scalar Tensor
    """

    n = X.shape[0]
    history: List[float] = []

    for epoch in range(1, epochs + 1):
        # Shuffle data each epoch
        indices = np.random.permutation(n)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        epoch_loss = 0.0
        num_batches = 0

        for start in range(0, n, batch_size):
            end = start + batch_size
            xb_np = X_shuffled[start:end]
            yb_np = y_shuffled[start:end]

            xb = Tensor(xb_np, requires_grad=False)
            yb = Tensor(yb_np, requires_grad=False)

            # Forward
            preds = model(xb)
            loss = loss_fn(preds, yb)

            # Backward + update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.data)
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        history.append(avg_loss)
        print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.6f}")

    return history


# ---------- Experiment: FNN learning sin(x) ----------

if __name__ == "__main__":
    # 1) Create synthetic dataset: y = sin(x) with noise
    np.random.seed(0)

    N = 500  # number of samples
    X = np.linspace(-2 * np.pi, 2 * np.pi, N, dtype=np.float32).reshape(-1, 1)
    y = np.sin(X).astype(np.float32)
    y += 0.1 * np.random.randn(*y.shape).astype(np.float32)  # add noise

    # 2) Build a small FNN
    model = FeedForward(
        d_in=1,
        d_hidden=32,
        d_out=1,
        activation="relu",  # gelu or relu
    )

    optimizer = SGD(model.parameters(), lr=0.001)

    # 3) Train the model
    epochs = 1000
    batch_size = N  # full-batch; change to 64 for mini-batch

    loss_history = train(
        model=model,
        X=X,
        y=y,
        epochs=epochs,
        batch_size=batch_size,
        optimizer=optimizer,
        loss_fn=mse_loss,
    )

    # 4) Plot training loss
    plt.figure()
    plt.plot(loss_history)
    plt.title("Training Loss (y = sin(x))")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)

    # 5) Plot true function vs model prediction
    X_tsr = Tensor(X, requires_grad=False)
    preds = model(X_tsr).data

    plt.figure()
    plt.plot(X, y, label="True sin(x)")
    plt.plot(X, preds, label="FNN prediction", linestyle="dashed")
    plt.title("FNN approximating sin(x)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)

    plt.show()

    """
    Loss going down → model is learning
    Loss flattening → training stable
    Loss exploding → learning rate too high
    Loss bouncing → unstable updates
    """
