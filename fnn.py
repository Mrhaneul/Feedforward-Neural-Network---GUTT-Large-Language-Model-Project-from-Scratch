import numpy as np
from lib.tinygradish import Tensor as tsr
from nn_from_scratch.model import FeedForward
from nn_from_scratch.losses import mse_loss
from nn_from_scratch.optimizers import SGD
import matplotlib.pyplot as plt

# 1) Large synthetic dataset (using y=sin(x))
np.random.seed(0)

N = 500 # number of samples
# Linear space from -2pi to 2pi
X = np.linspace(-2 * np.pi, 2 * np.pi, N, dtype=np.float32).reshape(-1, 1)
# Create a 2D input by duplicating X
y = np.sin(X).astype(np.float32)

# Add a bit of noise
y += 0.1 * np.random.randn(*y.shape).astype(np.float32)

# 2) Build a small FNN
model = FeedForward(
    d_in=1, 
    d_hidden=32, 
    d_out=1, 
    activation="relu",
    # options: "relu", "gelu", "sigmoid"
    )
optimizer = SGD(model.parameters(), lr=0.1)

# 3) Training loop
epochs = 1000

loss_history = []

for epoch in range(1, epochs + 1):
    xb = tsr(X, requires_grad=False)
    yb = tsr(y, requires_grad=False)

    # Forward
    preds = model(xb)
    loss = mse_loss(preds, yb)

    # Backward + update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Record loss
    loss_history.append(loss.data)

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, loss={loss.data:.6f}")

# 4) Plot training loss
plt.figure()
plt.plot(loss_history)
plt.title("Training Loss (y = sin(x))")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)

# 5) Plot true function vs model prediction
with_no_grad_x = tsr(X, requires_grad=False)
preds = model(with_no_grad_x).data

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