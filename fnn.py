import numpy as np
from lib.tinygradish import Tensor as tsr
from nn_from_scratch.model import FeedForward
from nn_from_scratch.losses import mse_loss
from nn_from_scratch.optimizers import SGD

# 1) XOR dataset
# Inputs:  (0,0), (0,1), (1,0), (1,1)
# Outputs: 0,      1,      1,      0
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
], dtype=np.float32)

y = np.array([
    [0],
    [1],
    [1],
    [0],
], dtype=np.float32)

# 2) Build a small FNN
model = FeedForward(d_in=2, d_hidden=4, d_out=1, activation="relu")
optimizer = SGD(model.parameters(), lr=0.1)

# 3) Training loop
epochs = 2000

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

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, loss={loss.data:.6f}")

# 4) Evaluate on XOR inputs
print("\nTesting on XOR inputs:")
xb = tsr(X, requires_grad=False)
preds = model(xb)
pred_vals = preds.data  # numpy array

for inp, target, pred in zip(X, y, pred_vals):
    # Threshold at 0.5 to turn regression output into 0/1
    pred_label = 1 if pred[0] > 0.5 else 0
    print(f"Input: {inp}, Target: {int(target[0])}, Pred: {pred[0]:.4f}, Pred_label: {pred_label}")
