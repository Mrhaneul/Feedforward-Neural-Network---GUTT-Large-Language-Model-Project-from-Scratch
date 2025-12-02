# nn-style framework + tiny autograd

import numpy as np

# Autograd Tensor
class Tensor:
    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        if data.dtype != np.float32:
            data = data.astype(np.float32)

        self.data = data
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        pass

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad})"
    
    # Core graph utilities
    @staticmethod
    def _ensure_tensor(x):
        return x if isinstance(x, Tensor) else Tensor(x)
    
    def zero_grad(self):
        if self.grad is not None:
            self.grad.fill(0.)
        for child in self._prev:
            if isinstance(child, Tensor):
                child.zero_grad()

    # Basic Operations
    def __add__(self, other):
        other = Tensor._ensure_tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or
                     other.requires_grad, _children=(self, other), _op="+")
        def _backward():
            if self.requires_grad:
                self.grad += Tensor._unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += Tensor._unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out
    
    def __radd__(self, other): return self + other

    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return (-self) + other

    def __neg__(self):
        out = Tensor(-self.data, requires_grad=self.requires_grad, _children=(self,), _op="neg")
        def _backward():
            if self.requires_grad:
                self.grad -= out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = Tensor._ensure_tensor(other)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other), _op="*")
        def _backward():
            if self.requires_grad:
                self.grad += Tensor._unbroadcast(other.data * out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += Tensor._unbroadcast(self.data * out.grad, other.data.shape)
        out._backward = _backward
        return out
    
    def __rmul__(self, other): return self * other

    def matmul(self, other):
        other = Tensor._ensure_tensor(other)
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other), _op="@")
        def _backward():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out
    
    def __matmul__(self, other): return self.matmul(other)

    @property
    def T(self):
        out = Tensor(self.data.transpose(), requires_grad=self.requires_grad, _children=(self,), _op="transpose")
        def _backward():
            if self.requires_grad:
                self.grad += out.grad.transpose()
        out._backward = _backward
        return out
    
    def transpose(self, *axes):
        out = Tensor(self.data.transpose(axes), requires_grad=self.requires_grad, _children=(self,), _op="transpose")
        def _backward():
            if self.requires_grad:
                inv_axes = np.argsort(axes)
                self.grad += out.grad.transpose(inv_axes)
        out._backward = _backward
        return out

    def reshape(self, *shape):
        old_shape = self.data.shape
        out = Tensor(self.data.reshape(shape), requires_grad=self.requires_grad, _children=(self,), _op="reshape")
        def _backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(old_shape)
        out._backward = _backward
        return out
    
    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, _children=(self,), _op="sum")
        def _backward():
            if self.requires_grad:
                grad = out.grad
                if axis is not None and not keepdims:
                    # expand back
                    shape = list(self.data.shape)
                    if isinstance(axis, int):
                        axes = [axis]
                    else:
                        axes = sorted(axis if isinstance(axis, (list, tuple)) else [axis])
                    for ax in axes:
                        grad = np.expand_dims(grad, axis=ax)
                self.grad += np.ones_like(self.data) * grad
        out._backward = _backward
        return out
    
    def mean(self, axis=None, keepdims=False):
        denom = self.data.size if axis is None else np.prod(np.array(self.data.shape)[list(axis if isinstance(axis, (list, tuple)) else [axis])])
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / float(denom))

    @staticmethod
    def _unbroadcast(grad, to_shape):
        # Sum gradients across broadcasted dimensions to match to_shape
        while len(grad.shape) > len(to_shape):
            grad = grad.sum(axis=0)
        for i, (gdim, tdim) in enumerate(zip(grad.shape, to_shape)):
            if tdim == 1 and gdim != 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    def relu(self):
        out = Tensor(np.maximum(self.data, 0), requires_grad=self.requires_grad, _children=(self,), _op="relu")
        def _backward():
            if self.requires_grad:
                self.grad += (self.data > 0).astype(np.float32) * out.grad
        out._backward = _backward
        return out
    
    def gelu(self):
        c = np.sqrt(2/np.pi).astype(np.float32)
        x = self.data
        out_data = 0.5 * x * (1 + np.tanh(c * (x + 0.044715 * (x*x*x))))
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,), _op="gelu")
        def _backward():
            if not self.requires_grad: return
            # Derivative of GELU (approx) wrt x
            t = c * (x + 0.044715 * x**3)
            sech2 = 1 - np.tanh(t)**2
            dgelu = 0.5 * (1 + np.tanh(t)) + 0.5 * x * sech2 * c * (1 + 0.134145 * x**2)
            self.grad += dgelu * out.grad
        out._backward = _backward
        return out
    
    def log_softmax(self, axis=-1):
        x = self.data
        x_shift = x - np.max(x, axis=axis, keepdims=True)
        logsumexp = np.log(np.sum(np.exp(x_shift), axis=axis, keepdims=True))
        out = Tensor(x_shift - logsumexp, requires_grad=self.requires_grad, _children=(self,), _op="log_softmax")
        def _backward():
            if not self.requires_grad: return
            grad = out.grad
            # grad of logsoftmax: dL/dx = g - exp(y)*sum(g)
            y = out.data
            sm = np.exp(y)
            sumgrad = np.sum(grad, axis=axis, keepdims=True)
            self.grad += grad - sm * sumgrad
        out._backward = _backward
        return out
    
    def softmax(self, axis=-1):
        y = self.log_softmax(axis=axis)   # y = log softmax(x)
        out_data = np.exp(y.data)         # true softmax values

        out = Tensor(
            out_data,
            requires_grad=y.requires_grad,
            _children=(y,),
            _op="softmax"
        )

        def _backward():
            if not y.requires_grad:
                return

            g = out.grad
            s = out.data

            # Jacobian-vector product:
            # dL/dx = g - s * sum(g * s)
            dot = np.sum(g * s, axis=axis, keepdims=True)

            if y.grad is None:
                y.grad = np.zeros_like(y.data)

            y.grad += g - dot * s

        out._backward = _backward
        return out

    
    # ----- backprop entry -----
    def backward(self):
        # Topo sort
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    if isinstance(child, Tensor):
                        build(child)
                topo.append(v)
        build(self)
        # seed gradient
        if self.grad is None: self.grad = np.zeros_like(self.data)
        self.grad += np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

# nn.Module and params
class Parameter(Tensor):
    def __init__(self, data):
        super().__init__(data, requires_grad=True)

class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self.training = True
        pass

    def add_parameter(self, name, value):
        assert isinstance(value, Tensor)
        self._parameters[name] = value
        return value
    
    def add_module(self, name, module):
        assert isinstance(module, Module)
        self._modules[name] = module
        return module
    
    def parameters(self):
        for p in self._parameters.values():
            yield p
        for m in self._modules.values():
            yield from m.parameters()
    
    def train(self, mode=True):
        self.training = mode
        for m in self._modules.values():
            m.train(mode)
        return self
    
    def eval(self):
        return self.train(False)
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def zero_grad(self):
        for p in self.parameters():
            if p.grad is not None:
                p.grad.fill(0.0)

# Layers
class Linear(Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()

        #Kaiming scale
        W = np.random.randn(in_dim, out_dim).astype(np.float32) / np.sqrt(in_dim)
        self.W = self.add_parameter('W', Parameter(W))
        self.b = self.add_parameter('b', Parameter(np.zeros(out_dim, dtype=np.float32))) if bias else None

    def forward(self, x: Tensor):
        y = x @ self.W
        if self.b is not None:
            y = y + self.b
        return y
    
class ReLU(Module):
    def forward(self, x: Tensor): return x.relu()

class GELU(Module):
    def forward(self, x: Tensor): return x.gelu()

class LayerNorm(Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.gamma = self.add_parameter('gamma', Parameter(np.ones((dim,), dtype=np.float32)))
        self.beta  = self.add_parameter('beta',  Parameter(np.zeros((dim,), dtype=np.float32)))
        self.eps = eps

    def forward(self, x: Tensor):
        # x: (..., D)
        D = x.data.shape[-1]

        # Compute batch statistics (in NumPy)
        mu = np.mean(x.data, axis=-1, keepdims=True)
        var = np.var(x.data, axis=-1, keepdims=True)
        std = np.sqrt(var + self.eps)

        x_mu = x.data - mu
        x_hat = x_mu / std

        # Output tensor
        y = Tensor(
            x_hat * self.gamma.data + self.beta.data,
            requires_grad=x.requires_grad or self.gamma.requires_grad or self.beta.requires_grad,
            _children=(x, self.gamma, self.beta),
            _op="layernorm"
        )

        def _backward():
            gy = y.grad  # gradient w.r.t. output

            if gy is None:
                return

            # gamma / beta grads
            if self.gamma.requires_grad:
                if self.gamma.grad is None:
                    self.gamma.grad = np.zeros_like(self.gamma.data)
                self.gamma.grad += np.sum(gy * x_hat, axis=tuple(range(gy.ndim-1)), keepdims=False)

            if self.beta.requires_grad:
                if self.beta.grad is None:
                    self.beta.grad = np.zeros_like(self.beta.data)
                self.beta.grad += np.sum(gy, axis=tuple(range(gy.ndim-1)), keepdims=False)

            # input gradients (LayerNorm full formula)
            if x.requires_grad:
                if x.grad is None:
                    x.grad = np.zeros_like(x.data)

                gx_hat = gy * self.gamma.data
                gx_mu = gx_hat / std
                gvar = np.sum(gx_hat * x_mu * -0.5 * std**-3, axis=-1, keepdims=True)
                gmu = np.sum(-gx_mu, axis=-1, keepdims=True) + gvar * np.mean(-2 * x_mu, axis=-1, keepdims=True)

                x.grad += gx_mu + (gvar * 2 * x_mu / D) + (gmu / D)

        y._backward = _backward
        return y


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor):
        if not self.training or self.p == 0.0:
            return x
        mask = (np.random.rand(*x.data.shape) > self.p).astype(np.float32) / (1.0 - self.p)
        # Make mask a constant Tensor so it doesn't backprop to randomness
        return x * Tensor(mask)
    
class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        for i, l in enumerate(layers):
            self.add_module(str(i), l)
    
    def forward(self, x: Tensor):
        for m in self._modules.values():
            x = m(x)
        return x
    
# Losses and Uilities
def cross_entropy(logits: Tensor, targets: np.ndarray):
    """
    logits: (N, C)
    targets: int labels shape (N,) in [0..C-1]
    """
    ls = logits.log_softmax(axis=1)
    N = targets.shape[0]

    idx = (np.arange(N), targets)
    picked = Tensor(ls.data[idx], requires_grad=True, _children=(ls,), _op="gather")
    def _backward():
        # scatter back into ls.grad: dL/dy = -1/N at target positions
        if ls.grad is None: ls.grad = np.zeros_like(ls.data)
        ls.grad[idx] += picked.grad
    picked._backward = _backward
    loss = -(picked).mean()
    return loss

# Optimizers
class Optimizer:
    def __init__(self, params): self.params = list(params)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.fill(0.)

class SGD(Optimizer):
    def __init__(self, params, lr=1e-2, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.wd = weight_decay

    def step(self):
        for p in self.params:
            if p.grad is None: continue
            if self.wd != 0.0:
                p.grad += self.wd * p.data
            p.data -= self.lr * p.grad

class Adam(Optimizer):
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params)
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, betas[0], betas[1], eps, weight_decay
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None: continue
            g = p.grad
            if self.wd != 0.0:
                g = g + self.wd * p.data
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)

# Example
if __name__ == "__main__":
    np.random.seed(0)

    # Tiny Model for sanity Checks
    model = Sequential(Linear(16, 64), GELU(),
                       Linear(64, 64), ReLU(),
                       LayerNorm(64),
                       Linear(64, 4)
    )

    X = Tensor(np.random.randn(128, 16).astype(np.float32))
    y = np.random.randint(0, 4, size=(128,))

    