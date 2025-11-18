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

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad})"
    
    # Core graph utilities
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
        out_data = 0.5 * x * (1 + np.tanh(c * (x + 0.044715 * (x**3))))
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
    
    
    # Extra utility functions

    #  Global back propagation
    def backward(self, grad=None):
        if not self.requires_grad:
            return
        
        # Topological order of nodes
        topo = []   # list of nodes in topological order
        visited = set() # set of visited nodes
        
        # If v not visited, visit all its children first
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    # Only visit if child is a Tensor
                    if isinstance(child, Tensor):
                        build_topo(child)
                topo.append(v)
        build_topo(self)
        
        # Self gradient as the output
        # If grad is None, assume gradient of 1 (for scalar output)
        if self.grad is None: 
            self.grad = np.zeros_like(self.data)
        # If grad is provided, use it; else use ones
        if grad is None:
            self.grad += np.ones_like(self.data)
        else:
            self.grad += grad
            
        # Backward pass
        for v in reversed(topo):
            v._backward()
