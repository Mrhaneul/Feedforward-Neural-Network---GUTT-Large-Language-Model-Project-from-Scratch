class SGD:
    # Stochastic Gradient Descent Optimizer
    def __init__(self, params, lr: float = 0.01):
        """
        params: iterable of Tensors with .grad
        lr: learning rate
        """
        
        # Materialize list in order to iterate multiple times
        self.params = list(params)
        self.lr = lr
        
    def step(self):
        """
        Perform a single optimization step:
            p = p - lr * grad
        """
        for p in self.params:
            if p.grad is None:
                continue
            # Update parameter in place
            p.data -= self.lr * p.grad

    def zero_grad(self):
        """
        Set gradients of all parameters to zero.
        """
        for p in self.params:
            if p.grad is not None:
                # Not using zero_grad() from Tensor to clear only this parameter's gradient
                p.grad.fill(0.0)