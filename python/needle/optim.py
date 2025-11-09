"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for param in self.params:
            if param.grad is None:
                continue
            if param not in self.u:
                self.u[param] = ndl.zeros_like(param.grad, requires_grad=False)
            self.u[param] = self.momentum * self.u[param].data + \
                (1 - self.momentum) * \
                (self.weight_decay * param.data + param.grad.data)
            param.data -= self.lr * self.u[param].data
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        Note: This does not need to be implemented for HW2 and can be skipped.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        for param in self.params:
            if param.grad is None:
                continue
            if param not in self.m:
                self.m[param] = ndl.zeros_like(param.grad, requires_grad=False)
            if param not in self.v:
                self.v[param] = ndl.zeros_like(param.grad, requires_grad=False)
            self.m[param] = self.beta1 * self.m[param].data + \
                (1 - self.beta1) * (param.grad.data + self.weight_decay * param.data)
            self.v[param] = self.beta2 * self.v[param].data + \
                (1 - self.beta2) * (param.grad.data +
                                    self.weight_decay * param.data) ** 2
            m_hat = self.m[param].data / (1 - self.beta1 ** (self.t + 1))
            v_hat = self.v[param].data / (1 - self.beta2 ** (self.t + 1))
            delta = self.lr * m_hat.data / (v_hat.data ** 0.5 + self.eps)
            param.data -= delta.data
        self.t += 1
        ### END YOUR SOLUTION