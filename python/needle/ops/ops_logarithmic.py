from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 


class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        # BEGIN YOUR SOLUTION
        Z -= Z.max(axis=1, keepdims=True)
        return Z - array_api.log(array_api.sum(array_api.exp(Z), axis=1, keepdims=True))
        # END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        # BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        softmax = exp(logsoftmax(Z))
        out_grad_sum = out_grad.sum(axes=(1,)).reshape(
            (out_grad.shape[0], 1)).broadcast_to(softmax.shape)
        return out_grad - softmax * out_grad_sum
        # END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        # BEGIN YOUR SOLUTION
        if self.axes is None:
            self.axes = tuple(range(len(Z.shape)))
        self.max_z = Z.max(axis=self.axes, keepdims=True)
        result = array_api.log(array_api.sum(array_api.exp(
            Z - self.max_z.broadcast_to(Z.shape)), self.axes))
        return result + self.max_z.reshape(result.shape)
        # END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        # BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        shape = list(Z.shape)
        axes = self.axes if self.axes is not None else tuple(range(len(shape)))
        for axis in axes:
            shape[axis] = 1
        shape = tuple(shape)
        diff = Z - self.max_z.broadcast_to(Z.shape)
        gradient = exp(diff) / summation(exp(diff), axes=axes).reshape(shape).broadcast_to(Z.shape)
        return broadcast_to(reshape(out_grad, shape), Z.shape) * gradient
        # END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)
