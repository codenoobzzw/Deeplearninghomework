from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence, Tuple

import numpy as np


ArrayLike = np.ndarray | float | int


def _ensure_array(data: ArrayLike) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return data.astype(np.float32, copy=False)
    return np.array(data, dtype=np.float32)



def _sum_to_shape(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Reverse numpy broadcasting so grad matches the original tensor shape."""
    if grad.shape == shape:
        return grad

    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    for axis, dim in enumerate(shape):
        if dim == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad.reshape(shape)


class Tensor:
    def __init__(
        self,
        data: ArrayLike,
        requires_grad: bool = False,
        _children: Sequence["Tensor"] = (),
        op: str = "",
        name: str | None = None,
    ) -> None:
        self.data = _ensure_array(data)
        self.requires_grad = requires_grad
        self.grad: np.ndarray | None = np.zeros_like(self.data, dtype=np.float32) if requires_grad else None
        self._backward: Callable[[], None] = lambda: None
        self._prev = tuple(_children)
        self.op = op
        self.name = name

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def zero_grad(self) -> None:
        if self.requires_grad:
            self.grad = np.zeros_like(self.data, dtype=np.float32)

    def detach(self) -> np.ndarray:
        return self.data.copy()

    def numpy(self) -> np.ndarray:
        return self.data

    def item(self) -> float:
        return float(self.data.reshape(-1)[0])

    def __repr__(self) -> str:
        return f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad}, op={self.op!r})"

    def backward(self, grad: ArrayLike | None = None) -> None:
        if not self.requires_grad:
            return

        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("非标量 Tensor 调用 backward 时必须显式提供 grad。")
            grad_arr = np.ones_like(self.data, dtype=np.float32)
        else:
            grad_arr = _ensure_array(grad)

        topo: list[Tensor] = []
        visited: set[int] = set()

        def build(node: Tensor) -> None:
            node_id = id(node)
            if node_id in visited:
                return
            visited.add(node_id)
            for child in node._prev:
                build(child)
            topo.append(node)

        build(self)

        self.grad = grad_arr.astype(np.float32, copy=False)
        for node in reversed(topo):
            node._backward()

    @staticmethod
    def _to_tensor(other: ArrayLike | "Tensor") -> "Tensor":
        if isinstance(other, Tensor):
            return other
        return Tensor(other, requires_grad=False)

    def __add__(self, other: ArrayLike | "Tensor") -> "Tensor":
        other = self._to_tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            op="add",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                assert self.grad is not None
                self.grad += _sum_to_shape(out.grad, self.shape)
            if other.requires_grad:
                assert other.grad is not None
                other.grad += _sum_to_shape(out.grad, other.shape)

        out._backward = _backward
        return out

    def __radd__(self, other: ArrayLike | "Tensor") -> "Tensor":
        return self.__add__(other)

    def __sub__(self, other: ArrayLike | "Tensor") -> "Tensor":
        other = self._to_tensor(other)
        out = Tensor(
            self.data - other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            op="sub",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                assert self.grad is not None
                self.grad += _sum_to_shape(out.grad, self.shape)
            if other.requires_grad:
                assert other.grad is not None
                other.grad -= _sum_to_shape(out.grad, other.shape)

        out._backward = _backward
        return out

    def __rsub__(self, other: ArrayLike | "Tensor") -> "Tensor":
        other = self._to_tensor(other)
        return other.__sub__(self)

    def __neg__(self) -> "Tensor":
        out = Tensor(-self.data, requires_grad=self.requires_grad, _children=(self,), op="neg")

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                assert self.grad is not None
                self.grad -= out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: ArrayLike | "Tensor") -> "Tensor":
        other = self._to_tensor(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            op="mul",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                assert self.grad is not None
                self.grad += _sum_to_shape(out.grad * other.data, self.shape)
            if other.requires_grad:
                assert other.grad is not None
                other.grad += _sum_to_shape(out.grad * self.data, other.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other: ArrayLike | "Tensor") -> "Tensor":
        return self.__mul__(other)

    def __truediv__(self, other: ArrayLike | "Tensor") -> "Tensor":
        other = self._to_tensor(other)
        out = Tensor(
            self.data / other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            op="div",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                assert self.grad is not None
                self.grad += _sum_to_shape(out.grad / other.data, self.shape)
            if other.requires_grad:
                assert other.grad is not None
                other.grad += _sum_to_shape(-out.grad * self.data / (other.data ** 2), other.shape)

        out._backward = _backward
        return out

    def __rtruediv__(self, other: ArrayLike | "Tensor") -> "Tensor":
        other = self._to_tensor(other)
        return other.__truediv__(self)

    def __pow__(self, power: float | int) -> "Tensor":
        if not isinstance(power, (int, float)):
            raise TypeError("只支持标量幂次。")
        out = Tensor(self.data ** power, requires_grad=self.requires_grad, _children=(self,), op=f"pow({power})")

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                assert self.grad is not None
                self.grad += out.grad * (power * (self.data ** (power - 1)))

        out._backward = _backward
        return out

    def matmul(self, other: ArrayLike | "Tensor") -> "Tensor":
        other = self._to_tensor(other)
        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            op="matmul",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                assert self.grad is not None
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                assert other.grad is not None
                other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other: ArrayLike | "Tensor") -> "Tensor":
        return self.matmul(other)

    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> "Tensor":
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _children=(self,),
            op="sum",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if not self.requires_grad:
                return
            assert self.grad is not None
            grad = out.grad
            if axis is None:
                grad = np.broadcast_to(grad, self.shape)
            else:
                axes = axis if isinstance(axis, tuple) else (axis,)
                normalized_axes = tuple(ax if ax >= 0 else ax + self.ndim for ax in axes)
                if not keepdims:
                    for ax in sorted(normalized_axes):
                        grad = np.expand_dims(grad, axis=ax)
                grad = np.broadcast_to(grad, self.shape)
            self.grad += grad

        out._backward = _backward
        return out

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> "Tensor":
        if axis is None:
            denom = float(self.data.size)
        else:
            axes = axis if isinstance(axis, tuple) else (axis,)
            denom = float(np.prod([self.data.shape[ax] for ax in axes]))
        return self.sum(axis=axis, keepdims=keepdims) / denom

    def reshape(self, *shape: int) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = Tensor(self.data.reshape(*shape), requires_grad=self.requires_grad, _children=(self,), op="reshape")

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                assert self.grad is not None
                self.grad += out.grad.reshape(self.shape)

        out._backward = _backward
        return out

    @property
    def T(self) -> "Tensor":
        out = Tensor(self.data.T, requires_grad=self.requires_grad, _children=(self,), op="transpose")

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                assert self.grad is not None
                self.grad += out.grad.T

        out._backward = _backward
        return out

    def relu(self) -> "Tensor":
        out = Tensor(np.maximum(self.data, 0.0), requires_grad=self.requires_grad, _children=(self,), op="relu")

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                assert self.grad is not None
                self.grad += out.grad * (self.data > 0.0)

        out._backward = _backward
        return out

    def sigmoid(self) -> "Tensor":
        sig = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(sig, requires_grad=self.requires_grad, _children=(self,), op="sigmoid")

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                assert self.grad is not None
                self.grad += out.grad * sig * (1.0 - sig)

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        tanh_val = np.tanh(self.data)
        out = Tensor(tanh_val, requires_grad=self.requires_grad, _children=(self,), op="tanh")

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                assert self.grad is not None
                self.grad += out.grad * (1.0 - tanh_val**2)

        out._backward = _backward
        return out

    def exp(self) -> "Tensor":
        exp_val = np.exp(self.data)
        out = Tensor(exp_val, requires_grad=self.requires_grad, _children=(self,), op="exp")

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                assert self.grad is not None
                self.grad += out.grad * exp_val

        out._backward = _backward
        return out

    def log(self) -> "Tensor":
        out = Tensor(np.log(self.data), requires_grad=self.requires_grad, _children=(self,), op="log")

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                assert self.grad is not None
                self.grad += out.grad / self.data

        out._backward = _backward
        return out


Parameter = Tensor


def tensor(data: ArrayLike, requires_grad: bool = False, name: str | None = None) -> Tensor:
    return Tensor(data=data, requires_grad=requires_grad, name=name)



def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    if logits.data.ndim != 2:
        raise ValueError("cross_entropy 期望 logits 形状为 [batch, num_classes]。")
    if logits.data.shape[0] != targets.shape[0]:
        raise ValueError("logits 的 batch 维度与 targets 长度不一致。")

    shifted = logits.data - logits.data.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    eps = 1e-12
    n = targets.shape[0]
    loss_value = -np.log(probs[np.arange(n), targets] + eps).mean()

    out = Tensor(loss_value, requires_grad=logits.requires_grad, _children=(logits,), op="cross_entropy")

    def _backward() -> None:
        if out.grad is None:
            return
        if not logits.requires_grad:
            return
        assert logits.grad is not None
        grad = probs.copy()
        grad[np.arange(n), targets] -= 1.0
        grad /= float(n)
        logits.grad += grad * out.grad

    out._backward = _backward
    return out



def l2_regularization(parameters: Iterable[Tensor], coefficient: float, include_bias: bool = False) -> Tensor:
    params = list(parameters)
    if coefficient <= 0.0 or not params:
        return Tensor(0.0, requires_grad=False)

    reg: Tensor | None = None
    for param in params:
        if not include_bias and param.ndim < 2:
            continue
        term = (param * param).sum()
        reg = term if reg is None else reg + term

    if reg is None:
        return Tensor(0.0, requires_grad=False)
    return reg * (0.5 * float(coefficient))
