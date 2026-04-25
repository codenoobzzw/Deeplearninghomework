from __future__ import annotations

from typing import Iterable

import numpy as np

from .autograd import Tensor


class SGD:
    def __init__(self, parameters: Iterable[Tensor], lr: float = 1e-2, momentum: float = 0.9) -> None:
        self.parameters = list(parameters)
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.velocities = [np.zeros_like(param.data, dtype=np.float32) for param in self.parameters]

    def step(self) -> None:
        for idx, param in enumerate(self.parameters):
            if not param.requires_grad or param.grad is None:
                continue
            self.velocities[idx] = self.momentum * self.velocities[idx] + param.grad
            param.data = param.data - self.lr * self.velocities[idx]

    def zero_grad(self) -> None:
        for param in self.parameters:
            param.zero_grad()

    def set_lr(self, lr: float) -> None:
        self.lr = float(lr)

    def get_lr(self) -> float:
        return float(self.lr)


class ExponentialLRScheduler:
    def __init__(self, optimizer: SGD, gamma: float = 1.0) -> None:
        self.optimizer = optimizer
        self.gamma = float(gamma)

    def step(self) -> float:
        self.optimizer.set_lr(self.optimizer.get_lr() * self.gamma)
        return self.optimizer.get_lr()
