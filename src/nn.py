from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import numpy as np

from .autograd import Parameter, Tensor
from .utils import parse_hidden_dims


class Module:
    def children(self) -> list["Module"]:
        children: list[Module] = []
        for value in self.__dict__.values():
            if isinstance(value, Module):
                children.append(value)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        children.append(item)
        return children

    def train(self) -> "Module":
        self._training = True
        for child in self.children():
            child.train()
        return self

    def eval(self) -> "Module":
        self._training = False
        for child in self.children():
            child.eval()
        return self

    def is_training(self) -> bool:
        return bool(getattr(self, "_training", True))

    def parameters(self) -> list[Parameter]:
        return [param for _, param in self.named_parameters()]

    def named_parameters(self, prefix: str = "") -> list[tuple[str, Parameter]]:
        params: list[tuple[str, Parameter]] = []
        for name, value in self.__dict__.items():
            if name.startswith("_"):
                continue
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(value, Tensor) and value.requires_grad:
                params.append((full_name, value))
            elif isinstance(value, Module):
                params.extend(value.named_parameters(full_name))
            elif isinstance(value, (list, tuple)):
                for idx, item in enumerate(value):
                    list_name = f"{full_name}.{idx}"
                    if isinstance(item, Tensor) and item.requires_grad:
                        params.append((list_name, item))
                    elif isinstance(item, Module):
                        params.extend(item.named_parameters(list_name))
        return params

    def state_dict(self) -> dict[str, np.ndarray]:
        return {name: param.data.copy() for name, param in self.named_parameters()}

    def load_state_dict(self, state_dict: dict[str, np.ndarray]) -> None:
        current_params = dict(self.named_parameters())
        missing = set(current_params.keys()) - set(state_dict.keys())
        unexpected = set(state_dict.keys()) - set(current_params.keys())
        if missing:
            raise KeyError(f"缺少参数键: {sorted(missing)}")
        if unexpected:
            raise KeyError(f"存在未使用的参数键: {sorted(unexpected)}")
        for name, param in current_params.items():
            value = np.asarray(state_dict[name], dtype=np.float32)
            if param.data.shape != value.shape:
                raise ValueError(f"参数 {name} 形状不匹配: 期望 {param.data.shape}, 实际 {value.shape}")
            param.data = value.copy()
            if param.requires_grad:
                param.zero_grad()

    def zero_grad(self) -> None:
        for param in self.parameters():
            param.zero_grad()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, activation: str = "relu") -> None:
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        activation = activation.lower()

        if activation == "relu":
            std = np.sqrt(2.0 / self.in_features)
            weight = np.random.randn(self.in_features, self.out_features).astype(np.float32) * std
        else:
            limit = np.sqrt(6.0 / (self.in_features + self.out_features))
            weight = np.random.uniform(-limit, limit, size=(self.in_features, self.out_features)).astype(np.float32)

        bias = np.zeros(self.out_features, dtype=np.float32)
        self.weight = Parameter(weight, requires_grad=True, name="weight")
        self.bias = Parameter(bias, requires_grad=True, name="bias")

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight + self.bias

    def forward_numpy(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weight.data + self.bias.data



def activation_forward_tensor(x: Tensor, activation: str) -> Tensor:
    activation = activation.lower()
    if activation == "relu":
        return x.relu()
    if activation == "sigmoid":
        return x.sigmoid()
    if activation == "tanh":
        return x.tanh()
    raise ValueError(f"不支持的激活函数: {activation}")



def activation_forward_numpy(x: np.ndarray, activation: str) -> np.ndarray:
    activation = activation.lower()
    if activation == "relu":
        return np.maximum(x, 0.0)
    if activation == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    if activation == "tanh":
        return np.tanh(x)
    raise ValueError(f"不支持的激活函数: {activation}")


def dropout_forward_tensor(x: Tensor, dropout: float, training: bool) -> Tensor:
    if dropout <= 0.0 or not training:
        return x
    if dropout >= 1.0:
        raise ValueError("dropout 必须在 [0, 1) 范围内。")
    keep_prob = 1.0 - dropout
    mask = (np.random.rand(*x.shape) < keep_prob).astype(np.float32) / keep_prob
    return x * Tensor(mask, requires_grad=False)


class MLPClassifier(Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: int | Sequence[int],
        num_classes: int,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        hidden_dims = parse_hidden_dims(hidden_dims)
        self.input_dim = int(input_dim)
        self.hidden_dims = list(hidden_dims)
        self.num_classes = int(num_classes)
        self.activation = activation.lower()
        self.dropout = float(dropout)

        dims = [self.input_dim] + self.hidden_dims + [self.num_classes]
        self.layers: list[Linear] = []
        for i in range(len(dims) - 1):
            layer_activation = self.activation if i < len(dims) - 2 else "linear"
            self.layers.append(Linear(dims[i], dims[i + 1], activation=layer_activation))

    def forward(self, x: Tensor) -> Tensor:
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx < len(self.layers) - 1:
                x = activation_forward_tensor(x, self.activation)
                x = dropout_forward_tensor(x, dropout=self.dropout, training=self.is_training())
        return x

    def forward_numpy(self, x: np.ndarray) -> np.ndarray:
        out = x.astype(np.float32, copy=False)
        for idx, layer in enumerate(self.layers):
            out = layer.forward_numpy(out)
            if idx < len(self.layers) - 1:
                out = activation_forward_numpy(out, self.activation)
        return out

    def first_layer_weight(self) -> np.ndarray:
        return self.layers[0].weight.data.copy()

    def config(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "hidden_dims": list(self.hidden_dims),
            "num_classes": self.num_classes,
            "activation": self.activation,
            "dropout": self.dropout,
        }
