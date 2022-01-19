import inspect
import math
from typing import List

import numpy as np

import metagrad.functions as F
from metagrad.paramater import Parameter
from metagrad.tensor import Tensor


class Module:
    '''
    所有模型的基类
    '''

    def parameters(self) -> List[Parameter]:
        parameters = []
        for name, value in inspect.getmembers(self):
            if isinstance(value, Parameter):
                parameters.append(value)
            elif isinstance(value, Module):
                parameters.extend(value.parameters())

        return parameters

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError


class Linear(Module):
    r"""
         对给定的输入进行线性变换: :math:`y=xA^T + b`

        Args:
            in_features: 每个输入样本的大小
            out_features: 每个输出样本的大小
            bias: 是否含有偏置，默认 ``True``
        Shape:
            - Input: `(*, H_in)` 其中 `*` 表示任意维度，包括none,这里 `H_{in} = in_features`
            - Output: :math:`(*, H_out)` 除了最后一个维度外，所有维度的形状都与输入相同，这里H_out = out_features`
        Attributes:
            weight: 可学习的权重，形状为 `(out_features, in_features)`.
            bias:   可学习的偏置，形状 `(out_features)`.
        """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(Tensor.empty((out_features, in_features)))
        if bias:
            self.bias = Parameter(Tensor.zeros(out_features))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.weight.assign(np.random.randn(self.out_features, self.in_features) / math.sqrt(self.in_features))

    def forward(self, input: Tensor) -> Tensor:
        x = input @ self.weight.T
        if self.bias is not None:
            x = x + self.bias

        return x


class Sequential(Module):
    def __init__(self, *layers):
        self._layers = layers

    @property
    def layers(self):
        return self._layers

    def parameters(self) -> List[Parameter]:
        parameters = []
        for layer in self._layers:
            parameters.extend(layer.parameters())

        return parameters

    def forward(self, input: Tensor) -> Tensor:
        for layer in self._layers:
            x = layer(x)

        return x


class Flatten(Module):
    def forward(self, input: Tensor) -> Tensor:
        # 保留批次大小
        return input.reshape(input.shape[0], -1)


# ****激活函数作为Module实现****
class ReLU(Module):
    def forward(self, input: Tensor) -> Tensor:
        return F.relu(input)
