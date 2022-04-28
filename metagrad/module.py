import inspect
from typing import List, Optional

import metagrad.functions as F
from metagrad import init
from metagrad.paramater import Parameter
from metagrad.tensor import Tensor


class Module:
    '''
    所有模型的基类
    '''

    training: bool

    def __init__(self) -> None:
        self.training = True

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

    def train(self, mode: bool = True):
        self.training = mode
        return self

    def eval(self):
        """
        只会影响某些模型，比如Dropout和BatchNorm等
        :return:
        """
        return self.train(False)


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
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(Tensor.empty((out_features, in_features)))
        if bias:
            self.bias = Parameter(Tensor.zeros(out_features))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_normal_(self.weight)  # 默认采用kaiming初始化

    def forward(self, input: Tensor) -> Tensor:
        x = input @ self.weight.T
        if self.bias is not None:
            x = x + self.bias

        return x


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, _weight: Optional[Tensor] = None) -> None:
        '''
        一个存储固定大小词汇表嵌入的查找表，可以通过索引(列表)直接访问，而不是one-hot向量。
        :param num_embeddings: 词汇表大小
        :param embedding_dim:  嵌入维度
        '''

        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # 也可以传预训练好的权重进来
        if _weight is None:
            self.weight = Parameter(Tensor.empty((num_embeddings, embedding_dim)))
            self.reset_parameters()
        else:
            assert list(_weight.shape) == [num_embeddings, embedding_dim], \
                'Shape of weight does not match num_embeddings and embedding_dim'
            self.weight = Parameter(_weight)

    def reset_parameters(self) -> None:
        init.normal_(self.weight)

    def forward(self, input: Tensor) -> Tensor:
        return F.embedding(input, self.weight)

class Sequential(Module):
    def __init__(self, *layers):
        super(Sequential, self).__init__()
        self._layers = layers

    @property
    def layers(self):
        return self._layers

    def parameters(self) -> List[Parameter]:
        parameters = []
        for layer in self._layers:
            parameters.extend(layer.parameters())

        return parameters

    def forward(self, x: Tensor) -> Tensor:
        for layer in self._layers:
            x = layer(x)

        return x

    def train(self, mode: bool = True):
        for layer in self._layers:
            layer.train(mode)

    def eval(self):
        for layer in self._layers:
            layer.train(False)


class Flatten(Module):
    def forward(self, input: Tensor) -> Tensor:
        # 保留批次大小
        return input.reshape(input.shape[0], -1)


# ****激活函数作为Module实现****
class ReLU(Module):
    def forward(self, input: Tensor) -> Tensor:
        return F.relu(input)


# Dropout
class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        """
        :param p: 丢弃率
        """

        super(Dropout, self).__init__()

        self.p = p

    def forward(self, input: Tensor) -> Tensor:
        return F.dropout(input, self.p, self.training)
