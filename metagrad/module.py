import inspect
import math
import pickle
from typing import List, Optional, Tuple

import metagrad.functions as F
from metagrad import init
from metagrad.paramater import Parameter
from metagrad.tensor import Tensor, no_grad


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

    def save(self, path='model.pt'):
        self.to_cpu()
        with open(path, 'wb') as f:
            print(f'Saving {self} to {path}')
            pickle.dump(self, f)

    def load(self, path='model.pt'):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def _apply(self, fn):
        with no_grad():
            for name, value in inspect.getmembers(self):
                if isinstance(value, Parameter):
                    fn(value)
                elif isinstance(value, Module):
                    [fn(p) for p in value.parameters()]

        return self

    def to_gpu(self, device):
        return self._apply(lambda t: t.to_gpu(device))

    def to_cpu(self):
        return self._apply(lambda t: t.to_cpu())

    def to(self, device):
        return self._apply(lambda t: t.to(device))

    def __repr__(self):
        return self.__class__.__name__


class Linear(Module):
    r"""
         对给定的输入进行线性变换: :math:`y=xA^T + b`

        Args:
            in_features: 每个输入样本的大小
            out_features: 每个输出样本的大小
            bias: 是否含有偏置，默认 ``True``
            device: CpuDevice或GpuDevice
            dtype: np.dtype
        Shape:
            - Input: `(*, H_in)` 其中 `*` 表示任意维度，包括none,这里 `H_{in} = in_features`
            - Output: :math:`(*, H_out)` 除了最后一个维度外，所有维度的形状都与输入相同，这里H_out = out_features`
        Attributes:
            weight: 可学习的权重，形状为 `(out_features, in_features)`.
            bias:   可学习的偏置，形状 `(out_features)`.
        """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        factory_kwargs = {'device': device, 'dtype': dtype}

        self.weight = Parameter(Tensor.empty((out_features, in_features)), **factory_kwargs)
        if bias:
            self.bias = Parameter(Tensor.zeros(out_features), **factory_kwargs)
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
    def __init__(self, num_embeddings: int, embedding_dim: int, _weight: Optional[Tensor] = None,
                 dtype=None, device=None) -> None:
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
            self.weight = Parameter(Tensor.empty((num_embeddings, embedding_dim), dtype=dtype, device=device))
            self.reset_parameters()
        else:
            assert list(_weight.shape) == [num_embeddings, embedding_dim], \
                'Shape of weight does not match num_embeddings and embedding_dim'
            self.weight = Parameter(_weight, device=device)

    def reset_parameters(self) -> None:
        init.uniform_(self.weight)

    def forward(self, input: Tensor) -> Tensor:
        return F.embedding(self.weight, input)

    @classmethod
    def from_pretrained(cls, embeddings: Tensor, freeze=True):
        assert embeddings.ndim == 2, \
            'Embeddings parameter is expected to be 2-dimensional'
        rows, cols = embeddings.shape
        embedding = cls(num_embeddings=rows, embedding_dim=cols, _weight=embeddings)
        embedding.weight.requires_grad = not freeze
        return embedding


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


def apply_permutation(tensor: Tensor, permutation: Tensor, dim: int = 1) -> Tensor:
    return tensor.index_select(dim, permutation)


# RNN
class RNNBase(Module):
    def __init__(self, mode: str, input_size: int, hidden_size: int,
                 num_layers: int = 1, bias: bool = True, batch_first: bool = False,
                 dropout: float = 0., bidirectional: bool = False, proj_size: int = 0,
                 device=None, dtype=None
                 ) -> None:
        '''
        RNN基类参数

        Args:
            mode: LSTM|GRU|RNN_TANH|RNN_RELU
            input_size: 输入x的特征数
            hidden_size: 隐藏状态h的特征数
            num_layers: 层数，默认为1。若大于1，表示堆叠多个RNN
            bias: 是否需要偏置项，如果为False，那么不使用b_ih和b_hh
            batch_first: 如果为True，那么输入和输出通过(batch,seq,feature)的形式，而不是(seq,batch,feature)
            dropout: 多层RNN间使用dropout的比率，默认为0，代表不使用dropout
            bidirectional: 如果为True，代表双向RNN
            proj_size: LSTM会投影到相应的大小，默认为0
            device: 设备GpuDevice或CpuDevice
            dtype: 数据类型
        '''

        super(RNNBase, self).__init__()
        factory_kwards = {'device': device, 'dtype': dtype}

        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = float(dropout)
        self.bidirectional = bidirectional
        self.proj_size = proj_size

        num_directions = 2 if bidirectional else 1

        # 支持LSTM、GRU和简单RNN
        # gate_size为所有门控中权重的维度
        if mode == 'LSTM':
            gate_size = 4 * hidden_size
        elif mode == 'GRU':
            gate_size = 3 * hidden_size
        elif mode == 'RNN_TANH':
            gate_size = hidden_size
        elif mode == 'RNN_RELU':
            gate_size = hidden_size
        else:
            raise ValueError("Unrecognized RNN mode: " + mode)

        self._flat_weights_names = []
        self._all_weights = []

        for layer in range(num_layers):
            for direction in range(num_directions):
                # 真正的隐藏层大小
                real_hidden_size = proj_size if proj_size > 0 else hidden_size
                # 输入大小
                layer_input_size = input_size if layer == 0 else real_hidden_size * num_directions
                # 输入(input)到隐藏层(hidden)的权重
                w_ih = Parameter(Tensor.empty((gate_size, layer_input_size), **factory_kwards))
                # 输入到隐藏层的偏置
                b_ih = Parameter(Tensor.empty(gate_size, **factory_kwards))
                # 隐藏层到隐藏层的权重
                w_hh = Parameter(Tensor.empty((gate_size, real_hidden_size), **factory_kwards))
                # 隐藏层到隐藏层的偏置
                b_hh = Parameter(Tensor.empty(gate_size, **factory_kwards))

                layer_params: Tuple[Tensor, ...] = ()
                if self.proj_size == 0:
                    if bias:
                        layer_params = (w_ih, w_hh, b_ih, b_hh)
                    else:
                        layer_params = (w_ih, w_hh)
                else:
                    w_hr = Parameter(Tensor.empty((proj_size, hidden_size), **factory_kwards))
                    if bias:
                        layer_params = (w_ih, w_hh, b_ih, b_hh, w_hr)
                    else:
                        layer_params = (w_ih, w_hh, w_hr)

                suffix = '_reverse' if direction == 1 else ''
                param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
                if bias:
                    param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
                if self.proj_size > 0:
                    param_names += ['weight_hr_l{}{}']
                param_names = [x.format(layer, suffix) for x in param_names]
                # 动态设值
                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)

                self._flat_weights_names.extend(param_names)
                self._all_weights.append(param_names)

        self._flat_weights = [(lambda wn: getattr(self, wn) if hasattr(self, wn) else None)(wn) for wn in
                              self._flat_weights_names]

        self.reset_parameters()

    def reset_parameters(self):
        '''
        通过 U(-sqrt(hidden_size), sqrt(hidden_size)来初始化所有的权重和偏置
        '''

        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)


class RNN(RNNBase):
    '''
    Elman RNN
    '''

    def __init__(self, *args, **kwargs):
        # 得到并删除nonlinearity对应的值，默认为tanh
        self.nonlinearity = kwargs.pop('nonlinearity', 'tanh')
        # 激活函数支持 tanh 和 relu
        if self.nonlinearity == 'tanh':
            mode = 'RNN_TANH'
        else:
            mode = 'RNN_RELU'

        super(RNN, self).__init__(mode, *args, **kwargs)

    def forward(self, input, hx=None):
        '''
        RNN中的前向传播
        Args:
            input: 输入，形状为(batch,seq,feature)或(seq,batch,feature)或(seq, feature)
            hx: 相应的隐藏状态

        Returns:

        '''
        # 如果输入维度数为3，说明是(batch,seq,feature)或(seq,batch,feature)这种批输入
        is_batched = input.dim() == 3
        batch_dim = 0 if self.batch_first else 1  # 获取batch大小所在的维度
        # 如果不是批输入
        if not is_batched:
            # 在batch_dim处增加维数1，变成大小为1的批输入
            input = input.unsqueeze(batch_dim)
            # 如果hx不为空，我们还需要增加hx的批次维度
            if hx is not None:
                if hx.dim() != 2:
                    raise RuntimeError(
                        f"For unbatched 2-D input, hx should also be 2-D but got {hx.dim()}-D tensor")
                # 隐藏状态的批次维度在1处
                hx = hx.unsqueeze(1)
        else:
            # 验证hx的维度
            if hx is not None and hx.dim() != 3:
                raise RuntimeError(
                    f"For batched 3-D input, hx should also be 3-D but got {hx.dim()}-D tensor")

        # 批次大小
        max_batch_size = input.size(0) if self.batch_first else input.size(1)

        if hx is None:
            # 初始时hx为None
            num_directions = 2 if self.bidirectional else 1
            hx = Tensor.zeros(self.num_layers * num_directions,
                              max_batch_size, self.hidden_size,
                              dtype=input.dtype, device=input.device)

        assert hx is not None

        assert self.mode == 'RNN_TANH' or self.mode == 'RNN_RELU'

        if self.mode == 'RNN_TANH':
            result = F.rnn_tanh(input, hx, self._flat_weights, self.bias, self.num_layers,
                                self.dropout, self.training, self.bidirectional,
                                self.batch_first)
        else:
            result = F.rnn_relu(input, hx, self._flat_weights, self.bias, self.num_layers,
                                self.dropout, self.training, self.bidirectional,
                                self.batch_first)

        output = result[0]
        hidden = result[1]

        if not is_batched:
            output = output.squeeze(batch_dim)
            hidden = hidden.squeeze(1)

        return output, hidden
