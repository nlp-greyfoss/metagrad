import copy
import math
import operator
import pickle
from collections import OrderedDict
from itertools import chain, islice
from typing import List, Optional, Tuple, Dict, Iterable, Union, Iterator, Set

import metagrad.functions as F
from metagrad import init
from metagrad.paramater import Parameter
from metagrad.tensor import Tensor, no_grad


def _addindent(s_, numSpaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s


class Module:
    '''
    所有模型的基类
    '''

    training: bool

    def __init__(self) -> None:
        self.training = True
        self._parameters: Dict[str, Optional[Parameter]] = OrderedDict()
        self._modules: Dict[str, Optional['Module']] = OrderedDict()

    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        self._parameters[name] = param

    def add_module(self, name: str, module: Optional['Module']) -> None:
        self._modules[name] = module

    def get_submodule(self, target: str) -> 'Module':
        if target == "":
            return self

        atoms: List[str] = target.split(".")
        mod: Module = self

        for item in atoms:
            mod = getattr(mod, item)

        return mod

    def get_parameter(self, target: str) -> Parameter:
        # 从最后一个.分隔
        module_path, _, param_name = target.rpartition(".")

        mod: Module = self.get_submodule(module_path)

        param: Parameter = getattr(mod, param_name)

        return param

    def _named_members(self, get_members_fn, prefix='', recurse=True):
        memo = set()
        modules = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                if v is None or v in memo:
                    continue
                memo.add(v)
                name = module_prefix + ('.' if module_prefix else '') + k
                yield name, v

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for name, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Parameter]]:
        '''

        Args:
            prefix:
            recurse: True，返回该module和所有submodule的参数；否则，仅返回该module的参数

        Yields:
            (string, Parameter): 包含名称和参数的元组

        '''
        gen = self._named_members(lambda module: module._parameters.items(), prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def children(self) -> Iterator['Module']:
        for name, module in self.named_children():
            yield module

    def named_children(self) -> Iterator[Tuple[str, 'Module']]:
        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module

    def modules(self) -> Iterator['Module']:
        for _, module in self.named_modules():
            yield module

    def named_modules(self, memo: Optional[Set['Module']] = None, prefix: str = '',
                      remove_duplicate: bool = True):
        if memo is None:
            memo = set()

        if self not in memo:
            if remove_duplicate:
                memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ('.' if prefix else '') + name
                for m in module.named_modules(memo, submodule_prefix, remove_duplicate):
                    yield m

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def train(self, mode: bool = True):
        self.training = mode
        for module in self.children():
            module.train(mode)

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
        for module in self.children():
            module._apply(fn)

        for key, param in self._parameters.items():
            if param is None:
                continue

            with no_grad():
                param_applied = fn(param)

            out_param = Parameter(param_applied)
            self._parameters[key] = out_param

        return self

    def to_gpu(self, device):
        return self._apply(lambda t: t.to_gpu(device))

    def to_cpu(self):
        return self._apply(lambda t: t.to_cpu())

    def to(self, device):
        return self._apply(lambda t: t.to(device))

    def __setattr__(self, name: str, value: Union[Tensor, 'Module']) -> None:
        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.discard(name)

        params = self.__dict__.get('_parameters')
        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError(
                    "cannot assign parameters before Module.__init__() call")
            remove_from(self.__dict__, self._modules)
            self.register_parameter(name, value)
        elif params is not None and name in params:
            if value is not None:
                raise TypeError(f"cannot assign '{value}' as parameter '{name}' "
                                "(torch.nn.Parameter or None expected)")
            self.register_parameter(name, value)
        else:
            modules = self.__dict__.get('_modules')
            if isinstance(value, Module):
                if modules is None:
                    raise AttributeError(
                        "cannot assign module before Module.__init__() call")
                remove_from(self.__dict__, self._parameters)
                modules[name] = value
            elif modules is not None and name in modules:
                if value is not None:
                    raise TypeError(f"cannot assign '{value}' as child module '{name}' "
                                    "(torch.nn.Module or None expected)")
                modules[name] = value
            else:
                object.__setattr__(self, name, value)

    def __getattr__(self, name: str) -> Union[Tensor, 'Module']:
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    def __delattr__(self, name):
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._modules:
            del self._modules[name]
        else:
            object.__delattr__(self, name)

    def _get_name(self):
        return self.__class__.__name__

    def extra_repr(self) -> str:
        return ''

    def __repr__(self):
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str


class Identity(Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return input


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
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Linear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(Tensor.empty((out_features, in_features)), **factory_kwargs)
        if bias:
            self.bias = Parameter(Tensor.zeros(out_features), **factory_kwargs)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_normal_(self.weight)  # 默认采用kaiming初始化

    def forward(self, input: Tensor) -> Tensor:
        x = input @ self.weight.T
        if self.bias is not None:
            x = x + self.bias

        return x

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, _weight: Optional[Tensor] = None,
                 dtype=None, device=None, padding_idx: Optional[int] = None) -> None:
        '''
        一个存储固定大小词汇表嵌入的查找表，可以通过索引(列表)直接访问，而不是one-hot向量。
        :param num_embeddings: 词汇表大小
        :param embedding_dim:  嵌入维度
        '''

        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

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
        self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            with no_grad():
                self.weight[self.padding_idx] = 0

    def forward(self, input: Tensor) -> Tensor:
        return F.embedding(self.weight, input)

    @classmethod
    def from_pretrained(cls, embeddings: Tensor, freeze=True, padding_idx=None):
        assert embeddings.ndim == 2, \
            'Embeddings parameter is expected to be 2-dimensional'
        rows, cols = embeddings.shape
        embedding = cls(num_embeddings=rows, embedding_dim=cols, _weight=embeddings, padding_idx=padding_idx)
        embedding.weight.requires_grad = not freeze
        return embedding


class Sequential(Module):
    def __init__(self, *args):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx):
        size = len(self)
        idx = operator.index(idx)
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx: int, module: Module) -> None:
        key: str = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx: Union[slice, int]) -> None:
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def forward(self, input) -> Tensor:
        for module in self:
            input = module(input)
        return input

    def append(self, module: Module) -> 'Sequential':
        self.add_module(str(len(self)), module)
        return self


class ModuleList(Module):
    _modules: Dict[str, Module]

    def __init__(self, modules: Optional[Iterable[Module]] = None) -> None:
        super(ModuleList, self).__init__()

        if modules is not None:
            self += modules

    def _get_abs_string_index(self, idx):
        '''返回绝对值索引'''
        idx = operator.index(idx)  # 转换为int
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        return str(idx)

    def __getitem__(self, idx: int) -> Union[Module, 'ModuleList']:
        if isinstance(idx, slice):
            return self.__class__(list(self._modules.values())[idx])
        return self._modules[self._get_abs_string_index(idx)]

    def __setitem__(self, idx: int, module: Module):
        idx = self._get_abs_string_index(idx)
        return setattr(self, str(idx), module)

    def __delitem__(self, idx: Union[int, slice]) -> None:
        if isinstance(idx, slice):
            for k in range(len(self._modules))[idx]:
                delattr(self, str(k))
        else:
            delattr(self, self._get_abs_string_index(idx))

        # 为了保留编号，在删除 self._modules 之后使用modules重新构建它
        str_indices = [str(i) for i in range(len(self._modules))]
        self._modules = OrderedDict(list(zip(str_indices, self._modules.values())))

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def __iadd__(self, modules: Iterable[Module]) -> 'ModuleList':
        return self.extend(modules)

    def __add__(self, other: Iterable[Module]) -> 'ModuleList':
        combined = ModuleList()
        # chain将self, other变成一个可迭代对象
        for i, module in enumerate(chain(self, other)):
            combined.add_module(str(i), module)
        return combined

    def insert(self, index: int, module: Module) -> None:
        '''
        在给定index之前插入module
        Args:
            index: 要插入的索引
            module: 要插入的module
        '''
        # 数组的插入算法，我们需要维护str(i)
        for i in range(len(self._modules), index, -1):
            self._modules[str(i)] = self._modules[str(i - i)]

        self._modules[str(index)] = module

    def append(self, module: Module) -> 'ModuleList':
        self.add_module(str(len(self)), module)
        return self

    def extend(self, modules: Iterable[Module]) -> 'ModuleList':
        offset = len(self)
        for i, module in enumerate(modules):
            self.add_module(str(offset + i), module)
        return self


# ****激活函数作为Module实现****
class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()

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

    def extra_repr(self) -> str:
        return f'p={self.p}'


class RNNCellBase(Module):
    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def __init__(self, input_size, hidden_size: int, num_chunks: int, bias: bool = True, num_directions=1, device=None,
                 dtype=None) -> None:
        '''
        RNN单时间步的抽象
        :param input_size: 输入x的特征数
        :param hidden_size: 隐藏状态的特征数
        :param bias: 线性层是否包含偏置
        :param nonlinearity: 非线性激活函数 tanh | relu (mode = RNN)
        '''
        factory_kwargs = {'device': device, 'dtype': dtype}

        super(RNNCellBase, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # 输入x的线性变换
        self.input_trans = Linear(num_directions * input_size, num_chunks * hidden_size, bias=bias, **factory_kwargs)
        # 隐藏状态的线性变换
        self.hidden_trans = Linear(hidden_size, num_chunks * hidden_size, bias=bias, **factory_kwargs)

        self.reset_parameters()

    def extra_repr(self) -> str:
        s = 'input_size={input_size}, hidden_size={hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        return s.format(**self.__dict__)


class RNNCell(RNNCellBase):
    def __init__(self, input_size, hidden_size: int, bias: bool = True, nonlinearity: str = 'tanh', num_directions=1,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(RNNCell, self).__init__(input_size, hidden_size, num_chunks=1, bias=bias, num_directions=num_directions,
                                      **factory_kwargs)

        if nonlinearity == 'tanh':
            self.activation = F.tanh
        else:
            self.activation = F.relu

    def forward(self, x: Tensor, h: Tensor, c: Tensor = None) -> Tensor:
        h_next = self.activation(self.input_trans(x) + self.hidden_trans(h))
        return h_next


class LSTMCell(RNNCellBase):
    def __init__(self, input_size, hidden_size: int, bias: bool = True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(LSTMCell, self).__init__(input_size, hidden_size, num_chunks=4, bias=bias, **factory_kwargs)

    def forward(self, x: Tensor, h: Tensor, c: Tensor) -> Tuple[Tensor, Tensor]:
        ifgo = self.input_trans(x) + self.hidden_trans(h)
        ifgo = F.chunk(ifgo, 4, -1)
        # 一次性计算三个门 与 g_t
        i, f, g, o = ifgo

        c_next = F.sigmoid(f) * c + F.sigmoid(i) * F.tanh(g)

        h_next = F.sigmoid(o) * F.tanh(c_next)

        return h_next, c_next


class GRUCell(RNNCellBase):
    def __init__(self, input_size, hidden_size: int, bias: bool = True, num_directions=1, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(GRUCell, self).__init__(input_size, hidden_size, num_chunks=3, bias=bias, num_directions=num_directions,
                                      **factory_kwargs)

    def forward(self, x: Tensor, h: Tensor, c: Tensor = None) -> Tensor:
        input_trans = self.input_trans(x)
        hidden_trans = self.hidden_trans(h)

        i_r, i_z, i_g = F.chunk(input_trans, 3, -1)
        h_r, h_z, h_g = F.chunk(hidden_trans, 3, -1)

        r = F.sigmoid(i_r + h_r)  # 重置门
        z = F.sigmoid(i_z + h_z)  # 更新门

        h_next = z * h + (1 - z) * F.tanh(i_g + r * h_g)  # g = i_g + r * h_g  候选状态 = tanh(g)
        return h_next


class RNNBase(Module):
    def __init__(self, cell: RNNCellBase, input_size: int, hidden_size: int, batch_first: bool = False,
                 num_layers: int = 1, bidirectional: bool = False, bias: bool = True, dropout: float = 0, ) -> None:
        '''
           :param input_size:  输入x的特征数
           :param hidden_size: 隐藏状态的特征数
           :param batch_first: 批次维度是否在前面
           :param num_layers: 层数
           :param bidirectional: 是否为双向
           :param bias: 线性层是否包含偏置
           :param dropout: 用于多层堆叠RNN，默认为0代表不使用dropout
       '''
        super(RNNBase, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.bias = bias

        self.num_directions = 2 if self.bidirectional else 1

        # 支持多层
        self.cells = ModuleList([cell(input_size, hidden_size, bias)] +
                                [cell(hidden_size, hidden_size, bias, num_directions=self.num_directions) for _ in
                                 range(num_layers - 1)])
        if self.bidirectional:
            # 支持双向
            self.back_cells = copy.deepcopy(self.cells)

        self.dropout = dropout
        if dropout != 0:
            # Dropout层
            self.dropout_layer = Dropout(dropout)

    def _one_directional_op(self, input, n_steps, cell, h, c) -> Tuple[Tensor, Tensor, Tensor]:
        hs, cs = [], []
        # 沿着input时间步进行遍历
        for t in range(n_steps):
            inp = input[t]

            h, c = cell(inp, h, c)
            hs.append(h)
            cs.append(c)

        if c is not None:
            c_n = F.stack(cs)
        else:
            c_n = None

        return h, F.stack(hs), c_n

        # def _one_directional_op(self, input, cells, n_steps, hs, cs=None, reverse=False):

    #     '''
    #
    #     Args:
    #         input: 输入 [n_steps, batch_size, input_size]
    #         cells: 正向或反向RNNCell的ModuleList
    #         hs: 隐藏状态
    #         cs: 单元状态
    #         n_steps: 步长
    #         reverse: true 反向
    #
    #     Returns:
    #
    #     '''
    #     output = []
    #
    #     for t in range(n_steps):
    #         inp = input[t]
    #
    #         for layer in range(self.num_layers):
    #             hs[layer] = cells[layer](inp, hs[layer])
    #             inp = hs[layer]
    #             if self.dropout and layer != self.num_layers - 1:
    #                 inp = self.dropout(inp)
    #
    #         # 收集最终层的输出
    #         output.append(hs[-1])
    #
    #     output = F.stack(output)  # (n_steps, batch_size, num_directions * hidden_size)
    #
    #     if reverse:
    #         output = F.flip(output, 0)  # 将输出时间步维度逆序，使得时间步t=0上，是看了整个序列的结果。
    #
    #     if self.batch_first:
    #         output = output.transpose((1, 0, 2))
    #
    #     h_n = F.stack(hs)
    #
    #     return output, h_n

    def _handle_hidden_state(self, input, state):
        assert input.ndim == 3  # 必须传入批数据，最小批大小为1

        if self.batch_first:
            batch_size, n_steps, _ = input.shape
            input = input.transpose((1, 0, 2))  # 将batch放到中间维度
        else:
            n_steps, batch_size, _ = input.shape

        if state is None:
            h = Tensor.zeros((self.num_layers * self.num_directions, batch_size, self.hidden_size), dtype=input.dtype,
                             device=input.device)
        else:
            h = state

        # 得到每层的状态
        hs = list(F.unbind(h))  # 按层数拆分h

        return hs, [None] * len(hs), input, n_steps, batch_size

    def _bidirectional_forward(self, input, n_steps, hs, cs=None):
        output_f, h_n_f = self._one_directional_op(input, self.cells, n_steps, hs[:self.num_layers])

        output_b, h_n_b = self._one_directional_op(F.flip(input, 0), self.back_cells, n_steps, hs[self.num_layers:],
                                                   reverse=True)

        output = F.cat([output_f, output_b], 2)
        h_n = F.cat([h_n_f, h_n_b], 0)

        return output, h_n

    def forward(self, input: Tensor, state: Tensor) -> Tuple[Tensor, Union[Tensor, Tuple[Tensor, Tensor]]]:
        '''
        RNN的前向传播
        :param input: 形状 [n_steps, batch_size, input_size] 若batch_first=False
        :param state: (隐藏状态，单元状态)元组， 每个元素形状 [num_layers, batch_size, hidden_size]
        :return:
            num_directions = 2 if self.bidirectional else 1

            output: (n_steps, batch_size, num_directions * hidden_size)若batch_first=False 或
                    (batch_size, n_steps, num_directions * hidden_size)若batch_first=True
                    包含每个时间步最后一层(多层RNN)的输出h_t
            h_n: (num_directions * num_layers, batch_size, hidden_size) 包含最终隐藏状态
        '''

        hs, cs, input, n_steps, batch_size = self._handle_hidden_state(input, state)

        h_last_f, h_last_b = [], []

        for layer in range(self.num_layers):
            h, hs_f, cs_f = self._one_directional_op(input, n_steps, self.cells[layer], hs[layer], cs[layer])

            h_last_f.append(h)  # 保存最后一个时间步的隐藏状态
            if self.bidirectional:
                h, hs_b, cs_b = self._one_directional_op(F.flip(input, 0), n_steps, self.back_cells[layer],
                                                         hs[layer + self.num_layers], cs[layer + self.num_layers])
                hs_b = F.flip(hs_b, 0)  # 将输出时间步维度逆序，使得时间步t=0上，是看了整个序列的结果。
                # 拼接两个方向上的输入

                h_last_b.append(h)
                input = F.cat([hs_f, hs_b], 2)  #
            else:
                input = hs_f  #

            if self.dropout and layer != self.num_layers - 1:
                input = self.dropout_layer(input)

        if self.bidirectional:
            output = F.cat([hs_f, hs_b], 2)
            h_n = F.cat([F.stack(h_last_f), F.stack(h_last_b)], 0)
        else:
            output = hs_f
            h_n = F.stack(h_last_f)

        if self.batch_first:
            output = output.transpose((1, 0, 2))

        return output, h_n

    def extra_repr(self) -> str:
        s = 'input_size={input_size}, hidden_size={hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout:
            s += ', dropout={dropout}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        return s.format(**self.__dict__)


class RNN(RNNBase):
    def __init__(self, *args, **kwargs) -> None:
        '''
        :param input_size:  输入x的特征数
        :param hidden_size: 隐藏状态的特征数
        :param batch_first:
        :param num_layers: 层数
        :param bidirectional: 是否为双向
        :param bias: 线性层是否包含偏置
        :param dropout: 用于多层堆叠RNN，默认为0代表不使用dropout
        :param nonlinearity: 非线性激活函数 tanh | relu
        '''
        super(RNN, self).__init__(RNNCell, *args, **kwargs)


class GRU(RNNBase):
    def __init__(self, *args, **kwargs):
        '''
        :param input_size:  输入x的特征数
        :param hidden_size: 隐藏状态的特征数
        :param batch_first:
        :param num_layers: 层数
        :param bidirectional: 是否为双向
        :param bias: 线性层是否包含偏置
        :param dropout: 用于多层堆叠RNN，默认为0代表不使用dropout
        '''
        super(GRU, self).__init__(GRUCell, *args, **kwargs)


class LSTM(RNNBase):
    '''
    写很多重复代码，是为了减少if-else判断，增加代码运行效率。
    '''

    def __init__(self, *args, **kwargs):
        super(LSTM, self).__init__(LSTMCell, *args, **kwargs)

    def _handle_hidden_state(self, input, state):
        h_0, c_0 = None, None
        if state is not None:
            h_0, c_0 = state

        is_batched = input.ndim == 3
        batch_dim = 0 if self.batch_first else 1
        if not is_batched:
            # 转换为批大小为1的输入
            input = input.unsqueeze(batch_dim)
            if state is not None:
                h_0 = h_0.unsqueeze(1)
                c_0 = c_0.unsqueeze(1)

        if self.batch_first:
            batch_size, n_steps, _ = input.shape
            input = input.transpose((1, 0, 2))  # 将batch放到中间维度
        else:
            n_steps, batch_size, _ = input.shape

        if state is None:
            num_directions = 2 if self.bidirectional else 1
            h_0 = Tensor.zeros((self.num_layers * num_directions, batch_size, self.hidden_size), dtype=input.dtype,
                               device=input.device)
            c_0 = Tensor.zeros((self.num_layers * num_directions, batch_size, self.hidden_size), dtype=input.dtype,
                               device=input.device)

        # 得到每层的状态
        hs, cs = list(F.split(h_0)), list(F.split(c_0))

        return hs, cs, input, n_steps, batch_size

    def _one_directional_op(self, input, cells, n_steps, hs, cs, reverse=False):
        '''

        Args:
            input: 输入 [n_steps, batch_size, input_size]
            cells: 正向或反向RNNCell的ModuleList
            n_steps: 步长
            hs: 隐藏状态
            cs: 单元状态
            reverse: true 反向

        Returns:

        '''
        output = []
        for t in range(n_steps):
            inp = input[t]

            for layer in range(self.num_layers):
                hs[layer], cs[layer] = cells[layer](inp, (hs[layer], cs[layer]))
                inp = hs[layer]
                if self.dropout and layer != self.num_layers - 1:
                    inp = self.dropout(inp)

            # 收集最终层的输出
            output.append(hs[-1])

        output = F.stack(output)  # (n_steps, batch_size, num_directions * hidden_size)

        if reverse:
            output = F.flip(output, 0)  # 将输出时间步维度逆序，使得时间步t=0上，是看了整个序列的结果。

        if self.batch_first:
            output = output.transpose((1, 0, 2))

        h_n = F.stack(hs)
        c_n = F.stack(cs)

        return output, (h_n, c_n)

    def _bidirectional_forward(self, input, n_steps, hs, cs):
        output_f, (h_n_f, c_n_f) = self._one_directional_op(input, self.cells, n_steps,
                                                            hs[:self.num_layers], cs[:self.num_layers])

        output_b, (h_n_b, c_n_b) = self._one_directional_op(F.flip(input, 0), self.back_cells, n_steps,
                                                            hs[self.num_layers:], cs[self.num_layers:],
                                                            reverse=True)

        output = F.cat([output_f, output_b], 2)
        h_n = F.cat([h_n_f, h_n_b], 0)
        c_n = F.cat([c_n_f, c_n_b], 0)

        return output, (h_n, c_n)
