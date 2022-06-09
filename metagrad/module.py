import inspect
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
    _modules: Dict[str, Optional['Module']]
    _parameters: Dict[str, Optional[Parameter]]

    def __init__(self) -> None:
        self.training = True
        self._modules = OrderedDict()
        self._parameters = OrderedDict()

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
        for name, module in self._modules.item():
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
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        factory_kwargs = {'device': device, 'dtype': dtype}

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


class ModuleList(Module):
    _modules: Dict[str, Module]

    def __init__(self, modules: Optional[Iterable[Module]] = None) -> None:
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


class LSTMCell(Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(LSTMCell, self).__init__()
        # 组合了 x->input gate; x-> forget gate; x-> g ; x-> output gate 的线性转换
        self.hidden_lin = Linear(hidden_size, 4 * hidden_size)
        # 组合了 h->input gate; h-> forget gate; h-> g ; h-> output gate 的线性转换
        self.input_lin = Linear(input_size, 4 * hidden_size, bias=False)
