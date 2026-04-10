import math
import operator
import pickle
from collections import OrderedDict
from itertools import chain, islice
from typing import List, Optional, Tuple, Dict, Iterable, Union, Iterator, Set

import metagrad.functions as F
from metagrad import init
from metagrad.paramater import Parameter
from metagrad.tensor import Tensor, no_grad, float_type
from metagrad.rnn_utils import PackedSequence


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
        """
        调用super().__setattr__('a', a)而不是self.a=a防止调用Module.__setattr__的开销

        Module.__setattr__具有额外的对parameters,submodules的处理
        """
        super().__setattr__('training', True)
        super().__setattr__('_parameters', OrderedDict())
        super().__setattr__('_modules', OrderedDict())

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

    def state_dict(self, destination=None, prefix=""):
        if destination is None:
            destination = OrderedDict()

        for name, param in self._parameters.items():
            if param is not None:
                destination[prefix + name] = param.data
        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(destination, prefix + name + ".")

        return destination

    def _load_from_state_dict(self, state_dict, prefix):
        local_name_params = self._parameters.items()
        local_state = {k: v for k, v in local_name_params if v is not None}

        for name, param in local_state.items():
            key = prefix + name
            if key in state_dict:
                input_param = state_dict[key]
                with no_grad():
                    # 赋值给param
                    param.data = input_param

    def load_state_dict(self, state_dict):
        state_dict = OrderedDict(state_dict)

        def load(module, local_state_dict, prefix=""):
            module._load_from_state_dict(local_state_dict, prefix)
            for name, child in module._modules.items():
                if child is not None:
                    child_prefix = prefix + name + '.'
                    child_state_dict = {k: v for k, v in local_state_dict.items() if k.startswith(child_prefix)}
                    load(child, child_state_dict, child_prefix)

        load(self, state_dict)
        del load

    def save(self, path='model.pkl'):
        state_dict = self.state_dict()
        with open(path, 'wb') as f:
            pickle.dump(state_dict, f)
            print(f"Save module to {path}")

    def load(self, path='model.pkl'):
        with open(path, 'rb') as f:
            state_dict = pickle.load(f)
        self.load_state_dict(state_dict)

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

    def apply(self, fn):
        for module in self.children():
            module.apply(fn)

        fn(self)
        return self

    def to_gpu(self, device):
        return self._apply(lambda t: t.to_gpu(device))

    def to_cpu(self):
        return self._apply(lambda t: t.to_cpu())

    def to(self, device):
        return self._apply(lambda t: t.to(device))

    def __setattr__(self, name: str, value: Union[Tensor, 'Module']) -> None:
        '''
        通过该魔法方法注册属性到Module中
        Args:
            name:
            value:

        Returns:

        '''

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
                super().__setattr__(name, value)

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __getstate(self):
        state = self.__dict__.copy()
        return state

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
            super().__delattr__(name)

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
                 dtype=float_type, device=None, padding_idx: Optional[int] = None) -> None:
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

    def extra_repr(self) -> str:
        s = '{num_embeddings}, {embedding_dim}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        return s.format(**self.__dict__)


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

        str_indices = [str(i) for i in range(len(self._modules))]
        self._modules = OrderedDict(list(zip(str_indices, self._modules.values())))

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
    def extra_repr(self) -> str:
        s = 'input_size={input_size}, hidden_size={hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        return s.format(**self.__dict__)


class RNNCell(RNNCellBase):
    def __init__(self, input_size, hidden_size: int, bias: bool = True, nonlinearity: str = 'tanh'):
        '''
        RNN单元基类
        Args:
            input_size: 输入大小
            hidden_size:  隐藏大小
            bias: 是否有偏置
            nonlinearity: 激活函数 tanh | relu 仅用于RNN
        '''

        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.weight_ih = Parameter(Tensor.empty((hidden_size, input_size)))  # input to hidden weight
        self.weight_hh = Parameter(Tensor.empty((hidden_size, hidden_size)))  # hidden to hidden weight
        if bias:
            self.bias_ih = Parameter(Tensor.empty(hidden_size))  # input to hidden bias
            self.bias_hh = Parameter(Tensor.empty(hidden_size))  # hidden to hidden bias
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def forward(self, input, hx) -> Tensor:
        if self.nonlinearity == "tanh":
            func = F.RNNTanhCell  # 以tanh作为激活函数的RNN单元
        else:
            func = F.RNNReLUCell  # 以ReLU作为激活函数的RNN单元

        return func(input, hx, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)


class LSTMCell(RNNCellBase):
    def __init__(self, input_size, hidden_size: int, bias: bool = True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(Tensor.empty((4 * hidden_size, input_size)))
        self.weight_hh = Parameter(Tensor.empty((4 * hidden_size, hidden_size)))
        if bias:
            self.bias_ih = Parameter(Tensor.empty(4 * hidden_size))
            self.bias_hh = Parameter(Tensor.empty(4 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def forward(self, input, hx):
        return F.LSTMCell(
            input, hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,
        )


class GRUCell(RNNCellBase):
    def __init__(self, input_size, hidden_size: int, bias: bool = True):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(Tensor.empty((3 * hidden_size, input_size)))
        self.weight_hh = Parameter(Tensor.empty((3 * hidden_size, hidden_size)))
        if bias:
            self.bias_ih = Parameter(Tensor.empty(3 * hidden_size))
            self.bias_hh = Parameter(Tensor.empty(3 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def forward(self, input, hx):
        return F.GRUCell(
            input, hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,
        )


class RNNBase(Module):
    def __init__(self, mode, input_size: int, hidden_size: int, num_layers: int = 1, bias: bool = True,
                 batch_first: bool = False, dropout: float = 0, bidirectional: bool = False,
                 device=None, dtype=None) -> None:
        '''
           :param mode:  RNN|GRU|LSTM
           :param input_size:  输入x的特征数
           :param hidden_size: 隐藏状态的特征数
           :param num_layers: 层数
           :param bias: 线性层是否包含偏置
           :param batch_first: 批次维度是否在前面
           :param dropout: 用于多层堆叠RNN，默认为0代表不使用dropout
           :param bidirectional: 是否为双向
           :param device:
           :param dtype:
       '''
        super(RNNBase, self).__init__()

        factory_kwargs = {'device': device, 'dtype': dtype}

        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        num_directions = 2 if self.bidirectional else 1

        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions
                if mode == 'LSTM':
                    gate_size = 4 * hidden_size
                elif mode == 'GRU':
                    gate_size = 3 * hidden_size
                else:
                    gate_size = hidden_size

                w_ih = Parameter(Tensor.empty((gate_size, layer_input_size), **factory_kwargs))
                w_hh = Parameter(Tensor.empty((gate_size, hidden_size), **factory_kwargs))
                b_ih = Parameter(Tensor.empty(gate_size, **factory_kwargs))
                b_hh = Parameter(Tensor.empty(gate_size, **factory_kwargs))

                if bias:
                    layer_params = (w_ih, w_hh, b_ih, b_hh)
                else:
                    layer_params = (w_ih, w_hh)

                suffix = '_reverse' if direction == 1 else ''
                param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
                if bias:
                    param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']

                param_names = [x.format(layer, suffix) for x in param_names]
                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)

                self._all_weights.append(param_names)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def forward(self, input, hx=None):
        # 判断是否为压缩序列
        is_packed = isinstance(input, PackedSequence)
        if is_packed:
            # 从中抽出输入和batch_sizes
            input, batch_sizes = input
            # 第0个时间步一定是最大批次
            max_batch_size = batch_sizes[0]
        else:
            # 否则max_batch_size就是批大小
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            # (num_layers * num_directions, batch_size, hidden_size)
            hx = Tensor.zeros((self.num_layers * num_directions, max_batch_size, self.hidden_size), device=input.device)

            if self.mode == 'LSTM':
                hx = (hx, hx)  # 如果是LSTM，同时初始化隐藏状态h，与单元状态c

        func = F.RNN(
            self.mode,
            num_layers=self.num_layers,
            batch_first=self.batch_first,
            dropout=self.dropout,
            train=self.training,
            bidirectional=self.bidirectional,
            batch_sizes=batch_sizes  # 传入batch_sizes
        )

        output, hidden = func(input, self.all_weights, hx)
        if is_packed:
            # 转换为PackedSequence
            # output (total_seq_len, hidden_size)
            output = PackedSequence(output, batch_sizes)

        return output, hidden

    @property
    def all_weights(self):
        return [[getattr(self, weight) for weight in weights] for weights in self._all_weights]

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
        :param num_layers: 层数
        :param nonlinearity: 非线性激活函数 tanh | relu
        :param bias: 线性层是否包含偏置
        :param batch_first:
        :param dropout: 用于多层堆叠RNN，默认为0代表不使用dropout
        :param bidirectional: 是否为双向
        '''
        if 'nonlinearity' in kwargs:
            if kwargs['nonlinearity'] == 'tanh':
                mode = 'RNN_TANH'
            else:
                mode = 'RNN_RELU'

            del kwargs['nonlinearity']
        else:
            mode = 'RNN_TANH'

        super(RNN, self).__init__(mode, *args, **kwargs)


class GRU(RNNBase):
    def __init__(self, *args, **kwargs):
        '''
        :param input_size:  输入x的特征数
        :param hidden_size: 隐藏状态的特征数
        :param num_layers: 层数
        :param bias: 线性层是否包含偏置
        :param batch_first:
        :param dropout: 用于多层堆叠RNN，默认为0代表不使用dropout
        :param bidirectional: 是否为双向
        '''
        super(GRU, self).__init__('GRU', *args, **kwargs)


class LSTM(RNNBase):

    def __init__(self, *args, **kwargs):
        super(LSTM, self).__init__('LSTM', *args, **kwargs)


class LayerNorm(Module):
    def __init__(self, features: int, eps: float = 1e-6):
        """

        Args:
            features: 特征个数
            eps:
        """

        super().__init__()
        self.gamma = Parameter(Tensor.ones(features))
        self.beta = Parameter(Tensor.zeros(features))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """

        Args:
            x: (batch_size, input_len, emb_size)

        Returns:

        """
        mean = x.mean(-1, keepdims=True)
        std = x.std(-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
