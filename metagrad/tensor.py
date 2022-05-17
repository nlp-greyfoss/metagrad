import contextlib
import importlib
import inspect
import time
from numbers import Number
from typing import Union, Tuple, Any, List

import numpy as np

from metagrad import cuda
from metagrad.cuda import (
    Device,
    get_device_from_array,
    get_device,
    CpuDevice,
    GpuDevice,
    check_cuda_available,
    get_gpu_device_or_current,
    get_array_module
)

_type = np.float32

# 设置显示精度
np.set_printoptions(precision=4)
# 抑制小数的科学计数法显示
np.set_printoptions(suppress=True)

NdArray = Union['np.ndarray', 'cuda.ndarray']

# 可以转换为数组的类型
Arrayable = Union[Number, NdArray, List]


def ensure_array(arrayable: Arrayable, dtype=None, device=None) -> NdArray:
    """
    :param arrayable:
    :param dtype:
    :return:
    """
    if device is not None:
        xp = device.xp
    else:
        xp = np

    if isinstance(arrayable, (Number, list)):
        # 让xp自己判断数据类型
        return xp.array(arrayable, dtype=dtype)
    elif isinstance(arrayable, (np.ndarray, cuda.ndarray)):
        if device is not None and get_array_module(arrayable) != xp:
            return device.transfer(arrayable)

    return arrayable


Tensorable = Union["Tensor", Number, NdArray]


def ensure_tensor(tensoralbe: Tensorable, device=None) -> "Tensor":
    '''
    确保是Tensor对象
    '''
    if isinstance(tensoralbe, Tensor):
        return tensoralbe
    return Tensor(tensoralbe, device=device)


class Config:
    debug = False
    backprop = True  # 是否需要计算并反向传播梯度


# 上下文管理器
# contextmanager 这个装饰器(decorator)接收一个生成器(generator)
# 该generator必须只yield一个值出来，该值会被用在with语句中，绑定到as后面的变量
# 我们这里只需要修改Config内部状态，不需要返回任何值，可以只加一个yield
@contextlib.contextmanager
def using_config(name, value):
    # 保存旧值
    old_value = getattr(Config, name)
    # 设置新值
    setattr(Config, name, value)
    try:
        yield
    finally:
        # 最终设回旧值
        setattr(Config, name, old_value)


def debug_mode():
    return using_config("debug", True)


def no_grad():
    return using_config("backprop", False)


class OpWrapper:
    '''
    支持反向传播的Debug
    '''

    def __init__(self, name, xs, backward=False):
        self.name = f"back_{name}" if backward else name
        self.xs = xs
        self.output = None

    def __enter__(self):
        if Config.debug:
            self.start = time.time()
        return self

    def __exit__(self, *junk):
        if Config.debug:
            end = (time.time() - self.start) * 1000
            print(
                f"{self.name:>20} : {end:>7.2f} ms {str([y.shape for y in self.xs]):>40} "
                f"{'-> ' + str(self.output.shape) if self.output is not None else ''}"
            )


class Tensor:
    def __init__(self, data: Arrayable, requires_grad: bool = False, dtype=None, device: Device = None) -> None:
        '''
        初始化Tensor对象
        Args:
            data: 数据
            requires_grad: 是否需要计算梯度
            dtype: 数据类型，默认为None
            device: 设备类型 CpuDevice 或 GpuDevice
        '''
        if isinstance(data, Tensor):
            if dtype is None:
                dtype = data.dtype
            if device is None:
                device = data.device
            data = data.data

        if device is None:
            device = get_device_from_array(data)

        self._device = device

        # data 是 NdArray
        self._data = ensure_array(data, dtype, self._device)

        self.requires_grad = requires_grad
        # 保存该Tensor的梯度
        self._grad = None

        if self.requires_grad:
            self.zero_grad()

        # 用于计算图的内部变量
        self._ctx = None

    @property
    def grad(self):
        return self._grad

    @property
    def data(self) -> NdArray:
        return self._data

    @data.setter
    def data(self, new_data: NdArray) -> None:
        self._data = ensure_array(new_data, self.device)
        # 重新赋值后就没有梯度了
        self._grad = None

    # ****一些常用属性****
    @property
    def shape(self) -> Tuple:
        '''返回Tensor各维度大小的元素'''
        return self.data.shape

    @property
    def ndim(self) -> int:
        '''返回Tensor的维度个数'''
        return self.data.ndim

    @property
    def dtype(self):
        '''返回Tensor中数据的类型'''
        return self.data.dtype

    @property
    def device(self):
        return self._device

    @property
    def xp(self):
        '''返回当前tensor基于的numpy或cupy'''
        device = self.device
        return np if device is None else device.xp

    def to(self, device):
        device = get_device(device)
        # 如果设备一致了
        if get_device_from_array(self._data) == device:
            return self
        # 转移到设备上
        self._data = device.transfer(self.data)

        self._device = device

        if self._grad is not None:
            self._grad = Tensor(device.transfer(self._grad.data), device=device)

        return self

    def to_cpu(self):
        '''拷贝数据和梯度到CPU'''
        return self.to(CpuDevice())

    def to_gpu(self, device=None):
        '''拷贝数据和梯度到指定的GPU'''
        check_cuda_available()
        return self.to(get_gpu_device_or_current(device))

    def zero_grad(self) -> None:
        '''
        将梯度初始化为0
        Returns:

        '''
        xp = self.xp
        self._grad = Tensor(xp.zeros_like(self.data))

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad}" \
               f"{', device:' + self.device.name if isinstance(self.device, GpuDevice) else ''})"

    def __len__(self) -> int:
        return len(self.data)

    def __gt__(self, other):
        other = ensure_tensor(other, self.device)
        return self.data > other.data

    def __lt__(self, other):
        other = ensure_tensor(other, self.device)
        return self.data < other.data

    def assign(self, x) -> "Tensor":
        '''将x的值赋予当前Tensor'''
        x = ensure_tensor(x, self.device)
        # 维度必须一致
        assert x.shape == self.shape
        self.data = x.data
        return self

    def size(self, dim=None) -> int:
        '''
        如果dim为None，返回Tensor中元素的个数 等同于np.prod(a.shape)；
        如果dim不为None，返回该维度上的元素个数
        Returns:
        '''

        return self.xp.size(self.data, dim)

    def array(self) -> NdArray:
        """转换为Numpy或Cupy数组"""
        return self.data

    def item(self) -> Number:
        '''将只有一个元素的Tensor转换为Python标量'''
        return self.array().item()

    #
    # def squeeze(self, axis=None) -> "Tensor":
    #     return Tensor(self.array().squeeze(axis=axis), device=self.device, requires_grad=self.requires_grad)
    #
    # def unsqueeze(self, axis=Union[int, Tuple]) -> "Tensor":
    #     return Tensor(self.xp.expand_dims(self.array(), axis), device=self.device, requires_grad=self.requires_grad)

    def uniform_(self, low: float = 0.0, high: float = 1.0) -> "Tensor":
        xp = self.device.xp
        self.data = xp.random.uniform(low, high, size=self.shape)
        return self

    def normal_(self, mean: float = 0.0, std: float = 1.0) -> "Tensor":
        xp = self.device.xp
        self.data = xp.random.normal(mean, std, size=self.shape)
        return self

    def index_fill_(self, dim: int, index: "Tensor", value: float) -> "Tensor":
        xp = self.device.xp
        index = index.data

        axis = list(range(self.ndim))
        axis.remove(dim)
        # 扩展索引，put_along_axis需要索引数组和原数组维度一致
        index = xp.expand_dims(index, axis=axis)
        xp.put_along_axis(self.data, index, value, axis=dim)
        return self

    # ****创造帮助函数****
    @classmethod
    def empty(cls, *shape, dtype=_type, device=None, **kwargs):
        return cls(np.empty(*shape, dtype=dtype), device=device, **kwargs)

    @classmethod
    def zeros(cls, *shape, dtype=_type, device=None, **kwargs) -> "Tensor":
        return cls(np.zeros(*shape, dtype=dtype), device=device, **kwargs)

    @classmethod
    def zeros_like(cls, t: "Tensor", **kwargs) -> "Tensor":
        return cls.zeros(t.shape, dtype=t.dtype, device=t.device, **kwargs)

    @classmethod
    def ones(cls, *shape, dtype=_type, device=None, **kwargs) -> "Tensor":
        return cls(np.ones(shape, dtype=dtype), device=device, **kwargs)

    @classmethod
    def ones_like(cls, t: "Tensor", **kwargs) -> "Tensor":
        return cls.ones(t.shape, t.dtype, t.device, **kwargs)

    @classmethod
    def randn(cls, *shape, dtype=_type, device=None, **kwargs) -> "Tensor":
        return cls(np.random.randn(*shape).astype(dtype), device=device, **kwargs)

    @classmethod
    def arange(cls, stop, start=0, step=1, dtype=int, device=None, **kwargs) -> "Tensor":
        stop, start = start, stop
        return cls(np.arange(start=start, stop=stop, step=step).astype(dtype), device=device, **kwargs)

    @classmethod
    def eye(cls, dim, dtype=_type, device=None, **kwargs) -> "Tensor":
        return cls(np.eye(dim).astype(dtype), device=device, **kwargs)

    @classmethod
    def uniform(cls, *shape, low: float = -1.0, high: float = 1.0,
                dtype=_type, device=None, **kwargs) -> "Tensor":
        return cls((np.random.uniform(low, high, size=shape)).astype(dtype), device=device, **kwargs)

    @classmethod
    def multinomial(cls, input: "Tensor", num_samples: int, replace=False) -> "Tensor":
        '''
        返回一个Tensor，每行包含num_samples个索引，从基于input对应行的多项式概率分布采样而来
        Args:
            input: 包含概率的输入，如果不是概率，那么会自动转换为概率
            num_samples: 生成的样本数
            replace: 是否为放回采样，默认为False
        '''

        size = input.size(-1)

        assert replace or num_samples <= size, "cannot sample n_sample > input.size(-1) samples without replacement"
        assert input.ndim <= 2, "prob_dist must be 1 or 2 dim"

        p = input.data / input.data.sum(-1, keepdims=True)
        xp = input.xp

        # 基于numpy.random.choice来实现multinomial
        if input.ndim == 1:
            return Tensor(xp.random.choice(xp.arange(size), replace=replace, size=num_samples, p=p),
                          device=input.device)
        else:
            # 如果input是2D，那么当成1D列表来处理
            ret = []
            for i in range(input.shape[0]):
                ret.append(xp.random.choice(xp.arange(size), replace=replace, size=num_samples, p=p[i]).tolist())

            return Tensor(ret, device=input.device)

    # 切片操作
    def __getitem__(self, idxs) -> "Tensor":
        return self.slice(idxs)

    def __setitem__(self, key, value):
        self.data[key] = value.item()
        # self._grad = None
        return self

    @property
    def T(self) -> "Tensor":
        return self.transpose(axes=None)

    """
     backward函数现在应该从当前节点(Tensor)回溯到所有依赖节点(depends_on)，计算路径上的偏导
        # 我们分为两部分
        # a) 遍历计算图
        #    如果c是a经过某个函数的结果( c=f(a) )，我们无法知道a的梯度，直到我们得到了c的梯度(链式法则)
        #    所以我们需要逆序计算图中的拓扑结构(reverse mode)，相当沿着有向图的←方向(从指向节点到起始节点)进行计算
        # b) 应用梯度
        #    现在我们能访问到每个node,我们用它的backward函数将梯度传递给它们的depends_on
    """

    def _rev_topo_sort(self):
        '''
        a) 遍历计算图，逆序计算图中的拓扑结构
        Returns:
        '''

        def visit(node, visited, nodes):
            # 标记为已访问
            visited.add(node)
            if node._ctx:
                # 遍历所有依赖节点，递归调用visit
                [visit(nd, visited, nodes) for nd in node._ctx.depends_on if nd not in visited]
                # 递归调用结束后将node入nodes
                nodes.append(node)
            # 返回遍历结果
            return nodes

        return reversed(visit(self, set(), []))

    def backward(self, grad: "Tensor" = None) -> None:
        '''
        实现Tensor的反向传播
        Args:
            grad: 如果该Tensor不是标量，则需要传递梯度进来

        Returns:

        '''
        # 只能在requires_grad=True的Tensor上调用此方法
        assert self.requires_grad, "called backward on tensor do not require grad"

        if not Config.backprop:
            return

        # 如果传递过来的grad为空
        if grad is None:
            if self.shape == ():
                # 设置梯度值为1，grad本身不需要计算梯度
                self._grad = Tensor(1, device=self.device)
            else:
                # 如果当前Tensor得到不是标量，那么grad必须制定
                raise RuntimeError("grad must be specified for non scalar")
        else:
            self._grad = ensure_tensor(grad, device=self.device)

        for t in self._rev_topo_sort():
            assert t.grad is not None

            with OpWrapper(t._ctx.__class__.__name__, [t.grad], backward=True):
                # 以逆序计算梯度，调用t相关运算操作的backward静态方法
                # 计算流向其依赖节点上的梯度(流向其下游)
                grads = t._ctx.backward(t._ctx, t.grad.data)

            # 如果只依赖一个输入，我们也通过列表来封装，防止zip将其继续拆分
            if len(t._ctx.depends_on) == 1:
                grads = [grads]

            for t, g in zip(t._ctx.depends_on, grads):
                # 计算其下游节点上的累积梯度，因为可能有多条边
                if t.requires_grad and g is not None:
                    # t.shape要和grad.shape保持一致
                    assert t.shape == g.shape, f"grad shape must match tensor shape in {self._ctx!r}, {g.shape!r} != {t.shape!r}"
                    # grad Tensor
                    gt = Tensor(g, device=self.device)
                    t._grad = gt if t.grad is None else t.grad + gt


def register(name, fxn):
    def dispatch(*xs, **kwargs):

        device = CpuDevice()
        for x in xs:
            gpu_device = GpuDevice.from_array(x.data if isinstance(x, Tensor) else x)
            if gpu_device is not None:
                device = gpu_device
                break
        # 把所有的输入都转换为Tensor
        xs = [ensure_tensor(x, device) for x in xs]
        
        return fxn.apply(fxn, *xs, **kwargs)

    if name in ["pow", "neg", "abs"]:
        setattr(Tensor, f"__{name}__", dispatch)
    # 为Tensor添加属性，名为name，值为dispatch函数引用
    setattr(Tensor, name, dispatch)

    # 这几个方法都有__xx__, __ixx__, __rxx__ 魔法方法
    if name in ["add", "sub", "mul", "truediv", "matmul"]:
        setattr(Tensor, f"__{name}__", dispatch)
        setattr(
            Tensor, f"__i{name}__", lambda self, x: self.assign(dispatch(self, x))
        )  # __i*__ 代表原地操作
        setattr(
            Tensor, f"__r{name}__", lambda self, x: dispatch(x, self)
        )  # __r*__ 代表 other在操作符前, self在操作符后


def _register_ops(namespace):
    for name, cls in inspect.getmembers(namespace, inspect.isclass):
        if name[0] != "_" and name != 'Tensor':
            # 注册所有Function的子类
            register(name.lower(), cls)


try:
    _register_ops(importlib.import_module("metagrad.ops"))
except ImportError as e:
    print(e)
