import contextlib
import importlib
import inspect
from typing import Union, Tuple, Any

import numpy as np
import time

# 默认数据类型
_type = np.float32

# 可以转换为Numpy数组的类型
Arrayable = Union[float, list, np.ndarray]


def ensure_array(arrayable: Arrayable) -> np.ndarray:
    """
    :param arrayable:
    :return:
    """
    if isinstance(arrayable, (np.ndarray, slice, tuple)):
        # 如果本身是ndarray或slice或tuple(元组里面都是slice)
        return arrayable
    # 转换为Numpy数组
    return np.array(arrayable, dtype=_type)


Tensorable = Union["Tensor", float, np.ndarray]


def ensure_tensor(tensoralbe: Tensorable) -> "Tensor":
    '''
    确保是Tensor对象
    '''
    if isinstance(tensoralbe, Tensor):
        return tensoralbe
    return Tensor(tensoralbe)


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
    def __init__(self, data: Arrayable, requires_grad: bool = False) -> None:
        '''
        初始化Tensor对象
        Args:
            data: 数据
            requires_grad: 是否需要计算梯度
        '''

        # data 是 np.ndarray
        self._data = ensure_array(data)

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
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, new_data: np.ndarray) -> None:
        self._data = ensure_array(new_data)
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
    def dtype(self) -> np.dtype:
        '''返回Tensor中数据的类型'''
        return self.data.dtype

    @property
    def size(self) -> int:
        '''
        返回Tensor中元素的个数 等同于np.prod(a.shape)
        Returns:
        '''
        return self.data.size

    def zero_grad(self) -> None:
        '''
        将梯度初始化为0
        Returns:

        '''
        self._grad = Tensor(np.zeros_like(self.data, dtype=_type))

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def __len__(self) -> int:
        return len(self.data)

    def assign(self, x) -> "Tensor":
        '''将x的值赋予当前Tensor'''
        x = ensure_tensor(x)
        # 维度必须一致
        assert x.shape == self.shape
        self.data = x.data
        return self

    def numpy(self) -> np.ndarray:
        """转换为Numpy数组"""
        return self.data

    def item(self) -> Any:
        return self.numpy().item()

    def squeeze(self) -> Any:
        return self.numpy().squeeze()

    # 切片操作
    def __getitem__(self, idxs) -> "Tensor":
        return self.slice(idxs)

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
                self._grad = Tensor(1)
            else:
                # 如果当前Tensor得到不是标量，那么grad必须制定
                raise RuntimeError("grad must be specified for non scalar")
        else:
            self._grad = ensure_tensor(grad)

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
                    gt = Tensor(g)
                    t._grad = gt if t.grad is None else t.grad + gt

    # ****帮助函数****
    @classmethod
    def empty(cls, *shape, **kwargs):
        return cls(np.empty(*shape, dtype=_type), **kwargs)

    @classmethod
    def zeros(cls, *shape, **kwargs):
        return cls(np.zeros(shape, dtype=_type), **kwargs)


def register(name, fxn):
    def dispatch(*xs, **kwargs):
        # 把所有的输入都转换为Tensor
        xs = [ensure_tensor(x) for x in xs]
        # 调用apply方法
        return fxn.apply(fxn, *xs, **kwargs)

    if name in ["pow", "neg"]:
        setattr(Tensor, f"__{name}__", dispatch)
    else:
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
            # 注册所有_Function的子类
            register(name.lower(), cls)


try:
    _register_ops(importlib.import_module("metagrad.ops"))
except ImportError as e:
    print(e)
