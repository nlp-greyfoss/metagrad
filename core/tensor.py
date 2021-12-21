from typing import Union, Tuple, Any


import numpy as np

# 默认数据类型
_type = np.float32

# 可以转换为Numpy数组的类型
Arrayable = Union[float, list, np.ndarray]


def ensure_array(arrayable: Arrayable) -> np.ndarray:
    """
    :param arrayable:
    :return:
    """
    if isinstance(arrayable, np.ndarray):
        # 如果本身是ndarray
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


class _Function:
    def __init__(self, *tensors: "Tensor") -> None:
        # 该操作所依赖的所有输入
        self.depends_on = [t for t in tensors]
        # 保存需要在backward()中使用的Tensor或其他对象(如Shape)
        self.saved_tensors = []

    def __new__(cls, *args, **kwargs):
        '''__new__是静态方法，当该类被实例化时调用'''
        # 把这两个方法转换为静态方法，我们可以通过类名直接调用
        cls.forward = staticmethod(cls.forward)
        cls.backward = staticmethod(cls.backward)
        return super().__new__(cls)

    def save_for_backward(ctx, *x: Any) -> None:
        ctx.saved_tensors.extend(x)

    def forward(ctx, *args: Any, **kwargs: Any) -> np.ndarray:
        '''前向传播，进行真正运算的地方'''
        raise NotImplementedError("You must implement the forward function for custom Function.")

    def backward(ctx, grad: Any) -> Any:
        '''实现反向传播，计算梯度'''
        raise NotImplementedError("You must implement the backward method for your custom Function "
                                  "to use it with backward mode AD.")

    def apply(self, ctx, *xs: "Tensor", **kwargs) -> "Tensor":
        '''与PyTorch一样，我们也不直接调用forward，而是调用此方法'''
        # 先调用构造函数，传入运算依赖的Tensor
        # [t.data for t in xs]遍历Tensor中的data(np.ndarray)值，参与实际计算的都是NumPy的数组。
        ret = Tensor(self.forward(ctx, *[t.data for t in xs], **kwargs),
                     requires_grad=any([t.requires_grad for t in xs]))

        if ret.requires_grad:
            ret._ctx = ctx

        return ret


class Add(_Function):

    def forward(ctx, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        '''
        实现 z = x + y ，我们这里的x和y都是Numpy数组，因此可能发生广播，
        在实现反向传播是需要注意
        '''
        # 我们只要保存输入各自的形状即可
        ctx.save_for_backward(x.shape, y.shape)
        # 进行真正的运算
        return x + y

    def backward(ctx, grad: Any) -> Any:
        # 输入有两个，都是需要计算梯度的，因此输出也是两个
        return grad, grad


class Mul(_Function):

    def forward(ctx, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        '''
        实现 z = x * y
        '''
        # 乘法需要保存输入x和y，用于反向传播
        ctx.save_for_backward(x, y)
        return x * y

    def backward(ctx, grad: Any) -> Any:
        x, y = ctx.saved_tensors
        # 分别返回∂L/∂x 和 ∂L/∂y
        return grad * y, grad * x


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

    def __add__(self, other):
        ctx = Add(self, ensure_tensor(other))
        return ctx.apply(ctx, self, ensure_tensor(other))

    def __mul__(self, other):
        ctx = Mul(self, ensure_tensor(other))
        return ctx.apply(ctx, self, ensure_tensor(other))

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

        self._grad = grad
        # 如果传递过来的grad为空
        if grad is None:
            if self.shape == ():
                # 设置梯度值为1，grad本身不需要计算梯度
                self._grad = Tensor(1)

        for t in self._rev_topo_sort():
            assert t.grad is not None
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


'''
ops.py保存所有运算操作相关的类
'''

if __name__ == '__main__':
    a, b = Tensor(2, requires_grad=True), Tensor(1, requires_grad=True)
    e = (a + b) * (b + 1)
    e.backward()
    print(f'grad of a:{a.grad}')
    print(f'grad of b:{b.grad}')

