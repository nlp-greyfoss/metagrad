from typing import Any, Tuple
from numpy import ndarray

import numpy as np

from core.tensor import Tensor

'''
ops.py保存所有运算操作相关的类
'''


class _Function:
    def __init__(self, *tensors: "Tensor") -> None:
        # 该操作所依赖的所有输入
        self.depends_on = [t for t in tensors]
        # 保存需要在backward()中使用的Tensor或其他对象(如Shape)
        self.saved_tensors = []

    def __new__(cls, *args, **kwargs):
        '''__new__是静态方法，当该类被实例化时调用'''
        # 把以下方法转换为静态方法，我们可以通过类名直接调用
        cls.forward = staticmethod(cls.forward)
        cls.backward = staticmethod(cls.backward)
        cls.apply = staticmethod(cls.apply)
        return super().__new__(cls)

    def save_for_backward(ctx, *x: Any) -> None:
        ctx.saved_tensors.extend(x)

    def forward(ctx, *args: Any, **kwargs: Any) -> ndarray:
        '''前向传播，进行真正运算的地方'''
        raise NotImplementedError("You must implement the forward function for custom Function.")

    def backward(ctx, grad: ndarray) -> Any:
        '''实现反向传播，计算梯度'''
        raise NotImplementedError("You must implement the backward method for your custom Function "
                                  "to use it with backward mode AD.")

    def apply(fxn, *xs: "Tensor", **kwargs) -> "Tensor":
        '''与PyTorch一样，我们也不直接调用forward，而是调用此方法'''
        # 先调用构造函数，传入运算依赖的Tensor
        ctx = fxn(*xs)  # 调用到了_Function的__init__方法
        # [t.data for t in xs]遍历Tensor中的data(ndarray)值，参与实际计算的都是NumPy的数组。
        ret = Tensor(ctx.forward(ctx, *[t.data for t in xs], **kwargs),
                     requires_grad=any([t.requires_grad for t in xs]))

        if ret.requires_grad:
            ret._ctx = ctx

        return ret


def unbroadcast(grad: ndarray, in_shape: Tuple) -> ndarray:
    '''
    广播操作的逆操作，确保grad转换成in_shape的形状
    Args:
        grad: 梯度
        in_shape: 梯度要转换的形状
    Returns:
    '''
    # 首先计算维度个数之差
    ndims_added = grad.ndim - len(in_shape)
    # 由于广播时，先从左边插入，再进行复制，所以逆操作时，也从左边开始，进行复制的逆操作（求和）
    for _ in range(ndims_added):
        # 在axis=0上进行求和，去掉第0个维度，如果ndims_added > 1，就需要不停的在第0个维度上面求和
        grad = grad.sum(axis=0)

    # 处理 (2,3) + (1,3) => (2,3) grad的情况
    # 看in_shape中有没有维度=1的情况
    for i, dim in enumerate(in_shape):
        if dim == 1:
            # 那么需要在该axis上求和，并且保持维度 这里(2,3) => (1,3) grad 就和输入维度保持一致了
            grad = grad.sum(axis=i, keepdims=True)

    return grad


class Add(_Function):

    def forward(ctx, x: ndarray, y: ndarray) -> ndarray:
        '''
        实现 z = x + y ，我们这里的x和y都是Numpy数组，因此可能发生广播，
        在实现反向传播是需要注意
        '''
        # 我们只要保存输入各自的形状即可
        ctx.save_for_backward(x.shape, y.shape)
        # 进行真正的运算
        return x + y

    def backward(ctx, grad: ndarray) -> Tuple[ndarray, ndarray]:
        shape_x, shape_y = ctx.saved_tensors
        # 输入有两个，都是需要计算梯度的，因此输出也是两个
        return unbroadcast(grad, shape_x), unbroadcast(grad, shape_y)


class Sub(_Function):
    def forward(ctx, x: ndarray, y: ndarray) -> ndarray:
        '''
        实现 z = x - y
        '''
        ctx.save_for_backward(x.shape, y.shape)
        return x - y

    def backward(ctx, grad: ndarray) -> Tuple[ndarray, ndarray]:
        print(type(grad))
        shape_x, shape_y = ctx.saved_tensors
        return unbroadcast(grad, shape_x), unbroadcast(-grad, shape_y)


class Mul(_Function):

    def forward(ctx, x: ndarray, y: ndarray) -> ndarray:
        '''
        实现 z = x * y
        '''
        # 乘法需要保存输入x和y，用于反向传播
        ctx.save_for_backward(x, y)
        return x * y

    def backward(ctx, grad: ndarray) -> Tuple[ndarray, ndarray]:
        x, y = ctx.saved_tensors
        # 分别返回∂L/∂x 和 ∂L/∂y
        return unbroadcast(grad * y, x.shape), unbroadcast(grad * x, y.shape)


# Python3 只有 __truediv__ 相关魔法方法
class TrueDiv(_Function):

    def forward(ctx, x: ndarray, y: ndarray) -> ndarray:
        '''
        实现 z = x / y
        '''
        ctx.save_for_backward(x, y)
        return x / y

    def backward(ctx, grad: ndarray) -> Tuple[ndarray, ndarray]:
        x, y = ctx.saved_tensors
        return unbroadcast(grad / y, x.shape), unbroadcast(grad * (-x / y ** 2), y.shape),
