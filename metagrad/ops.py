from typing import Any, Tuple

import numpy as np

from metagrad.cuda import get_array_module
from metagrad.tensor import Tensor, NdArray

'''
ops.py保存所有运算操作相关的类
'''


class Function:
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

    def forward(ctx, *args: Any, **kwargs: Any) -> NdArray:
        '''前向传播，进行真正运算的地方'''
        raise NotImplementedError("You must implement the forward function for custom Function.")

    def backward(ctx, grad: NdArray) -> Any:
        '''实现反向传播，计算梯度'''
        raise NotImplementedError("You must implement the backward method for your custom Function "
                                  "to use it with backward mode AD.")

    def apply(fxn, *xs: "Tensor", **kwargs) -> "Tensor":
        '''与PyTorch一样，我们也不直接调用forward，而是调用此方法'''
        # 先调用构造函数，传入运算依赖的Tensor
        ctx = fxn(*xs)  # 调用到了_Function的__init__方法
        # [t.data for t in xs]遍历Tensor中的data(NdArray)值，参与实际计算的都是NumPy的数组。
        ret = Tensor(ctx.forward(ctx, *[t.data for t in xs], **kwargs),
                     requires_grad=any([t.requires_grad for t in xs]))

        if ret.requires_grad:
            ret._ctx = ctx

        return ret


def unbroadcast(grad: NdArray, in_shape: Tuple) -> NdArray:
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


# ****二元运算****
class Add(Function):

    def forward(ctx, x: NdArray, y: NdArray) -> NdArray:
        '''
        实现 z = x + y ，我们这里的x和y都是Numpy数组，因此可能发生广播，
        在实现反向传播是需要注意
        '''
        # 我们只要保存输入各自的形状即可
        ctx.save_for_backward(x.shape, y.shape)
        # 进行真正的运算
        return x + y

    def backward(ctx, grad: NdArray) -> Tuple[NdArray, NdArray]:
        shape_x, shape_y = ctx.saved_tensors
        # 输入有两个，都是需要计算梯度的，因此输出也是两个
        return unbroadcast(grad, shape_x), unbroadcast(grad, shape_y)


class Sub(Function):
    def forward(ctx, x: NdArray, y: NdArray) -> NdArray:
        '''
        实现 z = x - y
        '''
        ctx.save_for_backward(x.shape, y.shape)
        return x - y

    def backward(ctx, grad: NdArray) -> Tuple[NdArray, NdArray]:
        shape_x, shape_y = ctx.saved_tensors
        return unbroadcast(grad, shape_x), unbroadcast(-grad, shape_y)


class Mul(Function):

    def forward(ctx, x: NdArray, y: NdArray) -> NdArray:
        '''
        实现 z = x * y
        '''
        # 乘法需要保存输入x和y，用于反向传播
        ctx.save_for_backward(x, y)
        return x * y

    def backward(ctx, grad: NdArray) -> Tuple[NdArray, NdArray]:
        x, y = ctx.saved_tensors
        # 分别返回∂L/∂x 和 ∂L/∂y
        return unbroadcast(grad * y, x.shape), unbroadcast(grad * x, y.shape)


# Python3 只有 __truediv__ 相关魔法方法
class TrueDiv(Function):

    def forward(ctx, x: NdArray, y: NdArray) -> NdArray:
        '''
        实现 z = x / y
        '''
        ctx.save_for_backward(x, y)
        return x / y

    def backward(ctx, grad: NdArray) -> Tuple[NdArray, NdArray]:
        x, y = ctx.saved_tensors
        return unbroadcast(grad / y, x.shape), unbroadcast(grad * (-x / y ** 2), y.shape)


# ****聚合运算****
class Sum(Function):
    def forward(ctx, x: NdArray, axis=None, keepdims=False) -> NdArray:
        ctx.save_for_backward(x.shape)
        return x.sum(axis, keepdims=keepdims)

    def backward(ctx, grad: NdArray) -> NdArray:
        x_shape, = ctx.saved_tensors
        # 将梯度广播成input_shape形状,梯度的维度要和输入的维度一致
        xp = get_array_module(grad)
        return xp.broadcast_to(grad, x_shape)


class Mean(Function):
    def forward(ctx, x: NdArray, axis=None, keepdims=False) -> NdArray:
        out = x.mean(axis, keepdims=keepdims)
        ctx.save_for_backward(x.shape, out.shape, axis, keepdims)
        return out

    def backward(ctx, grad: NdArray) -> NdArray:
        x_shape, out_shape, axis, keepdims = ctx.saved_tensors
        grad = grad * (np.prod(out_shape) / np.prod(x_shape))
        ndim = len(x_shape)
        axis = (axis,) if np.isscalar(axis) else axis
        if not (ndim == 0 or axis is None or keepdims):
            actual_axis = [ax if ax > 0 else ax + ndim for ax in axis]
            shape = list(grad.shape)
            for ax in sorted(actual_axis):
                shape.insert(ax, 1)
            grad = grad.reshape(shape)
        # 将梯度广播成input_shape形状,梯度的维度要和输入的维度一致
        xp = get_array_module(grad)

        return xp.broadcast_to(grad, x_shape)


class Max(Function):
    def forward(ctx, x: NdArray, axis=None, keepdims=False) -> NdArray:
        ret = np.amax(x, axis=axis, keepdims=keepdims)
        ctx.save_for_backward(x, axis, ret, keepdims)
        return ret

    def backward(ctx, grad: NdArray) -> NdArray:
        x, axis, ret, keepdims = ctx.saved_tensors
        mask = (x == ret)
        div = mask.sum(axis=axis, keepdims=keepdims)
        return mask * grad / div


class Clip(Function):
    def forward(ctx, x: NdArray, x_min=None, x_max=None) -> NdArray:
        if x_min is None:
            x_min = np.min(x)
        if x_max is None:
            x_max = np.max(x)

        ctx.save_for_backward(x, x_min, x_max)
        xp = get_array_module(x)
        return xp.clip(x, x_min, x_max)

    def backward(ctx, grad: NdArray) -> NdArray:
        x, x_min, x_max = ctx.saved_tensors
        mask = (x >= x_min) * (x <= x_max)
        return grad * mask


# ****矩阵运算****
class Matmul(Function):
    def forward(ctx, x: NdArray, y: NdArray) -> NdArray:
        '''
        z = x @ y
        '''
        assert x.ndim > 1 and y.ndim > 1, f"the dim number of x or y must >=2, actual x:{x.ndim}  and y:{y.ndim}"
        ctx.save_for_backward(x, y)
        return x @ y

    def backward(ctx, grad: NdArray) -> Tuple[NdArray, NdArray]:
        x, y = ctx.saved_tensors
        return unbroadcast(grad @ y.swapaxes(-2, -1), x.shape), unbroadcast(x.swapaxes(-2, -1) @ grad, y.shape)


# ****一元运算****
class Pow(Function):
    def forward(ctx, x: NdArray, c: NdArray) -> NdArray:
        ctx.save_for_backward(x, c)
        return x ** c

    def backward(ctx, grad: NdArray) -> Tuple[NdArray, None]:
        x, c = ctx.saved_tensors
        # 把c当成一个常量，不需要计算梯度
        return grad * c * x ** (c - 1), None


class Log(Function):
    def forward(ctx, x: NdArray) -> NdArray:
        ctx.save_for_backward(x)
        # log = ln
        xp = get_array_module(x)
        return xp.log(x)

    def backward(ctx, grad: NdArray) -> NdArray:
        x, = ctx.saved_tensors
        return grad / x


class Exp(Function):
    def forward(ctx, x: NdArray) -> NdArray:
        xp = get_array_module(x)
        ret = xp.exp(x)
        ctx.save_for_backward(ret)
        return ret

    def backward(ctx, grad: NdArray) -> NdArray:
        ret, = ctx.saved_tensors
        return grad * ret


class Neg(Function):
    def forward(ctx, x: NdArray) -> NdArray:
        return -x

    def backward(ctx, grad: NdArray) -> NdArray:
        return -grad


class Abs(Function):
    def forward(ctx, x: NdArray) -> NdArray:
        xp = get_array_module(x)
        ctx.save_for_backward(x, xp)
        return xp.abs(x)

    def backward(ctx, grad: NdArray) -> NdArray:
        x, xp = ctx.saved_tensors
        # x中元素为0的位置，返回0
        # 否则返回+1/-1
        return grad * xp.where(x == 0, 0, x / xp.abs(x))


# ****变形和切片****
class Slice(Function):
    def forward(ctx, x: NdArray, slices: Any) -> NdArray:
        '''
        z = x[slices]
        '''
        ctx.save_for_backward(x.shape, slices)
        return x[slices]

    def backward(ctx, grad) -> Tuple[NdArray, None]:
        x_shape, slices = ctx.saved_tensors

        xp = get_array_module(grad)

        bigger_grad = xp.zeros(x_shape, dtype=grad.dtype)

        if xp is np:
            np.add.at(bigger_grad, slices, grad)
        else:
            bigger_grad.scatter_add(slices, grad)

        return bigger_grad, None


class Reshape(Function):
    def forward(ctx, x: NdArray, shape: Tuple) -> NdArray:
        ctx.save_for_backward(x.shape)
        return x.reshape(shape)

    def backward(ctx, grad: NdArray) -> Tuple[NdArray, None]:
        x_shape, = ctx.saved_tensors
        return grad.reshape(x_shape), None


class Transpose(Function):
    def forward(ctx, x: NdArray, axes) -> NdArray:
        ctx.save_for_backward(axes)
        return x.transpose(axes)

    def backward(ctx, grad: NdArray) -> Any:
        axes, = ctx.saved_tensors
        if axes is None:
            return grad.transpose()
        xp = get_array_module(grad)

        return grad.transpose(tuple(xp.argsort(axes))), None
