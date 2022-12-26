import weakref
from typing import Any, Tuple, Union
import numpy as np

from metagrad import cuda
from metagrad.cuda import get_array_module, ndarray

from metagrad.tensor import Tensor, NdArray, Config

'''
ops.py保存所有运算操作相关的类
'''


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


class Function:
    def __init__(self) -> None:
        # 保存需要在backward()中使用的Tensor或其他对象(如Shape)
        self.saved_tensors = []

    def save_for_backward(self, *x: Any) -> None:
        self.saved_tensors.extend(x)

    def forward(self, *args: Any, **kwargs: Any) -> NdArray:
        '''前向传播，进行真正运算的地方'''
        raise NotImplementedError("You must implement the forward function for custom Function.")

    def backward(self, grad: NdArray) -> Any:
        '''实现反向传播，计算梯度'''
        raise NotImplementedError("You must implement the backward method for your custom Function "
                                  "to use it with backward mode AD.")

    def __call__(self, *xs: "Tensor", **kwargs) -> "Tensor":
        raw_xs = [x.data if isinstance(x, Tensor) else x for x in xs]
        # [t.data for t in xs]遍历Tensor中的data(NdArray)值，参与实际计算的都是NumPy的数组。
        ys = self.forward(*raw_xs, **kwargs)
        requires_grad = any([t.requires_grad for t in xs if isinstance(t, Tensor)])

        return_tuple = True
        if not isinstance(ys, tuple):
            return_tuple = False
            ys = (ys,)

        outputs = [Tensor(y, requires_grad=requires_grad) for y in ys]

        if Config.backprop:
            self.generation = max([x.generation for x in xs if isinstance(x, Tensor)])
            for output in outputs:  # 设定每个输出是由此函数得到的
                output.set_creator(self)
            self.inputs = xs  # 记录输入
            self.outputs = [weakref.ref(output) for output in outputs]  # 通过弱引用保存输出

        # 返回多个则通过元组
        if return_tuple or len(outputs) > 1:
            return tuple(outputs)
        return outputs[0]


def broadcast_grad_shape(original_shape, xp, grad: NdArray, axis=None, keepdims=None):
    '''
    在Mean、Sum、Squeeze等方法中，可能会丢失dim=1的维度，不能直接进行广播，需要调用此方法进行一些处理，广播到原来的维度
     Args:
         original_shape: 原来的形状
         xp: numpy 或 cupy
         grad: 梯度
         axis: None 或 int 或 int元组，操作的维度
         keepdims: 是否保存维度，如果为True，那么就不需要调用此方法了。调用此方法的目的就是得到keepdims=True的结果
     Returns: 转换好形状并广播了的grad
     '''

    # 原来的维度数
    ndim = len(original_shape)
    # 如果ndim为零 或 axis 为 None 或 keepdims 为 True
    if ndim == 0 or axis is None or keepdims:
        # 直接进行广播即可
        return xp.broadcast_to(grad, original_shape)

    grad = grad.reshape(restore_grad_shape(original_shape, ndim, axis))

    # 将梯度广播成input_shape形状,梯度的维度要和输入的维度一致
    return xp.broadcast_to(grad, original_shape)


def restore_grad_shape(original_shape, ndim: int, axis=None):
    '''
       在Mean、Sum、Squeeze等方法中，可能会丢失dim=1的维度，不能直接进行广播，需要调用此方法恢复出原来的维度
        Args:
            original_shape: 原来的形状
            axis: None 或 int 或 int元组，操作的维度
        Returns: 得到keepdims=True的形状
    '''
    # 我们需要恢复之前的形状，得到keepdims=True的结果
    if isinstance(axis, (np.ndarray, ndarray)):
        axis = axis.item()
    # 如果axis为int，变成可迭代的列表，方便统一处理
    axis = [axis] if np.isscalar(axis) else axis

    # 支持 -1(最后一个元素)这种索引，转换为正数索引
    axis = tuple(ax % ndim for ax in axis)

    return [s if ax not in axis else 1 for ax, s in enumerate(original_shape)]


# ****二元运算****
class Add(Function):

    def forward(self, x: NdArray, y: NdArray) -> NdArray:
        '''
        实现 z = x + y ，我们这里的x和y都是Numpy数组，因此可能发生广播，
        在实现反向传播是需要注意
        '''
        # 我们只要保存输入各自的形状即可
        self.save_for_backward(x.shape, y.shape)
        # 进行真正的运算
        return x + y

    def backward(self, grad: NdArray) -> Tuple[NdArray, NdArray]:
        shape_x, shape_y = self.saved_tensors
        # 输入有两个，都是需要计算梯度的，因此输出也是两个
        return unbroadcast(grad, shape_x), unbroadcast(grad, shape_y)


class AddConstant(Function):
    '''
    Tensor + 常量
    '''

    def forward(self, x: NdArray, c) -> NdArray:
        return x + x.dtype.type(c)

    def backward(self, grad: NdArray) -> NdArray:
        return grad


def add(self, rhs):
    if np.isscalar(rhs):
        return AddConstant()(self, rhs)
    return Add()(self, rhs)


class Sub(Function):
    def forward(self, x: NdArray, y: NdArray) -> NdArray:
        '''
        实现 z = x - y
        '''
        self.save_for_backward(x.shape, y.shape)
        return x - y

    def backward(self, grad: NdArray) -> Tuple[NdArray, NdArray]:
        shape_x, shape_y = self.saved_tensors
        return unbroadcast(grad, shape_x), unbroadcast(-grad, shape_y)


def sub(self, rhs):
    if np.isscalar(rhs):
        return AddConstant()(self, -rhs)
    return Sub()(self, rhs)


class SubFromConstant(Function):
    '''
    常量 - Tensor
    '''

    def forward(self, x: NdArray, constant) -> NdArray:
        return x.dtype.type(constant) - x

    def backward(self, grad: NdArray) -> NdArray:
        return -grad


def rsub(self, rhs):
    if np.isscalar(rhs):
        return SubFromConstant()(self, rhs)
    return Sub()(rhs, self)


class Mul(Function):

    def forward(self, x: NdArray, y: NdArray) -> NdArray:
        '''
        实现 z = x * y
        '''
        # 乘法需要保存输入x和y，用于反向传播
        self.save_for_backward(x, y)
        return x * y

    def backward(self, grad: NdArray) -> Tuple[NdArray, NdArray]:
        x, y = self.saved_tensors
        # 分别返回∂L/∂x 和 ∂L/∂y

        return unbroadcast(grad * y, x.shape), unbroadcast(grad * x, y.shape)


def mul(self, rhs):
    if np.isscalar(rhs):
        return MulConstant()(self, rhs)
    return Mul()(self, rhs)


class MulConstant(Function):
    """
     Tensor * 常量
    """

    def forward(self, x: NdArray, c) -> NdArray:
        # 乘法需要保存输入x和y，用于反向传播
        self.save_for_backward(c)
        return x * c

    def backward(self, grad: NdArray) -> NdArray:
        c, = self.saved_tensors
        return grad * c


class Div(Function):

    def forward(self, x: NdArray, y: NdArray) -> NdArray:
        '''
        实现 z = x / y
        '''
        self.save_for_backward(x, y)
        return x / y

    def backward(self, grad: NdArray) -> Tuple[NdArray, NdArray]:
        xp = get_array_module(grad)
        x, y = self.saved_tensors

        if xp is np:
            gx = grad / y
            gy = -gx * x / y
        else:
            # 使用自定义内核代码加速
            gx, gy = cuda.elementwise(
                'T x, T y, T grad',
                'T gx, T gy',
                '''
                gx = grad / y;
                gy = -gx * x / y;
                ''',
                'div_bwd'
            )(x, y, grad)

        return unbroadcast(gx, x.shape), unbroadcast(gy, y.shape)


def div(self, rhs):
    if np.isscalar(rhs):
        return MulConstant()(self, 1.0 / rhs)
    return Div()(self, rhs)


class DivFromConstant(Function):
    """
      常量/Tensor
    """

    def forward(self, x: NdArray, c) -> NdArray:
        self.save_for_backward(x, c)
        return c / x

    def backward(self, grad: NdArray) -> NdArray:
        x, c = self.saved_tensors
        xp = get_array_module(grad)

        if xp is np:
            gx = -c * grad / (x ** 2)
        else:
            # 使用自定义内核代码加速
            gx = cuda.elementwise(
                'T x, T y, T grad',
                'T gx',
                '''
                gx = -y * grad / (x*x);
                ''',
                'div_from_const_bwd'
            )(x, c, grad)

        return gx


def rdiv(self, rhs):
    if np.isscalar(rhs):
        return DivFromConstant()(self, rhs)
    return Div()(rhs, self)


# ****聚合运算****
class Sum(Function):
    def forward(self, x: NdArray, axis=None, keepdims=False) -> NdArray:
        self.save_for_backward(x.shape, axis, keepdims)
        return x.sum(axis, keepdims=keepdims)

    def backward(self, grad: NdArray) -> NdArray:
        x_shape, axis, keepdims = self.saved_tensors

        return broadcast_grad_shape(x_shape, get_array_module(grad), grad, axis, keepdims)


class Mean(Function):
    def forward(self, x: NdArray, axis=None, keepdims=False) -> NdArray:
        out = x.mean(axis, keepdims=keepdims)
        self.save_for_backward(x.shape, out.shape, axis, keepdims)
        return out

    def backward(self, grad: NdArray) -> NdArray:
        x_shape, out_shape, axis, keepdims = self.saved_tensors
        grad = grad * (np.prod(out_shape) / np.prod(x_shape))

        return broadcast_grad_shape(x_shape, get_array_module(grad), grad, axis, keepdims)


class Max(Function):
    def forward(self, x: NdArray, axis=None, keepdims=False) -> NdArray:
        '''
        y = x.max()
        '''
        xp = get_array_module(x)
        y = xp.amax(x, axis=axis, keepdims=keepdims)
        self.save_for_backward(x, axis, y, keepdims)
        return y

    def backward(self, grad: NdArray) -> NdArray:
        x, axis, y, keepdims = self.saved_tensors

        if axis is None:
            mask = (x == y)
            div = mask.sum(axis=axis, keepdims=keepdims)
        else:
            shape = restore_grad_shape(x.shape, x.ndim, axis)
            grad = grad.reshape(shape)
            y = y.reshape(shape)

            mask = (x == y)
            div = mask.sum(axis=axis, keepdims=True)

        return mask * grad / div


class Min(Function):
    def forward(self, x: NdArray, axis=None, keepdims=False) -> NdArray:
        '''
        y = x.min()
        '''
        xp = get_array_module(x)
        y = xp.amin(x, axis=axis, keepdims=keepdims)
        self.save_for_backward(x, axis, y, keepdims)
        return y

    def backward(self, grad: NdArray) -> NdArray:
        x, axis, y, keepdims = self.saved_tensors

        if axis is None:
            mask = (x == y)
            div = mask.sum(axis=axis, keepdims=keepdims)
        else:
            shape = restore_grad_shape(x.shape, x.ndim, axis)
            grad = grad.reshape(shape)
            y = y.reshape(shape)

            mask = (x == y)
            div = mask.sum(axis=axis, keepdims=True)

        return mask * grad / div


class Clip(Function):
    def forward(self, x: NdArray, x_min=None, x_max=None) -> NdArray:
        xp = get_array_module(x)
        if x_min is None:
            x_min = xp.min(x)
        if x_max is None:
            x_max = xp.max(x)

        self.save_for_backward(x, x_min, x_max)
        return xp.clip(x, x_min, x_max)

    def backward(self, grad: NdArray) -> NdArray:
        x, x_min, x_max = self.saved_tensors
        mask = (x >= x_min) * (x <= x_max)
        return grad * mask


def dim_one(shape):
    '''
    找到之前维度大小为1的dim
    '''
    result = []
    for i, s in enumerate(shape):
        if s == 1:
            result.append(i)
    return result


class Squeeze(Function):
    def forward(self, x: NdArray, axis: Union[int, Tuple, None] = None) -> NdArray:
        xp = get_array_module(x)

        if isinstance(axis, (np.ndarray, ndarray)):
            axis = axis.item()

        self.save_for_backward(x.shape, axis)

        return xp.squeeze(x, axis)

    def backward(self, grad: NdArray) -> NdArray:
        x_shape, axis = self.saved_tensors

        if axis is None:
            axis = tuple(dim_one(x_shape))

        shape = restore_grad_shape(x_shape, len(x_shape), axis)

        return grad.reshape(shape)


class UnSqueeze(Function):
    def forward(self, x: NdArray, axis: int) -> NdArray:
        xp = get_array_module(x)
        self.save_for_backward(x.shape)

        if isinstance(axis, (np.ndarray, ndarray)):
            axis = axis.item()

        return xp.expand_dims(x, axis)

    def backward(self, grad: NdArray) -> NdArray:
        x_shape, = self.saved_tensors
        return grad.reshape(x_shape)


# ****矩阵运算****
class Matmul(Function):
    def forward(self, x: NdArray, y: NdArray) -> NdArray:
        '''
        z = x @ y
        '''
        assert x.ndim > 1 and y.ndim > 1, f"the dim number of x or y must >=2, actual x:{x.ndim}  and y:{y.ndim}"
        self.save_for_backward(x, y)
        return x @ y

    def backward(self, grad: NdArray) -> Tuple[NdArray, NdArray]:
        x, y = self.saved_tensors
        return unbroadcast(grad @ y.swapaxes(-2, -1), x.shape), unbroadcast(x.swapaxes(-2, -1) @ grad, y.shape)


# ****一元运算****
class Pow(Function):
    """
    Tensor ** 常量
    """

    def forward(self, x: NdArray, c: float) -> NdArray:
        self.save_for_backward(x, c)
        return x ** c

    def backward(self, grad: NdArray) -> NdArray:
        x, c = self.saved_tensors
        xp = get_array_module(x)
        # 把c当成一个常量，不需要计算梯度
        if xp is np:
            return grad * c * x ** (c - 1)
        else:
            return cuda.elementwise(
                'T x, T grad, T c', 'T gx',
                'gx = c * pow(x, c - 1) * grad',
                'pow_bwd')(x, grad, c)


class Log(Function):
    def forward(self, x: NdArray) -> NdArray:
        self.save_for_backward(x)
        # log = ln
        xp = get_array_module(x)
        return xp.log(x)

    def backward(self, grad: NdArray) -> NdArray:
        x, = self.saved_tensors
        return grad / x


class Exp(Function):
    def forward(self, x: NdArray) -> NdArray:
        xp = get_array_module(x)
        ret = xp.exp(x)
        self.save_for_backward(ret)
        return ret

    def backward(self, grad: NdArray) -> NdArray:
        ret, = self.saved_tensors
        return grad * ret


class Neg(Function):
    def forward(self, x: NdArray) -> NdArray:
        return -x

    def backward(self, grad: NdArray) -> NdArray:
        return -grad


class Abs(Function):
    def forward(self, x: NdArray) -> NdArray:
        xp = get_array_module(x)
        self.save_for_backward(x, xp)
        return xp.abs(x)

    def backward(self, grad: NdArray) -> NdArray:
        x, xp = self.saved_tensors
        # x中元素为0的位置，返回0
        # 否则返回+1/-1
        if xp is np:
            return grad * np.sign(x)
        else:
            return cuda.elementwise(
                'T x, T gy', 'T gx',
                'gx = ((x > 0) - (x < 0)) * gy',
                'abs_bwd')(x, grad)


class Sqrt(Function):
    def forward(self, x: NdArray) -> NdArray:
        xp = get_array_module(x)
        ret = xp.sqrt(x)
        self.save_for_backward(ret)
        return ret

    def backward(self, grad: NdArray) -> NdArray:
        ret, = self.saved_tensors
        return grad / (ret * 2.0)


def argone(shape):
    '''
    找到之前维度大小为1的dim
    '''
    result = []
    for i, s in enumerate(shape):
        if s == 1:
            result.append(i)
    return result


# ****变形和切片****
class Slice(Function):

    def forward(self, x: NdArray, slices: Any) -> NdArray:
        '''
        z = x[slices]
        '''
        self.save_for_backward(x.shape, slices)
        return x[slices]

    def backward(self, grad) -> NdArray:
        x_shape, slices = self.saved_tensors

        xp = get_array_module(grad)

        bigger_grad = xp.zeros(x_shape, dtype=grad.dtype)

        if xp is np:
            np.add.at(bigger_grad, slices, grad)
        else:
            bigger_grad.scatter_add(slices, grad)

        return bigger_grad


class _IndexSelect(Function):
    '''
    返回索引基类，这种类是没有梯度的，因为返回的只是索引
    '''

    def fwd(self, x: NdArray, xp, axis):
        raise NotImplementedError("You must implement the fwd function in sub class.")

    def forward(self, x: NdArray, axis=None) -> NdArray:
        xp = get_array_module(x)
        return self.fwd(x, xp, axis)

    def backward(self, grad: NdArray) -> NdArray:
        return None


class ArgMax(_IndexSelect):
    def fwd(self, x: NdArray, xp, axis=None):
        return xp.argmax(x, axis=axis)


class ArgMin(_IndexSelect):
    def fwd(self, x: NdArray, xp, axis=None):
        return xp.argmin(x, axis=axis)


class Reshape(Function):
    def forward(self, x: NdArray, shape: Tuple) -> NdArray:
        self.save_for_backward(x.shape)
        return x.reshape(shape)

    def backward(self, grad: NdArray) -> NdArray:
        x_shape, = self.saved_tensors
        return grad.reshape(x_shape)


class ExpandDims(Function):
    def forward(self, x: NdArray, axis: int) -> NdArray:
        xp = get_array_module(x)
        self.save_for_backward(x.shape)
        return xp.expand_dims(x, axis)

    def backward(self, grad: NdArray) -> NdArray:
        x_shape, = self.saved_tensors
        return grad.reshape(x_shape)


class Transpose(Function):
    def forward(self, x: NdArray, axes) -> NdArray:
        self.save_for_backward(axes)
        return x.transpose(axes)

    def backward(self, grad: NdArray) -> NdArray:
        axes, = self.saved_tensors
        if axes is None:
            return grad.transpose()

        return grad.transpose(tuple(np.argsort(axes)))


Permute = Transpose


class Repeat(Function):
    def forward(self, x: NdArray, repeats) -> NdArray:
        xp = get_array_module(x)

        if isinstance(repeats, int):
            repeats = repeats,
        elif isinstance(repeats, Tuple):
            repeats = xp.array(repeats)

        self.save_for_backward(x, np.array(repeats), xp)
        return xp.tile(x, repeats)

    def backward(self, grad: NdArray) -> Any:
        x, repeats, xp = self.saved_tensors
        # 应该要扩展的维度数
        num_unsqueezed = grad.ndim - x.ndim

        for _ in range(num_unsqueezed):
            grad = grad.sum(0, keepdims=False)

        if repeats.ndim == 0:
            # 将repeats的维度进行扩展
            repeats = xp.expand_dims(repeats, 0)

        for dim, repeat in enumerate(repeats[num_unsqueezed:]):
            if repeat == 1:
                continue

            grad = sum(xp.array_split(grad, repeat, dim))

        return grad


def install_ops():
    Tensor.__add__ = add
    Tensor.__radd__ = add
    Tensor.__sub__ = sub
    Tensor.__rsub__ = rsub
    Tensor.__mul__ = mul
    Tensor.__rmul__ = mul
    Tensor.__truediv__ = div
    Tensor.__rtruediv__ = rdiv
