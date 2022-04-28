import numpy as np
from numpy import ndarray

from metagrad.ops import Function
from metagrad.tensor import Tensor


# ----激活函数----
class Relu(Function):
    '''
    实现relu激活函数
    '''

    def forward(ctx, x: ndarray) -> ndarray:
        ctx.save_for_backward(x)
        return np.maximum(x, 0)

    def backward(ctx, grad: ndarray) -> ndarray:
        x, = ctx.saved_tensors
        return grad * (x > 0)


class LeakyRelu(Function):
    def forward(ctx, x: ndarray, slope: float = 0.01) -> ndarray:
        ctx.save_for_backward(x, slope)
        return np.maximum(x, 0) + slope * np.minimum(x, 0)

    def backward(ctx, grad: ndarray) -> ndarray:
        x, slope = ctx.saved_tensors
        mask = np.array(x > 0).astype(grad.dtype)  # x > 0 : 1
        mask[mask <= 0] = slope  # x <=0 : slope
        return grad * mask


class ELU(Function):
    def forward(ctx, x: ndarray, alpha: float = 1) -> ndarray:
        ctx.save_for_backward(x, alpha)
        return np.maximum(x, 0) + np.minimum(0, alpha * (np.exp(x) - 1))

    def backward(ctx, grad: ndarray) -> ndarray:
        x, alpha = ctx.saved_tensors
        mask = np.array(x > 0).astype(grad.dtype)  # x > 0 : 1 加上np.array 兼容标量
        indices = (mask <= 0)
        mask[indices] = alpha * np.exp(x)[indices]  # x <= 0 :  alpha * exp(x)
        return grad * mask


def logsumexp(x: Tensor, axis=-1):
    b = x.max(axis=axis, keepdims=True)
    return b + (x - b).exp().sum(axis=axis, keepdims=True).log()


def sigmoid(x: Tensor) -> Tensor:
    return 1 / (1 + (-x).exp())


# def relu(x: Tensor) -> Tensor:
#     return x * (x > 0)
def relu(x: Tensor) -> Tensor:
    return Relu.apply(Relu, x)


# def leaky_relu(x: Tensor, slope: float = 0.1) -> Tensor:
#    return x * (x > 0) + slope * x * (x < 0)
def leaky_relu(x: Tensor, slope: float = 0.01) -> Tensor:
    return LeakyRelu.apply(LeakyRelu, x, slope=slope)


# def elu(x: Tensor, a: float = 1) -> Tensor:
#    return x * (x > 0) + a * (x.exp() - 1) * (x < 0)
def elu(x: Tensor, alpha: float = 1) -> Tensor:
    return ELU.apply(ELU, x, alpha=alpha)


def swish(x: Tensor) -> Tensor:
    return x * sigmoid(x)


def softplus(x: Tensor, beta: float = 1) -> Tensor:
    return (1 + (beta * x).exp()).log() / beta


def tanh(x: Tensor) -> Tensor:
    return 2 * sigmoid(2 * x) - 1


def softmax(x: Tensor, axis=-1):
    b = x.max(axis=axis, keepdims=True)
    y = (x - b).exp()
    return y / y.sum(axis=axis, keepdims=True)


def log_softmax(x: Tensor, axis=-1):
    '''
    :param x: logits
    :param axis:
    :return:
    '''
    return x - logsumexp(x, axis)


def binary_cross_entropy(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    '''

    :param input: logits
    :param target: 真实标签 0或1
    :param reduction:
    :return: binary cross entropy loss
    '''

    neg_abs = - abs(input)
    errors = input.clip(x_min=0) - input * target + (1 + neg_abs.exp()).log()

    if reduction == "mean":
        loss = errors.mean()
    elif reduction == "sum":
        loss = errors.sum()
    else:
        loss = errors
    return loss


def cross_entropy(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    '''

    :param input: logits
    :param target: 真实标签one-hot向量
    :param reduction:
    :return:
    '''

    axis = -1

    errors = target.sum(axis=axis, keepdims=True) * logsumexp(input, axis=axis) - (target * input).sum(axis=axis,
                                                                                                       keepdims=True)

    if reduction == "mean":
        loss = errors.mean()
    elif reduction == "sum":
        loss = errors.sum()
    else:
        loss = errors
    return loss


def dropout(input: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    """
    input: 输入
    p: dropout ratio 丢弃率
    training: 是否为训练阶段
    """
    if training:
        # 丢弃掩码 1代表保留，0代表丢弃 以1-p的概率生成输出为1伯努利分布，做了input的元素个数这么多次实验
        mask = np.random.binomial(1, 1 - p, size=input.shape)
        # 让输入乘上这个与之同shape的丢弃掩码，然后除以1-p进行缩放，这样在测试时，可以原样输出
        return input * Tensor(mask, requires_grad=False) / (1 - p)
    else:
        return input


class Embedding(Function):
    def forward(ctx, x: ndarray, weight: ndarray) -> ndarray:
        ctx.save_for_backward(x, weight.shape)
        return weight[x]

    def backward(ctx, grad: ndarray) -> ndarray:
        x, w_shape = ctx.saved_tensors

        bigger_grad = np.zeros(w_shape, dtype=grad.dtype)
        np.add.at(bigger_grad, x, grad)

        return bigger_grad


def embedding(x: Tensor, weight: Tensor) -> Tensor:
    return Embedding.apply(Embedding, x, weight)
