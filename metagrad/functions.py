from typing import Tuple, Union, List

import numpy as np

from metagrad import cuda
from metagrad.cuda import get_array_module
from metagrad.ops import Function
from metagrad.tensor import Tensor, NdArray


# ----激活函数----
class ReLU(Function):
    def forward(self, x: NdArray) -> NdArray:
        xp = get_array_module(x)
        y = xp.maximum(x, 0, dtype=x.dtype)
        self.save_for_backward(y, xp)

        return y

    def backward(self, grad: NdArray) -> NdArray:
        y, xp = self.saved_tensors
        if xp is np:
            return grad * (y > 0)
        else:
            return cuda.elementwise(
                'T y, T gy', 'T gx',
                'gx = y > 0 ? gy : (T)0', 'relu_bwd')(y, grad)


# def relu(x: Tensor) -> Tensor:
#     return x * (x > 0)
def relu(x: Tensor) -> Tensor:
    return ReLU()(x)


class LeakyRelu(Function):
    def forward(self, x: NdArray, slope: float = 0.01) -> NdArray:
        self.save_for_backward(x, slope)
        xp = get_array_module(x)
        return xp.maximum(x, 0) + slope * xp.minimum(x, 0)

    def backward(self, grad: NdArray) -> NdArray:
        x, slope = self.saved_tensors
        mask = np.array(x > 0).astype(grad.dtype)  # x > 0 : 1
        mask[mask <= 0] = slope  # x <=0 : slope
        return grad * mask


class ELU(Function):
    def forward(self, x: NdArray, alpha: float = 1) -> NdArray:
        xp = get_array_module(x)
        self.save_for_backward(x, alpha, xp)
        return xp.maximum(x, 0) + xp.minimum(0, alpha * (xp.exp(x) - 1))

    def backward(self, grad: NdArray) -> NdArray:
        x, alpha, xp = self.saved_tensors
        mask = xp.array(x > 0).astype(grad.dtype)  # x > 0 : 1 加上np.array 兼容标量
        indices = (mask <= 0)
        mask[indices] = alpha * xp.exp(x)[indices]  # x <= 0 :  alpha * exp(x)
        return grad * mask


class Sigmoid(Function):
    def forward(self, x: NdArray) -> NdArray:
        xp = get_array_module(x)

        # assert xp is np

        if xp is np:
            half = x.dtype.type(0.5)
            y = np.tanh(x * half) * half + half
        else:
            y = cuda.elementwise(
                'T x', 'T y', 'y = tanh(x * 0.5) * 0.5 + 0.5',
                'sigmoid_fwd')(x)

        self.save_for_backward(y, xp)

        return y

    def backward(self, grad: NdArray) -> NdArray:
        y, xp = self.saved_tensors
        if xp is np:
            one = y.dtype.type(1)
            return grad * y * (one - y)
        else:
            return cuda.elementwise(
                'T y, T gy', 'T gx',
                'gx = gy * y * (1 - y)',
                'sigmoid_bwd')(y, grad)


def sigmoid(x: Tensor) -> Tensor:
    '''
        重写 return 1. / (1. + (-x).exp()) 加快速度

    Args:
        x:

    Returns:

    '''
    return Sigmoid()(x)


def logsigmoid(x: Tensor) -> Tensor:
    return sigmoid(x).log()


# def leaky_relu(x: Tensor, slope: float = 0.1) -> Tensor:
#    return x * (x > 0) + slope * x * (x < 0)
def leaky_relu(x: Tensor, slope: float = 0.01) -> Tensor:
    return LeakyRelu()(x, slope=slope)


# def elu(x: Tensor, a: float = 1) -> Tensor:
#    return x * (x > 0) + a * (x.exp() - 1) * (x < 0)
def elu(x: Tensor, alpha: float = 1) -> Tensor:
    return ELU()(x, alpha=alpha)


def swish(x: Tensor) -> Tensor:
    return x * sigmoid(x)


def softplus(x: Tensor, beta: float = 1) -> Tensor:
    return (1 + (beta * x).exp()).log() / beta


class Tanh(Function):
    def forward(self, x: NdArray) -> NdArray:
        xp = get_array_module(x)
        if xp is np:
            y = np.tanh(x)
        else:
            y = cuda.elementwise(
                'T x', 'T y', 'y = tanh(x)',
                'tanh_fwd')(x)

        self.save_for_backward(y, xp)

        return y

    def backward(self, grad: NdArray) -> NdArray:
        y, xp = self.saved_tensors
        if xp is np:
            one = y.dtype.type(1)
            return grad * (one - y * y)
        else:
            return cuda.elementwise(
                'T y, T gy', 'T gx',
                'gx = gy *  (1 - y*y)',
                'tanh_bwd')(y, grad)


def tanh(x: Tensor) -> Tensor:
    return Tanh()(x)


class Softmax(Function):
    def __init__(self, axis=1):
        super().__init__()
        self.axis = axis

    def forward(self, x: NdArray) -> NdArray:
        xp = get_array_module(x)

        y = x - x.max(axis=self.axis, keepdims=True)
        xp.exp(y, out=y)
        y /= y.sum(axis=self.axis, keepdims=True)

        self.save_for_backward(y, xp)

        return y

    def backward(self, grad: NdArray) -> NdArray:
        y, xp = self.saved_tensors

        gx = y * grad
        dx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * dx

        return gx


def softmax(x: Tensor, axis=-1):
    # b = x.max(axis=axis, keepdims=True)
    # y = (x - b).exp()
    # return y / y.sum(axis=axis, keepdims=True)

    return Softmax(axis=axis)(x)


def _logsumexp(x: NdArray, axis=-1):
    xp = get_array_module(x)
    b = x.max(axis=axis, keepdims=True)
    y = x - b
    xp.exp(y, out=y)
    s = y.sum(axis=axis, keepdims=True)
    xp.log(s, out=s)
    b += s
    return b


class LogSoftmax(Function):

    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def forward(self, x: NdArray) -> NdArray:
        xp = get_array_module(x)

        y = x - _logsumexp(x, self.axis)
        self.save_for_backward(y, xp)

        return y

    def backward(self, grad: NdArray) -> NdArray:
        y, xp = self.saved_tensors
        return grad - xp.exp(y) * grad.sum(axis=self.axis, keepdims=True)


def log_softmax(x: Tensor, axis=-1):
    '''
    :param x: logits
    :param axis:
    :return:
    '''
    return LogSoftmax(axis=axis)(x)


def _reduction(errors: Tensor, method: str) -> Tensor:
    if method == "mean":
        loss = errors.sum() / errors.shape[0]
    elif method == "sum":
        loss = errors.sum()
    else:
        loss = errors

    return loss


def _softmax(x, axis=1):
    b = x.max(axis=axis, keepdims=True)
    y = (x - b).exp()
    return y / y.sum(axis=axis, keepdims=True)


class NLLLoss(Function):
    def __init__(self, ignore_index=-100, reduction: str = "mean"):
        """

        Args:
            ignore_index: 忽略的标签，可以是padding的标签(一般为0)，默认为-100
            reduction:
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target) -> NdArray:
        """
        Args:
            input: 对数概率 即 log_softmax 形状 (batch_size, num_classes)
            target:  类别索引 或 one-hot向量 形状为 (batch_size,) 或 (batch_size, num_classes)

        Returns:
        """
        xp = get_array_module(input)

        # 如果target是ont-hot向量，转换为一维向量
        if target.ndim > 1:
            target = xp.argmax(target, axis=1)

        batch_size, num_classes = input.shape
        # 根据ignore_index对标签进行忽略
        mask = (target != self.ignore_index).astype(int)

        errors = -xp.sum(input[xp.arange(batch_size), target] * mask, dtype=input.dtype)
        if self.reduction == 'mean':
            errors = xp.divide(errors, mask.sum(), dtype=input.dtype)

        self.save_for_backward(xp, target, input, batch_size, num_classes, mask)
        return errors

    def backward(self, grad: NdArray) -> NdArray:
        xp, target, input, batch_size, num_classes, mask = self.saved_tensors

        if target.ndim > 1:
            target = xp.argmax(target, axis=1)

        bigger_grad = xp.zeros((batch_size, num_classes), dtype=grad.dtype)
        bigger_grad[xp.arange(batch_size), target] = xp.divide(-mask, mask.sum(), dtype=input.dtype)

        return bigger_grad


def nll_loss(input: Tensor, target: Tensor, reduction: str = "mean", ignore_index=-100):
    return NLLLoss(ignore_index, reduction)(input, target)


def binary_cross_entropy(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    '''

    :param input: logits
    :param target: 真实标签 0或1
    :param reduction:
    :return: binary cross entropy loss
    '''

    neg_abs = - abs(input)
    errors = input.clip(x_min=0) - input * target + (1 + neg_abs.exp()).log()

    return _reduction(errors, reduction)


def cross_entropy(input: Tensor, target: Tensor, reduction: str = "mean", ignore_index=-100) -> Tensor:
    '''

    :param input: logits
    :param target: 真实标签one-hot向量 或 类别索引
    :param reduction:
    :return:
    '''
    # 先计算logsoftmax
    log_y = log_softmax(input)
    # 基于nll实现交叉熵损失
    return nll_loss(log_y, target, reduction, ignore_index)


class Dropout(Function):

    def __init__(self, p: float = 0.5):
        '''
        丢弃掩码 1代表保留，0代表丢弃 以1-p的概率生成输出为1伯努利分布，做了input的元素个数这么多次实验

        Args:
            p: dropout ratio 丢弃率
        '''
        super().__init__()
        self.p = p

    def forward(self, x: NdArray) -> NdArray:
        xp = get_array_module(x)

        if xp is np:
            scale = x.dtype.type(1. / 1 - self.p)
            flag = np.random.rand(*x.shape) >= self.p
            mask = scale * flag
            # 让输入乘上这个与之同shape的flag，然后除以1-p进行缩放，这样在测试时，可以原样输出
            y = x * mask
        else:
            rand = xp.random.rand(*x.shape, dtype=np.float32)
            scale = x.dtype.type(1. / (1 - self.p))
            mask, y = cuda.elementwise(
                'T x, R r, T scale, T p', 'T mask, T y',
                '''
                mask = (r >= p) * scale;
                y = x * mask;
                ''',
                'dropout_fwd',
            )(x, rand, scale, self.p)

        self.save_for_backward(mask)

        return y

    def backward(self, grad: NdArray) -> NdArray:
        mask, = self.saved_tensors
        return grad * mask


def dropout(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    """
    x: 输入
    p: dropout ratio 丢弃率
    training: 是否为训练阶段
    """
    if training:
        return Dropout(p=p)(x)
    else:
        return x


class Embedding(Function):
    def forward(self, weight: NdArray, indices: NdArray) -> NdArray:
        self.save_for_backward(weight.shape, indices)
        return weight[indices]

    def backward(self, grad: NdArray) -> Tuple[NdArray, None]:
        w_shape, indices = self.saved_tensors

        xp = get_array_module(grad)

        bigger_grad = xp.zeros(w_shape, dtype=grad.dtype)

        if xp is np:
            np.add.at(bigger_grad, indices, grad)
        else:
            bigger_grad.scatter_add(indices, grad)

        # 因为它有两个输入，防止错误地拆开bigger_grad
        # indices 不需要梯度
        return bigger_grad, None


def embedding(weight: Tensor, indices: Tensor) -> Tensor:
    return Embedding()(weight, indices)


class MaskedSelect(Function):
    def forward(self, x: NdArray, mask: NdArray) -> NdArray:
        self.save_for_backward(x.shape, mask)
        return x[mask]

    def backward(self, grad: NdArray) -> NdArray:
        x_shape, mask = self.saved_tensors
        xp = get_array_module(grad)

        bigger_grad = xp.zeros(x_shape, dtype=grad.dtype)

        bigger_grad[mask] = grad
        return bigger_grad


def masked_select(x: Tensor, mask):
    return MaskedSelect()(x, mask)


class Split(Function):
    '''Stack的逆操作'''

    def forward(self, inputs: NdArray, axis: int) -> NdArray:
        xp = get_array_module(inputs)
        xs = xp.split(inputs, inputs.shape[axis], axis)
        ys = [xp.squeeze(y, axis) for y in xs]  # 去掉维度axis
        self.save_for_backward(xp, axis, ys[0].shape, inputs.dtype)

        return tuple(ys)

    def backward(self, *grad: List[NdArray]) -> NdArray:
        xp, axis, shape, dtype = self.saved_tensors
        grads = [xp.zeros(shape, dtype) if g is None else g for g in grad]
        return xp.stack(grads, axis=axis)


def split(x: Tensor, axis: int = 0):
    return Split()(x, axis=axis)


unbind = split


class Stack(Function):
    '''
    在指定维度上进行堆叠，会增加维度
    维数：指有多少维
    维度：某个维的元素个数

    比如(2,3)的维数是2；第1个维度是2；第2个维度是3

    '''

    def forward(self, *inputs: Union[Tuple[NdArray, ...], List[NdArray]], axis: int) -> NdArray:
        xp = get_array_module(inputs[0])
        ret = xp.stack(inputs, axis=axis)
        self.save_for_backward(axis, xp)
        return ret

    def backward(self, grad: NdArray) -> NdArray:
        axis, xp = self.saved_tensors

        grads = xp.split(grad, grad.shape[axis], axis)
        grads = [xp.squeeze(g, axis) for g in grads]  # 去掉维度axis
        return tuple(grads)


def stack(xs: Union[Tuple[Tensor, ...], List[Tensor]], axis: int = 0):
    return Stack()(*xs, axis=axis)


class Cat(Function):
    '''
    在原有某一维度进行拼接，拼接的结果是Tensor的总维数不变，其中用于拼接的那一维度等于各分量维度之和
    '''

    def forward(self, *inputs: Union[Tuple[Tensor, ...], List[Tensor]], axis: int = -1) -> NdArray:
        xp = get_array_module(inputs[0])
        self.save_for_backward(inputs, axis, xp)
        return xp.concatenate(inputs, axis)

    def backward(self, grad: NdArray) -> NdArray:
        inputs, axis, xp = self.saved_tensors
        if len(inputs) == 1:
            return grad

        # 可能会不均分，所以大小可能不一致
        sizes = np.array(
            [x.shape[axis] for x in inputs[:-1]]
        ).cumsum()  # 计算累积和

        return tuple(xp.array_split(grad, sizes, axis))


def cat(xs: Union[Tuple[Tensor, ...], List[Tensor]], axis: int = 0):
    return Cat()(*xs, axis=axis)


class Chunk(Function):
    '''
    cat的逆操作，将Tensor沿某一维分开，chunks为分割的份数，axis为分割的维度
    '''

    def forward(self, inputs: NdArray, chunks: Union[int, NdArray], axis: int) -> Tuple[NdArray]:
        xp = get_array_module(inputs)
        ret = xp.array_split(inputs, chunks, axis)
        shapes = [x.shape for x in ret]
        self.save_for_backward(xp, axis, shapes, inputs.dtype)

        return tuple(ret)

    def backward(self, *grad: List[NdArray]) -> NdArray:
        xp, axis, shapes, dtype = self.saved_tensors
        grads = [xp.zeros(shape, dtype=dtype) if g is None else g for g, shape in zip(grad, shapes)]
        return xp.concatenate(grads, axis)


def chunk(input: Tensor, chunks: int, axis=0):
    return Chunk()(input, chunks=chunks, axis=axis)


class Flip(Function):
    def forward(self, inputs: NdArray, axis: Union[int, Tuple] = None) -> NdArray:
        xp = get_array_module(inputs)
        self.save_for_backward(axis, xp)
        return xp.flip(inputs, axis=axis)

    def backward(self, grad: NdArray) -> NdArray:
        axis, xp = self.saved_tensors
        return xp.flip(grad, axis=axis)


def flip(x: Tensor, axis: Union[int, Tuple] = None):
    return Flip()(x, axis=axis)


class Bmm(Function):
    def forward(self, x: NdArray, y: NdArray) -> NdArray:
        self.save_for_backward(x, y)
        return x @ y

    def backward(self, grad: NdArray) -> Tuple[NdArray, NdArray]:
        x, y = self.saved_tensors
        return grad @ y.swapaxes(-2, -1), x.swapaxes(-2, -1) @ grad


def bmm(x: Tensor, y: Tensor):
    return Bmm()(x, y)


# 简单的norm实现
def norm(input: Tensor, p: int = 2, axis=None, keepdims=False):
    assert p in (1, 2), "Only support L2 normalization(p=2) and L1 normalization(p=1)"
    if p == 1:
        return abs(input).sum(axis=axis, keepdims=keepdims)
    else:
        return ((input ** 2).sum(axis=axis, keepdims=keepdims)) ** (1 / 2)


def cos_sim(u: Tensor, v: Tensor, axis=1):
    print(u.shape, v.shape)

    u_norm = norm(u, axis=axis)
    v_norm = norm(v, axis=axis)

    print(u_norm.shape, v_norm.shape)

    return u_norm @ v_norm.T
    #
    # fz = (u * v)
    # print(f'shape:{fz.shape}')
    # print(f'u_norm:{(u_norm * v_norm).shape}')
    # return (u / u_norm) * (v / v_norm)


"""
RNN 相关
"""


def linear(input, weight, bias=None):
    '''
    简单的带偏置的线性运算
    Args:
        input:
        weight:
        bias:

    Returns:

    '''
    output = input @ weight.T
    if bias is not None:
        output += bias

    return output


def RNNReLUCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    hy = relu(linear(input, w_ih, b_ih) + linear(hidden, w_hh, b_hh))
    return hy


def RNNTanhCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    # tanh(input * w_ih + b_ih + hidden * w_hh + b_hh)
    hy = tanh(linear(input, w_ih, b_ih) + linear(hidden, w_hh, b_hh))
    return hy


def LSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    hx, cx = hidden
    ifgo = linear(input, w_ih, b_ih) + linear(hx, w_hh, b_hh)
    # 一次性计算三个门与g_t
    i, f, g, o = chunk(ifgo, 4, 1)

    i = sigmoid(i)
    f = sigmoid(f)
    g = tanh(g)
    o = sigmoid(o)

    cy = (f * cx) + (i * g)
    hy = o * tanh(cy)

    return hy, cy


def GRUCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    gi = linear(input, w_ih, b_ih)
    gh = linear(hidden, w_hh, b_hh)

    i_r, i_z, i_g = chunk(gi, 3, 1)
    h_r, h_z, h_g = chunk(gh, 3, 1)

    r = sigmoid(i_r + h_r)  # 重置门
    i = sigmoid(i_z + h_z)  # 更新门

    n_g = tanh(i_g + r * h_g)  # 候选状态
    hy = n_g + i * (hidden - n_g)
    return hy


def StackedRNN(inners, num_layers, lstm=False, dropout_ratio=0, train=True):
    '''

    Args:
        inners: RNN Recurrent，双向的话就是一个元组
        num_layers: 层数
        lstm: 是否为lstm
        dropout_ratio: dropout 比率
        train: 是否为训练阶段

    Returns:

    '''
    num_directions = len(inners)  # 长度就是方向数
    total_layers = num_layers * num_directions  # 带方向的“总层数”为 传入的层数 乘 方向数

    def forward(input, hidden, weight):
        """
        StackedRNN的真正执行函数
        Args:
            input:  输入大小始终为 (seq_len, batch_size, input_size)
            hidden: 隐藏状态（num_layers * num_directions, batch_size, hidden_size)
            weight: all_weights

        Returns:

        """
        assert len(weight) == total_layers  # 确保总层数等于all_weights长度
        next_hidden = []  # 保存每层RNN最后的输出
        if lstm:
            # 对(h, c)第0个维度，即total_layers，上打包成大小为2的元组并转换为元组列表，列表大小为total_layers，元组中元素大小为(batch_size, hidden_size)
            hidden = list(zip(*hidden))

        for i in range(num_layers):  # 一层一层计算，由底向上
            all_output = []  # 保存所有方向上每个时间步的输出
            # 如果是双向，相当于多了一层
            for j, inner in enumerate(inners):
                l = i * num_directions + j  # 层数
                # 在输入input上应用rnn，传入每层的隐藏状态，以及参数
                # hy:
                #   LSTM ((batch_size, hidden_size), (batch_size, hidden_size))
                #   非LSTM (batch_size, hidden_size)
                # output (seq_len, batch_size, hidden_size))
                hy, output = inner(input, hidden[l], weight[l])
                next_hidden.append(hy)
                all_output.append(output)
            # 在最后一个维度上拼接all_output，作为下一层(如果有的话)的输入。 双向的情况下会使得最后一个维度数乘2
            input = cat(all_output, input.ndim - 1)
            # 多层之间进行dropout
            if dropout_ratio != 0 and i < num_layers - 1:  # 如果不是最后一层，且dropout不为零
                input = dropout(input, p=dropout_ratio, training=train)

        if lstm:
            next_h, next_c = zip(*next_hidden)
            # 根据隐藏状态和单元状态，分别拼接
            next_hidden = (
                cat(next_h, 0).view(total_layers, *next_h[0].shape),
                cat(next_c, 0).view(total_layers, *next_c[0].shape)
            )
        else:
            # 拼接next_hidden列表，转换为大小为(total_layers, batch_size, hidden_size)
            next_hidden = cat(next_hidden, 0).view(
                total_layers, *next_hidden[0].shape
            )

        return next_hidden, input

    return forward


def Recurrent(inner, reverse=False):
    """

    Args:
        inner: RNN单元具体计算
        reverse: 是否为反向

    Returns:

    """

    def forward(input, hidden, weight):
        """
        单层计算：按照时间步从左到右(正向)，或者从右到左(反向)
        Args:
            input: 单层的输入 大小 (seq_len, batch_size, hidden_size)
            hidden: (batch_size, hidden_size)
            weight:  list[w_ih, w_hh, b_ih, b_hh]

        Returns:

        """
        # 保存每个时间步隐藏状态的输出
        output = []
        # 得到步数索引 正向range(seq_len): [0,1,2,3,4] ; 逆向 range(seq_len-1, -1,-1): [4,3,2,1,0]
        steps = range(input.size(0) - 1, -1, -1) if reverse else range(input.size(0))
        # 通过循环处理每步，下一步的执行依赖上一步的结果
        for i in steps:
            # 根据当前时间步输入input[i]、上一步隐藏状态hidden，以及权重(按w_ih, w_hh, b_ih, b_hh拆包) 进行具体的RNN计算
            hidden = inner(input[i], hidden, *weight)
            # 如果tuple就是LSTM，即包含隐藏状态和单元状态，output只需要隐藏状态的输出
            output.append(hidden[0] if isinstance(hidden, tuple) else hidden)
        # 如果是逆序的，也要将输出逆序。在逆序情况下，output先保存的是最后一个时间步的结果，output逆序后，才能使得时间步位置对应上
        if reverse:
            output.reverse()
        # output原本是一个python列表，调用cat在维度0上间拼接，并转换成一个Tensor，大小为(seq_len * batch_size, hidden_size)
        # 接着调用reshape，变成大小(seq_len, batch_size, hidden_size)
        output = cat(output, 0).view(input.size(0), *output[0].shape)
        # 返回最后一个时间步的隐藏状态和输出(所有时间步的隐藏状态)
        return hidden, output

    return forward


def VariableRecurrentReverse(batch_sizes, inner):
    def forward(input, hidden, weight):
        """

        :param input:  单层的输入 大小 PackedSequence (total_seq_len, embed_size) 输入的顺序不变，通过offset从后往前取
        :param hidden:  (batch_size, hidden_size)
        :param weight:  list[w_ih, w_hh, b_ih, b_hh]
        :return:
        """
        output = []
        input_offset = input.size(0) # 初始化为total_seq_len
        # 逆序先取最后一个批大小
        last_batch_size = batch_sizes[-1]
        initial_hidden = hidden
        flat_hidden = not isinstance(hidden, tuple)
        if flat_hidden:
            hidden = (hidden,)
            initial_hidden = (initial_hidden,)
        # 前batch_sizes[-1]个
        hidden = tuple(h[:batch_sizes[-1]] for h in hidden)
        for batch_size in reversed(batch_sizes): # 以逆序的形式遍历batch_sizes
            inc = batch_size - last_batch_size # 当前时间步增加的批大小数
            if inc > 0:
                # 拼接上一个时间步hidden和initial_hidden中的inc个到hidden中
                hidden = tuple(cat((h, ih[last_batch_size:batch_size]), 0)
                               for h, ih in zip(hidden, initial_hidden))

            last_batch_size = batch_size
            # 从后往前取
            step_input = input[input_offset - batch_size:input_offset]
            input_offset -= batch_size

            if flat_hidden:
                hidden = (inner(step_input, hidden[0], *weight), )
            else:
                hidden = inner(step_input, hidden, *weight)
            # 保存每个时间步输出的隐藏状态
            output.append(hidden[0])
        # 按逆序的方式计算的隐藏状态，因此需要进行逆序，使得输入和隐藏状态的位置匹配
        output.reverse()
        output = cat(output, 0)
        if flat_hidden:
            hidden = hidden[0]
        # hidden (batch_size, hidden_size) 直接使用第0个时间步的隐藏状态，在逆序时，就是最后一个时间步的隐藏状态
        # output (total_seq_len, hidden_size)
        return hidden, output

    return forward


def VariableRecurrent(batch_sizes, inner):
    """
    可变长度(PackedSequence)RNN的正向计算过程
    :param batch_sizes: 每个时间步的批大小
    :param inner: 具体的RNN
    :return:
    """

    def forward(input, hidden, weight):
        """
        :param input:  单层的输入 大小 PackedSequence (total_seq_len, embed_size)
        :param hidden:  (batch_size, hidden_size)
        :param weight:  list[w_ih, w_hh, b_ih, b_hh]
        :return:
        """
        output = []  # 保存每个时间步的隐藏状态
        input_offset = 0  # 输入的偏移量
        last_batch_size = batch_sizes[0]  # 最近的batch_size 初始化
        hiddens = []
        # 如果hidden不为tuple，即非LSTM
        flat_hidden = not isinstance(hidden, tuple)
        if flat_hidden:
            # hidden 转换成 tuple
            hidden = (hidden,)

        # 从batch_sizes遍历所有的batch_size
        for batch_size in batch_sizes:
            # 得到当前时间步的输入，偏移量开始到batch_size个输入
            step_input = input[input_offset:input_offset + batch_size]
            # 更新偏移量
            input_offset += batch_size
            # 上一个batch_size 减去 当前batch_size，剩下dec个
            dec = last_batch_size - batch_size
            if dec > 0:
                # h[-dec:]从隐藏状态h的末尾截取dec长度的子序列
                hiddens.append(tuple(h[-dec:] for h in hidden))
                # h[:-dec]截取和当前batch_size相同的长度
                hidden = tuple(h[:-dec] for h in hidden)
            # 更新最近的batch_size
            last_batch_size = batch_size

            if flat_hidden:
                # 如果是非LSTM，则取hidden[0]，实际的RNN计算，注意：并且把返回结果转换为一个元组
                hidden = (inner(step_input, hidden[0], *weight),)
            else:
                # 否则直接传入
                hidden = inner(step_input, hidden, *weight)
            # 保存所有的hidden：非LSTM就是真实结果；LSTM即hidden，非cell
            output.append(hidden[0])
        # 保存最后的时间步计算出来的有效hidden
        hiddens.append(hidden)
        # hiddens列表中元素的顺序是从最长序列到最短序列
        # 将hiddens列表中元素的顺序反转，变成从最短序列到最长序列
        # 与后续处理步骤相匹配
        hiddens.reverse()
        # 拼接hiddens
        hidden = tuple(cat(h, 0) for h in zip(*hiddens))
        # hidden[0]是得到的与最长序列长度相同的隐藏状态
        assert hidden[0].size(0) == batch_sizes[0]

        if flat_hidden:
            hidden = hidden[0]
        # 拼接output，变成 (total_seq_len, hidden_size)
        output = cat(output, 0)
        # hidden (batch_size, hidden_size)
        return hidden, output

    return forward


def variable_recurrent_factory(batch_sizes):
    def fac(inner, reverse=False):
        # 如果是逆序
        if reverse:
            return VariableRecurrentReverse(batch_sizes, inner)
        else:
            return VariableRecurrent(batch_sizes, inner)

    return fac


def AutogradRNN(mode, num_layers=1, batch_first=False, dropout=0, train=True, bidirectional=False, batch_sizes=None):
    if mode == 'RNN_RELU':
        cell = RNNReLUCell
    elif mode == 'RNN_TANH':
        cell = RNNTanhCell
    elif mode == 'LSTM':
        cell = LSTMCell
    else:
        cell = GRUCell

    if batch_sizes is None:
        # 非压缩模式
        rec_factory = Recurrent
    else:
        # 压缩模式，通过 rnn 工厂方法返回具体的函数
        rec_factory = variable_recurrent_factory(batch_sizes)

    if bidirectional:
        layer = (rec_factory(cell), rec_factory(cell, reverse=True))
    else:
        layer = (rec_factory(cell),)
    # 堆叠RNN
    func = StackedRNN(layer,
                      num_layers,
                      (mode == 'LSTM'),
                      dropout_ratio=dropout,
                      train=train)

    def forward(input, weight, hidden):
        """

        Args:
            input:  输入 (batch_size, seq_len, input_size) 如果是batch_first；否则 (seq_len, batch_size, input_size)
            weight: all_weights
            hidden: 隐藏状态（num_layers * num_directions, batch_size, hidden_size)

        Returns:

        """
        # 确保输入(seq_len, batch_size, input_size)的形式，方便后续通过input[timestep]的方式访问每个时间步的输入
        if batch_first and batch_sizes is None:
            input = input.permute(1, 0, 2)
        # 在输入input上应用RNN，返回(可能包含多层和双向)结果
        # nexth
        #   LSTM 包含隐藏状态和单元状态 ((num_layers * num_directions, batch_size, hidden_size), (num_layers * num_directions, batch_size, hidden_size))
        #   非LSTM 隐藏状态 (num_layers * num_directions, batch_size, hidden_size)
        # output (seq_len, batch_size, num_directions * hidden_size)
        nexth, output = func(input, hidden, weight)
        # 确保output的形状符合batch_first参数定义
        if batch_first and batch_sizes is None:
            output = output.permute(1, 0, 2)

        return output, nexth

    return forward


def RNN(*args, **kwargs):
    """
    Args:
        *args: mode
        **kwargs:
            num_layers
            batch_first
            dropout
            train
            bidirectional
    Returns:

    """

    def forward(input, *fargs, **fkwargs):
        """
        Args:
            input: 输入 (batch_size, seq_len, input_size) 如果是batch_first；否则 (seq_len, batch_size, input_size)
            *fargs: (all_weights, hx)
                all_weights:  所有的参数，包含weight_ih_lxx, weight_hh_lxx, (bias_ih_lxx,  bias_hh_lxx 如果有bias的话) 的参数值。
                    其中第一个x表示层数；第二个x表示为是否逆向参数。 all_weights保存的是这些参数值
                hx: 隐藏状态（num_layers * num_directions, batch_size, hidden_size)
            **fkwargs: None

        Returns:

        """
        func = AutogradRNN(*args, **kwargs)

        return func(input, *fargs, **fkwargs)

    return forward
