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


def softmax(x: Tensor, axis=-1):
    b = x.max(axis=axis, keepdims=True)
    y = (x - b).exp()
    return y / y.sum(axis=axis, keepdims=True)


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


def nll_loss(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    '''
    负对数似然损失
    :param input: 对数概率 即 log_softmax
    :param target:  类别索引 或 one-hot向量
    :param reduction:
    :return:
    '''
    # 如果target是ont-hot向量
    if input.ndim == target.ndim and input.shape == target.shape:
        errors = - target * input
    else:
        xp = input.xp
        # 如果target是类别索引
        errors = -input[xp.arange(target.shape[0]), target.array()]

    return _reduction(errors, reduction)


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


def cross_entropy(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    '''

    :param input: logits
    :param target: 真实标签one-hot向量 或 类别索引
    :param reduction:
    :return:
    '''
    # 先计算logsoftmax
    log_y = log_softmax(input)
    # 基于nll实现交叉熵损失
    return nll_loss(log_y, target, reduction)


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
    output = input @ weight.T
    if bias is not None:
        output += bias

    return output


def RNNReLUCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    hy = relu(linear(input, w_ih, b_ih) + linear(hidden, w_hh, b_hh))
    return hy


def RNNTanhCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
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
    num_directions = len(inners)
    total_layers = num_layers * num_directions

    def forward(input, hidden, weight):
        assert len(weight) == total_layers
        next_hidden = []
        if lstm:
            hidden = list(zip(*hidden))

        for i in range(num_layers):
            all_output = []
            for j, inner in enumerate(inners):
                l = i * num_directions + j  # 层数

                hy, output = inner(input, hidden[l], weight[l])
                next_hidden.append(hy)
                all_output.append(output)

            input = cat(all_output, input.ndim - 1)

            if dropout_ratio != 0 and i < num_layers - 1:
                input = dropout(input, p=dropout_ratio, training=train)

        if lstm:
            next_h, next_c = zip(*next_hidden)
            next_hidden = (
                cat(next_h, 0).view(total_layers, *next_h[0].shape),
                cat(next_c, 0).view(total_layers, *next_c[0].shape)
            )
        else:
            next_hidden = cat(next_hidden, 0).view(
                total_layers, *next_hidden[0].shape
            )

        return next_hidden, input

    return forward


def VariableRecurrent(batch_sizes, inner):
    def forward(input, hidden, weight):
        output = []
        input_offset = 0
        last_batch_size = batch_sizes[0]
        hiddens = []
        flat_hidden = not isinstance(hidden, tuple)
        if flat_hidden:
            hiddens = (hidden,)

        for batch_size in batch_sizes:
            step_input = input[input_offset: input_offset + batch_size]
            input_offset += batch_size

            dec = last_batch_size - batch_size
            if dec > 0:
                hiddens.append(tuple(h[-dec:] for h in hidden))
                hidden = tuple(h[:-dec] for h in hidden)

            last_batch_size = batch_size

            if flat_hidden:
                hidden = (inner(step_input, hidden[0], *weight))
            else:
                hidden = inner(step_input, hidden, *weight)

            output.append(hidden[0])

        hiddens.append(hidden)
        hiddens.reverse()

        hidden = tuple(cat(h, 0) for h in zip(*hiddens))
        assert hidden[0].size(0) == batch_sizes[0]
        if flat_hidden:
            hidden = hidden[0]
        output = cat(output, 0)

        return hidden, output

    return forward


def VariableRecurrentReverse(batch_sizes, inner):
    def forward(input, hidden, weight):
        output = []
        input_offset = input.size(0)
        last_batch_size = batch_sizes[-1]
        initial_hidden = hidden
        flat_hidden = not isinstance(hidden, tuple)
        if flat_hidden:
            hidden = (hidden,)
            initial_hidden = (initial_hidden,)
        hidden = tuple(h[:batch_sizes[-1]] for h in hidden)
        for batch_size in reversed(batch_sizes):
            inc = batch_size - last_batch_size
            if inc > 0:
                hidden = tuple(cat((h, ih[last_batch_size:batch_size]), 0)
                               for h, ih in zip(hidden, initial_hidden))
            last_batch_size = batch_size
            step_input = input[input_offset - batch_size:input_offset]
            input_offset -= batch_size

            if flat_hidden:
                hidden = (inner(step_input, hidden[0], *weight),)
            else:
                hidden = inner(step_input, hidden, *weight)
            output.append(hidden[0])

        output.reverse()
        output = cat(output, 0)
        if flat_hidden:
            hidden = hidden[0]
        return hidden, output

    return forward


def variable_recurrent_factory(batch_sizes):
    def fac(inner, reverse=False):
        if reverse:
            return VariableRecurrentReverse(batch_sizes, inner)
        else:
            return VariableRecurrent(batch_sizes, inner)

    return fac


def Recurrent(inner, reverse=False):
    def forward(input, hidden, weight):
        output = []
        steps = range(input.size(0) - 1, -1, -1) if reverse else range(input.size(0))

        for i in steps:
            hidden = inner(input[i], hidden, *weight)
            output.append(hidden[0] if isinstance(hidden, tuple) else hidden)

        if reverse:
            output.reverse()

        output = cat(output, 0).view(input.size(0), *output[0].shape)

        return hidden, output

    return forward


def AutogradRNN(mode, input_size, hidden_size, num_layers=1, batch_first=False,
                dropout=0, train=True, bidirectional=False, batch_sizes=None,
                dropout_state=None):
    if mode == 'RNN_RELU':
        cell = RNNReLUCell
    elif mode == 'RNN_TANH':
        cell = RNNTanhCell
    elif mode == 'LSTM':
        cell = LSTMCell
    else:
        cell = GRUCell

    if batch_sizes is None:
        rec_factory = Recurrent
    else:
        rec_factory = variable_recurrent_factory(batch_sizes)

    if bidirectional:
        layer = (rec_factory(cell), rec_factory(cell, reverse=True))
    else:
        layer = (rec_factory(cell),)

    func = StackedRNN(layer,
                      num_layers,
                      (mode == 'LSTM'),
                      dropout_ratio=dropout,
                      train=train)

    def forward(input, weight, hidden):
        if batch_first and batch_sizes is None:
            input = input.transpose((1, 0, 2))

        nexth, output = func(input, hidden, weight)

        if batch_first and batch_sizes is None:
            output = output.transpose((1, 0, 2))

        return output, nexth

    return forward


def RNN(*args, **kwargs):
    def forward(input, *fargs, **fkwargs):
        func = AutogradRNN(*args, **kwargs)

        return func(input, *fargs, **fkwargs)

    return forward
