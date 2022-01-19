from metagrad.tensor import Tensor


def logsumexp(x, axis=-1):
    b = x.max(axis=axis, keepdims=True)
    return b + (x - b).exp().sum(axis=axis, keepdims=True).log()


def sigmoid(x: Tensor) -> Tensor:
    return 1 / (1 + (-x).exp())


def softmax(x, axis=-1):
    b = x.max(axis=axis, keepdims=True)
    y = (x - b).exp()
    return y / y.sum(axis=axis, keepdims=True)


def log_softmax(x, axis=-1):
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

    N = len(target)

    if reduction == "mean":
        loss = errors.sum() / N
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

    N = len(target)
    axis = -1

    errors = target.sum(axis=axis, keepdims=True) * logsumexp(input, axis=axis) - (target * input).sum(axis=axis,
                                                                                                       keepdims=True)

    if reduction == "mean":
        loss = errors.sum() / N
    elif reduction == "sum":
        loss = errors.sum()
    else:
        loss = errors
    return loss
