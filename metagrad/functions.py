from metagrad.tensor import Tensor


def sigmoid(x: Tensor) -> Tensor:
    return 1 / (1 + (-x).exp())


def softmax(x, axis=-1):
    y = x.exp()
    return y / y.sum(axis=axis, keepdims=True)


def binary_cross_entropy(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    errors = -(target * input.log() + (1 - target) * (1 - input).log())

    N = len(target)

    if reduction == "mean":
        loss = errors.sum() / N
    elif reduction == "sum":
        loss = errors.sum()
    else:
        loss = errors
    return loss


def cross_entropy(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    N = len(target)

    p = softmax(input)

    errors = - target * p.log()
    # errors = - p[np.arange(N), target.data].log()

    if reduction == "mean":
        loss = errors.sum() / N
    elif reduction == "sum":
        loss = errors.sum()
    else:
        loss = errors
    return loss
