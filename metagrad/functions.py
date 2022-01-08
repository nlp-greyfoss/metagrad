from metagrad.tensor import Tensor


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
