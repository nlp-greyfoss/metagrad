from metagrad.module import Module
from metagrad.tensor import Tensor


class _Loss(Module):
    '''
    损失的基类
    '''
    reduction: str  # none | mean | sum

    def __init__(self, reduction: str = "mean") -> None:
        self.reduction = reduction


class MSELoss(_Loss):
    def __init__(self, reduction: str = "mean") -> None:
        '''
        均方误差
        '''
        super().__init__(reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        errors = (input - target) ** 2
        if self.reduction == "mean":
            loss = errors.sum(keepdims=False) / len(input)
        elif self.reduction == "sum":
            loss = errors.sum(keepdims=False)
        else:
            loss = errors

        return loss
