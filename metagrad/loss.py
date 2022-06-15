from metagrad import functions as F
from metagrad.module import Module
from metagrad.tensor import Tensor


class _Loss(Module):
    '''
    损失的基类
    '''
    reduction: str  # none | mean | sum

    def __init__(self, reduction: str = "mean") -> None:
        super(_Loss, self).__init__()
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
            loss = errors.mean()
        elif self.reduction == "sum":
            loss = errors.sum()
        else:
            loss = errors

        return loss


class BCELoss(_Loss):
    '''
    像torch BCEWithLogitsLoss
    '''
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__(reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        '''
        :param input: logits
        :param target:  真实标签 0或1
        :return:
        '''
        return F.binary_cross_entropy(input, target, self.reduction)


class CrossEntropyLoss(_Loss):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__(reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
         '''
         :param input: logits
         :param target: 真实标签one-hot向量
         :return:
         '''
         return F.cross_entropy(input, target, self.reduction)


class NLLLoss(_Loss):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__(reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        '''
        :param input: 对数概率 即 log_softmax
        :param target: 类别索引 或 one-hot向量
        :return:
        '''
        return F.nll_loss(input, target, self.reduction)