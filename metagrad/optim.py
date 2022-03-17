from typing import List

from metagrad.paramater import Parameter
from metagrad.tensor import no_grad


class Optimizer:
    def __init__(self, params, defaults) -> None:
        '''

        :param params: Tensor序列或字典
        :param defaults:
        '''
        self.defaults = defaults
        self.params = params
        # 参数分组，比如分为
        param_groups = list(params)

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    '''
    随机梯度下降
    '''

    def __init__(self, params: List[Parameter], lr: float = 1e-3) -> None:
        super().__init__(params)
        self.lr = lr

    def step(self) -> None:
        with no_grad():
            for p in self.params:
                p -= p.grad * self.lr
