from typing import List

from metagrad.paramater import Parameter


class Optimizer:
    def __init__(self, params: List[Parameter]) -> None:
        self.params = params

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
        for p in self.params:
            p -= p.grad * self.lr
