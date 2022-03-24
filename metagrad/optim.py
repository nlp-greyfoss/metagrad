from metagrad.paramater import Parameter
from metagrad.tensor import no_grad


class Optimizer:
    def __init__(self, params, defaults) -> None:
        '''

        :param params: Tensor列表或字典
        :param defaults: 包含优化器默认值的字典
        '''

        self.defaults = defaults

        # 参数分组，比如分为
        # 需要正则化的参数和不需要的
        # 需要更新的参数和不需要的
        self.param_groups = []
        param_groups = list(params)

        # 如果不是字典
        if not isinstance(param_groups[0], dict):
            # 就把它转换为字典列表
            param_groups = [{'params': param_groups}]

        for param_group in param_groups:
            self.add_param_group(param_group)

    def zero_grad(self) -> None:
        for group in self.param_groups:
            for p in group['params']:
                p.zero_grad()

    def step(self) -> None:
        raise NotImplementedError

    def add_param_group(self, param_group: dict):
        assert isinstance(param_group, dict), "param group must be a dict"
        params = param_group['params']

        # 转换为列表
        if isinstance(params, Parameter):
            param_group['params'] = [params]
        else:
            param_group['params'] = list(params)

        for name, default in self.defaults.items():
            param_group.setdefault(name, default)

        self.param_groups.append(param_group)


class SGD(Optimizer):
    '''
    随机梯度下降
    '''

    def __init__(self, params, lr: float = 1e-3, weight_decay=0) -> None:
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self) -> None:
        with no_grad():
            for group in self.param_groups:
                weight_decay = group['weight_decay']
                lr = group['lr']

                for p in group['params']:
                    d_p = p.grad  # 我们不能直接修改p.grad
                    # 对于设置了weight_decay的参数
                    if weight_decay != 0:
                        d_p += weight_decay * p
                    p -= d_p * lr
