import math
from collections import defaultdict

from metagrad.paramater import Parameter
from metagrad.tensor import no_grad, Tensor


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


class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)
        self.state = defaultdict(dict)

    def step(self) -> None:
        with no_grad():
            for group in self.param_groups:
                beta1, beta2 = group['betas']
                eps = group['eps']
                lr = group['lr']
                grads = []
                state_steps = []
                exp_avgs = []
                exp_avg_sqs = []

                params = group['params']

                for p in params:
                    state = self.state[p]

                    grads.append(p.grad)

                    if len(state) == 0:
                        state['step'] = Tensor(0)
                        state['exp_avg'] = Tensor.zeros_like(p)  # m
                        state['exp_avg_sq'] = Tensor.zeros_like(p)  # v

                    state_steps.append(state['step'])
                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                for i, param in enumerate(params):
                    grad = grads[i]

                    step_t = state_steps[i]
                    step_t += 1
                    step = step_t.item()

                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step

                    a = lr * math.sqrt(bias_correction2) / bias_correction1

                    exp_avgs[i] = beta1 * exp_avgs[i] + (1.0 - beta1) * grad
                    exp_avg_sqs[i] = beta2 * exp_avg_sqs[i] + (1.0 - beta2) * grad * grad

                    p -= a * exp_avgs[i] / (exp_avg_sqs[i].sqrt() + eps)
