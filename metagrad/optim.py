import math
from collections import defaultdict
from copy import deepcopy
from itertools import chain
from typing import Dict, Any, Tuple, Optional

from metagrad.paramater import Parameter
from metagrad.tensor import Tensor
from metagrad import no_grad


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
        # 为每个Parameter维护一个状态
        self.state = defaultdict(dict)

        param_groups = list(params)

        # 如果不是字典
        if not isinstance(param_groups[0], dict):
            # 就把它转换为字典列表
            param_groups = [{'params': param_groups}]

        for param_group in param_groups:
            self.add_param_group(param_group)

    def __getstate__(self):
        return {
            "defaults": self.defaults,
            "state": self.state,
            "param_groups": self.param_groups
        }

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __repr__(self):
        format_string = self.__class__.__name__ + ' ('
        for i, group in enumerate(self.param_groups):
            format_string += f'\nParameter Group {i}\n'
            for key in sorted(group.keys()):
                if key != 'params':
                    format_string += f'    {key}: {group[key]}\n'
        format_string += ')'
        return format_string

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
        if isinstance(params, Tensor):
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
                    p.add_(d_p, alpha=-lr)


class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def _init_group(self, group, params_with_grad, grads, exp_avgs, exp_avg_sqs, state_steps):
        for p in group["params"]:
            if p.grad is not None:
                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = Tensor(0.)
                    state['exp_avg'] = Tensor.zeros_like(p)
                    state['exp_avg_sq'] = Tensor.zeros_like(p)  # v

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                state_steps.append(state['step'])

    def step(self) -> None:
        with no_grad():
            for group in self.param_groups:
                params_with_grad = []
                grads = []
                exp_avgs = []
                exp_avg_sqs = []
                state_steps = []
                beta1, beta2 = group['betas']

                weight_decay = group['weight_decay']
                lr = group['lr']
                eps = group['eps']

                self._init_group(group, params_with_grad, grads, exp_avgs, exp_avg_sqs, state_steps)

                for i, param in enumerate(params_with_grad):
                    grad = grads[i]
                    exp_avg = exp_avgs[i]
                    exp_avg_sq = exp_avg_sqs[i]
                    step_t = state_steps[i]

                    # 更新step
                    step_t += 1
                    if weight_decay != 0:
                        grad = grad + weight_decay * param.grad

                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    step = step_t.item()

                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step

                    step_size = lr / bias_correction1
                    bias_sqrt = math.sqrt(bias_correction2)

                    denom = (exp_avg_sq.sqrt() / bias_sqrt).add_(eps)

                    param.addcdiv_(exp_avg, denom, value=-step_size)
