import math
from collections import defaultdict, Counter

from metagrad import no_grad
from metagrad.tensor import Tensor


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


class LRScheduler:
    def __init__(self, optimizer: Optimizer, last_epoch: int = -1, verbose: bool = False):
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                # 设置初始值
                group.setdefault("initial_lr", group["lr"])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError(f"param 'initial_lr' is not specified "
                                   "in param_groups[{i}] when resuming an optimizer")
        # 保存param_groups的初始学习率
        self.base_lrs = [group["initial_lr"] for group in optimizer.param_groups]
        self.last_epoch = last_epoch
        self.verbose = verbose
        self._initial_step()

    def get_lr(self):
        return NotImplementedError

    def get_last_lr(self):
        return self._last_lr

    def print_lr(self, is_verbose, group, lr, epoch=None):
        """如果 is_verbose为True， 打印当前的学习率"""
        if is_verbose:
            if epoch is None:
                print(f"Adjusting learning rate of group {group} to {lr:.4e}.")
            else:
                epoch_str = ("%.2f" if isinstance(epoch, float) else "%.5d") % epoch
                print(f'Epoch {epoch_str}: adjusting learning rate of group {group} to {lr:.4e}.')

    def _initial_step(self):
        """初始化step count并调用一次step"""
        self.optimizer._step_count = 0
        self._step_count = 0
        self.step()

    def step(self, epoch=None):

        self._step_count += 1

        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch

        for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
            param_group, lr = data
            param_group["lr"] = lr  # 用新的学习率覆盖当前学习率
            self.print_lr(self.verbose, i, lr, epoch)
        # 保存最近一次学习率
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


class ExponentialLR(LRScheduler):
    def __init__(self, optimizer, gamma, last_epoch=-1, verbose=False):
        """
        每个epoch通过gamma衰减每个parameter group的学习率，当last_epoch=-1，学习率设为初始值
        :param optimizer: 优化器
        :param gamma: 学习率衰减的乘法因子
        :param last_epoch: 最后一次epoch的索引
        :param verbose: 是否为每次更新打印信息
        """
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch == 0:
            # 第一次迭代就是初始学习率
            return [group["lr"] for group in self.optimizer.param_groups]
        # 然后是当前学习率乘以gamma
        return [group["lr"] * self.gamma for group in self.optimizer.param_groups]


class StepLR(LRScheduler):
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1, verbose=False):
        """
        每step_size个epoch通过gamma衰减每个parameter group的学习率，当last_epoch=-1，学习率设为初始值

        :param optimizer:
        :param step_size:
        :param gamma:
        :param last_epoch:
        :param verbose:
        """
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch == 0 or self.last_epoch % self.step_size != 0:
            # 第一次迭代或在第一个step_size间隔内
            return [group["lr"] for group in self.optimizer.param_groups]
        # 然后是当前学习率乘以gamma
        return [group["lr"] * self.gamma for group in self.optimizer.param_groups]


class MultiStepLR(LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1, verbose=False):
        """
        一旦epoch次数达到milestones中的次数，则通过gamma衰减每个parameter group的学习率，当last_epoch=-1，学习率设为初始值

        :param optimizer:
        :param milestones: epoch索引列表，注意必须是递增的
        :param gamma:
        :param last_epoch:
        :param verbose:
        """
        self.milestones = Counter(milestones)
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch not in self.milestones:
            # 如果不在milestones内，则返回当前的学习率
            return [group["lr"] for group in self.optimizer.param_groups]
        # 然后是当前学习率乘以gamma的milestones[last_epoch]次
        return [group["lr"] * self.gamma ** self.milestones[self.last_epoch] for group in self.optimizer.param_groups]


class LambdaLR(LRScheduler):
    def __init__(self, optimizer, lr_lambda, last_epoch=-1, verbose=False):
        """
        让每个parameter group的学习率为初始学习率乘以一个给定的函数lr_lambda
        :param optimizer:
        :param lr_lambda(function or list): 一个基于epoch计算乘法因子的函数；或是一个这样的函数列表，列表中每个函数
                                            对应optimizer.param_groups的每个group
        :param last_epoch:
        :param verbose:
        """
        self.optimizer = optimizer

        if not isinstance(lr_lambda, list) and not isinstance(lr_lambda, tuple):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            # 如果是列表的话必须和param_groups的大小一致
            if len(lr_lambda) != len(optimizer.param_groups):
                raise ValueError(f"Expected {len(optimizer.param_groups)} lr_lambdas, but got {len(lr_lambda)}")
            self.lr_lambdas = list(lr_lambda)

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        return [base_lr * lmbda(self.last_epoch) for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)]


class CosineAnnealingLR(LRScheduler):
    pass


class NoamLR(LRScheduler):
    def __init__(self, optimizer, model_size, factor=1., warmup_steps=4000, last_epoch=-1, verbose=False):
        """
        参考 http://nlp.seas.harvard.edu/annotated-transformer 实现的Transformer提出的学习率衰减方法
        在第一个warmup_steps内线性地增大学习率，然后按步长的平方倒数成比例地减小
        :param optimizer: 优化器
        :param model_size: 模型嵌入层大小
        :param factor: 乘法因子
        :param warmup_steps: 加热步
        :param last_epoch:
        :param verbose:
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.model_size = model_size
        self.factor = factor

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        # 避免0的负幂次
        if self.last_epoch == 0:
            self.last_epoch = 1

        step = self.last_epoch
        lr = self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup_steps ** (-1.5)))
        return [lr] * len(self.optimizer.param_groups)
