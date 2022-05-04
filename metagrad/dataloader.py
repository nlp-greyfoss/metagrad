import math
from typing import Optional, Callable, List, Any, TypeVar

import numpy as np

from metagrad.dataset import Dataset

T = TypeVar('T')  # 泛型
# 输入是一个T列表，返回可以是任何类型
_collate_fn_t = Callable[[List[T]], Any]


def default_collate(batch):
    '''
    默认整理批次函数
    :param batch: 批次
    :return:
    '''

    # 先只处理(X, y)这种元组情况
    elem = batch[0]
    if isinstance(elem, tuple) and len(elem) == 2:
        X_batch, y_batch = batch
        return X_batch, y_batch

    return batch


class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int = 1,
                 shuffle: bool = True, collate_fn: Optional[_collate_fn_t] = None):
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size

        self.data_size = len(dataset)

        if collate_fn is None:
            collate_fn = default_collate

        self.collate_fn = collate_fn

        self.max_its = math.ceil(self.data_size / batch_size)
        self.it = 0  # 迭代次数
        self.indices = None
        self.reset()

    def reset(self):
        self.it = 0
        if self.shuffle:
            self.indices = np.random.permutation(self.data_size)
        else:
            self.indices = np.arange(self.data_size)

    def __next__(self):
        if self.it >= self.max_its:
            self.reset()
            raise StopIteration

        i, batch_size = self.it, self.batch_size
        batch_indices = self.indices[i * batch_size:(i + 1) * batch_size]
        batch = self.dataset[batch_indices]
        self.it += 1

        return self.collate_fn(batch)

    def next(self):
        return self.__next__()

    def __iter__(self):
        return self
