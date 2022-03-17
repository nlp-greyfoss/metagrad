import math

import numpy as np

from metagrad.dataset import Dataset


class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int = 1,
                 shuffle: bool = True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size

        self.data_size = len(dataset)

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
        X_batch, y_batch = batch
        return X_batch, y_batch

    def next(self):
        return self.__next__()

    def __iter__(self):
        return self
