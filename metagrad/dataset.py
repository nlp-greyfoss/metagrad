from metagrad.tensor import Tensor


class Dataset:

    def __getitem__(self, index):
        return NotImplementedError


class TensorDataset(Dataset):
    def __init__(self, *tensors: Tensor) -> None:
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return len(self.tensors[0])
