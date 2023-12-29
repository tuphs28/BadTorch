import torch, random
from typing import Tuple

class Dataset:
    """
    Dataset object for input and target tensors. Observation ID/number must be first dimension of both tensors.

    Args:
        inputs (torch.Tensor): tensor of inputs.
        targets (torch.Tensor): tensor of targets corresponding to inputs with same inpit index in 0th dimension.
    """

    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:

        assert inputs.shape[0] == targets.shape[0], "Different number of inputs and targets"
        self.n = inputs.shape[0]

        self.inputs = inputs
        self.targets = targets

    def __len__(self) -> int:
        return self.n
        

class DataLoader:
    """
    Iterator that generates mini-batches of data from provided Dataset of input and target tensors.

    Args:
        dataset (Dataset): dataset of input and corresponding target tensors to iterate through.
        batch_size (int): number of inputs and targets to select.
        shuffle (bool): whether to randomly shuffle the order in which data is presented. Defaults to True
    """

    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = True) -> None:
        self.inputs = dataset.inputs
        self.targets = dataset.targets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = dataset.n
        self.iter_count = 0

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.iter_count == 0:
            self.idxs = list(range(self.n))
            if self.shuffle:
                random.shuffle(self.idxs)
        if (self.iter_count+1)*self.batch_size > self.n:
            self.iter_count = 0
            raise StopIteration
        else:
            batch_idxs = self.idxs[self.iter_count*self.batch_size: (self.iter_count+1)*self.batch_size]
            current_inputs = self.inputs[batch_idxs]
            current_targets = self.targets[batch_idxs]
            self.iter_count += 1
            return current_inputs, current_targets

    def __len__(self) -> int:
        return self.n