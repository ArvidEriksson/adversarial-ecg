import math
import torch
import numpy as np


class BatchDataloader:
    def __init__(self, *tensors, batch_size=1):
        self.tensors = tensors
        self.batch_size = batch_size
        self.start_idx = 0
        self.end_idx = len(tensors[0])

    def __next__(self):
        if self.start == self.end_idx:
            raise StopIteration
        end = min(self.start + self.batch_size, self.end_idx)
        batch = [np.array(t[self.start:end]) for t in self.tensors]
        self.sum += len(batch[0])
        self.start = end
        return [torch.tensor(b, dtype=torch.float32) for b in batch]

    def __iter__(self):
        self.start = self.start_idx
        self.sum = 0
        return self

    def __len__(self):
        count = 0
        start = self.start_idx
        while start != self.end_idx:
            end = min(start + self.batch_size, self.end_idx)
            count += 1
            start = end
        return count