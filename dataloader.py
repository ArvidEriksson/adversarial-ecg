import math
import torch
from torch.utils.data import Dataset
import numpy as np
import h5py


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
        batch = [np.array(t[self.start : end]) for t in self.tensors]
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


class H5Dataset(Dataset):
    def __init__(self, file_path, dataset_name, labels):
        self.file_path = file_path
        self.dataset_name = dataset_name
        self.labels = labels
        self.file = h5py.File(file_path, 'r')
        self.length = len(self.file[dataset_name])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = self.file[self.dataset_name][idx]
        # Convert to torch tensor
        sample = torch.tensor(sample, dtype=torch.float32)
        label = self.labels[idx]
        return (sample, label)

    def __del__(self):
        self.file.close()


class MultExamH5Dataset(Dataset):
    def __init__(self, file_path, labels, patient_ids):
        self.file_path = file_path
        self.labels = labels
        self.patient_ids = patient_ids
        self.file = h5py.File(file_path, 'r')
        self.length = len(self.file['tracings'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = self.file['tracings'][idx]
        # Convert to torch tensor
        sample = torch.tensor(sample, dtype=torch.float32)
        label = self.labels[idx]
        patient_id = self.patient_ids[idx]
        return (sample, patient_id, label)

    def __del__(self):
        self.file.close()
