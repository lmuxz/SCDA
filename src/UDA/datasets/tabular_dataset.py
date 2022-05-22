import torch
import numpy as np
from torch.utils.data import Dataset

class TabularDataset(Dataset):
    def __init__(self, input:np.ndarray, cate_index:int, domain:int, output:np.ndarray=None) -> None:
        super().__init__()

        self.cate_input = torch.from_numpy(input[:, :cate_index]).long()
        self.num_input = torch.from_numpy(input[:, cate_index:]).float()

        self.output = None
        if output is not None:
            self.output = torch.from_numpy(output.reshape(-1)).long()

        self._domain = domain

    @property
    def domains(self):
        return self._domain


    def __getitem__(self, index):

        if self.output is not None:
            return {
                'cate_input': self.cate_input[index],
                'num_input': self.num_input[index],
                'output': self.output[index],
                'domain': self._domain
            }
        else:
            return {
                'cate_input': self.cate_input[index],
                'num_input': self.num_input[index],
                'domain': self._domain
            }

    def __len__(self):
            return len(self.cate_input)