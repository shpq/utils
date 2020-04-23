import numpy as np
import h5py
import torch
from torch.utils import data
from utils import Cipher


class HDF5Dataset(data.Dataset):
    def __init__(self, file_path, csv, transform=None, lb=None, torch=True):
        super().__init__()
        self.data_info = []
        self.data_cache = {}
        self.transform = transform
        self.store_raw = h5py.File(file_path, "r")
        self.csv = csv.reset_index()
        self.labels = (
            self.csv.label.values
            if lb is None
            else lb.transform(self.csv.label.values)
        )

    def __getitem__(self, index):
        # get data
        url = self.csv.at[index, "url"]
        x = self.store_raw[Cipher.encode(url)][:]
        y = self.csv.at[index, "label"]

        if self.transform:
            if isinstance(self.transform, dict):
                x = self.transform[y](x)
            else:
                x = self.transform(x)
        if torch:
            x = np.swapaxes(np.swapaxes(x, 2, 0), 1, 2)

        x = torch.from_numpy(x).float()

        # get label

        y = torch.tensor(y, dtype=torch.long)
        return (x, y)

    def __len__(self):
        return len(self.csv)
