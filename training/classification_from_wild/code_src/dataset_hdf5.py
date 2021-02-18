import numpy as np
import h5py
import torch
from torch.utils import data
import os
from PIL import Image


class HDF5Dataset(data.Dataset):
    def __init__(self, cfg, img_dir, csv, size, transform=None, mode="classification"):
        self.img_dir = img_dir
        self.csv = csv
        self.cfg = cfg
        self.img_names = self.csv["name"].values
        self.y = self.csv["label"].values
        if self.cfg.general.custom == "make_feed":
            self.bottom = self.csv["bottom"].values
            self.top = self.csv["top"].values
        self.transform = transform
        self.size = size
        self.mode = mode
        self.num_classes = len(np.unique(self.y))

    def __getitem__(self, index):

        img = Image.open(os.path.join(self.img_dir, self.img_names[index]))
        rgbimg = Image.new("RGB", img.size)
        rgbimg.paste(img)
        img = rgbimg
        label = self.y[index]
        if self.cfg.general.custom == "make_feed":
            image = Image.fromarray(np.array(img)[int(self.bottom):int(self.top), :, :])
        else:
            image = img
        
        if self.transform is not None:

            if isinstance(self.transform, dict):
                transform = self.transform[label]
            else:
                transform = self.transform

            image = transform(image=image, size=self.size)
        if self.mode == "ordinal_regression":
            label = int(label)
            levels = [1] * label + [0] * (self.num_classes - 1 - label)
            levels = torch.tensor(levels, dtype=torch.float32)
            return image, torch.tensor(label, dtype=torch.long), levels
        elif self.mode == "classification":
            return image, torch.tensor(label, dtype=torch.long)
        elif self.mode == "regression":
            return image, torch.tensor(label, dtype=torch.float)

    def __len__(self):
        return self.y.shape[0]
