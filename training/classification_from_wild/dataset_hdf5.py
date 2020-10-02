import numpy as np
import h5py
import torch
import cv2
from torch.utils import data
from utils import Cipher
import os
from PIL import Image


class HDF5Dataset(data.Dataset):
    def __init__(self, img_dir, csv, size, transform=None, lb=None, torch=True):
        self.img_dir = img_dir
        self.csv = csv
        self.img_names = self.csv["name"].values
        self.y = self.csv["label"].values
        self.transform = transform
        self.size = size

    def __getitem__(self, index):

        img = Image.open(os.path.join(self.img_dir, self.img_names[index]))

        label = self.y[index]
        image = img
        if self.transform is not None:

            if isinstance(self.transform, dict):
                transform = self.transform[label]
            else:
                transform = self.transform

            image = transform(image=image, x=self.size, y=self.size)

        return image, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return self.y.shape[0]
