import numpy as np
import h5py
import torch
import cv2
from torch.utils import data
from utils import Cipher
import os


class HDF5Dataset(data.Dataset):
    def __init__(self, img_dir, csv,  transform=None, lb=None, torch=True):
        self.img_dir = img_dir
        self.csv = csv
        self.img_names = self.csv["name"].values
        self.y = self.csv["label"].values
        self.transform = transform

    def __getitem__(self, index):
        # img = Image.open(os.path.join(self.img_dir, self.img_names[index]))
        image = cv2.imread(os.path.join(self.img_dir, self.img_names[index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        
        label = self.y[index]
        img = image
        if self.transform is not None:
            # img = self.transform(image=np.swapaxes(
            #    np.swapaxes(np.array(img), 2, 0), 1, 2))["image"]
            if isinstance(self.transform, dict):
                augmented = self.transform[label](image=image)
            else:
                augmented = self.transform(image=image)
            img = np.moveaxis(augmented, -1, 0) 
            
        return torch.from_numpy(img).float(), torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return self.y.shape[0]
    