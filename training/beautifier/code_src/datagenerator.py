import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from random import shuffle


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, cfg, transform=None):
        self.batch_size = cfg.training.batch_size
        # self.img_dir = cfg.dataset.train_path if mode == "train" else cfg.dataset.test_path
        self.ugly_pics = cfg.dataset.ugly_pics
        self.beauty_pics = cfg.dataset.beauty_pics
        self.ugly_images_paths = [os.path.join(
            self.ugly_pics, img_name) for img_name in os.listdir(self.ugly_pics)]
        self.beauty_images_paths = [os.path.join(
            self.beauty_pics, img_name) for img_name in os.listdir(self.beauty_pics)]
        self.cfg = cfg
        self.transform = transform
        self.size = tuple(cfg.training.size)
        self.on_epoch_end()

    def __len__(self):
        return min(len(self.ugly_images_paths), len(self.beauty_images_paths)) // self.batch_size

    def __getitem__(self, index):
        X, y = self.__get_data(index)
        return X, y

    def on_epoch_end(self):
        if self.cfg.training.shuffle:
            shuffle(self.ugly_images_paths)
            shuffle(self.beauty_images_paths)

    def __get_data(self, index):
        X = np.empty((self.batch_size, *self.size, 3))
        y = np.empty((self.batch_size, *self.size, 3))

        for i, id in enumerate(range(
                index * self.batch_size, (index+1) * self.batch_size)):
            image_beauty = np.array(Image.open(self.beauty_images_paths[id]))
            image_ugly = np.array(Image.open(self.ugly_images_paths[id]))

            _, image_beauty = self.transform.augment_saving_beauty(
                image_beauty)
            _, image_ugly = self.transform.augment_saving_beauty(
                image_ugly)
            # image_ugly = self.transform.augment_reducing_beauty(image_for_augm)

            X[i, ] = image_ugly
            y[i, ] = image_beauty

        return X, y
