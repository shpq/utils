import base64
import os
import h5py
import pandas as pd
from threading import Thread


main_directory = os.path.dirname(__file__)


def load_parallel(func, arg_list):
    loaded_objects = {}

    class MyThread(Thread):
        def __init__(self, arg):
            Thread.__init__(self)
            self.arg = arg

        def run(self):
            loaded_objects[self.arg[0]] = func(self.arg[0], self.arg[1])

    threads = [MyThread(arg) for arg in arg_list]
    [thr.start() for thr in threads]
    [thr.join() for thr in threads]
    return loaded_objects


def create_folder(folder):
    if not os.path.exists(os.path.join('.', os.path.dirname(folder))):
        os.mkdir(os.path.join('.', os.path.dirname(folder)))


def read_dataset(filename):
    path =  os.path.join(Config.dataset_path, f"{filename}.csv")
    dataset = pd.read_csv(path)
    return dataset


class Config:
    dataset_path = os.path.join(main_directory, "csvs")


class TrainConfig:
    checkpoints_folder = os.path.join(main_directory, "models_checkpoints")
    train_size = 0.85


class StorageName:
    storage_path = os.path.join(main_directory, "storages")


class Cipher:
    @staticmethod
    def encode(s):
        return base64.urlsafe_b64encode(s.encode()).decode()

    @staticmethod
    def decode(s):
        return base64.urlsafe_b64decode(s.encode()).decode()


def get_storage_keys(storage_name):
    if not os.path.exists(storage_name):
        print('storage doesnt exist')
        return []
    print(f'storage exists, len ({len(os.listdir(storage_name))})')
    return os.listdir(storage_name)
