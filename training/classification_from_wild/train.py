from __future__ import print_function, division
from utils import TrainConfig, Config, StorageName, read_dataset
from sklearn.utils import class_weight
import numpy as np
from sklearn.model_selection import train_test_split
import os
import pandas as pd
from dataset_hdf5 import HDF5Dataset
from torch.utils.data import DataLoader
from augmentation import get_augmentation
from torch_module import train_torch


def torch_generators(
    train, test, batch_size, FLAGS, class_weights, dataset_path
):
    kwargs = dict()
    storage_path = StorageName.storage_path + FLAGS.storage
    transformations_for_phase = get_augmentation()
    dataloaders = {
        x: DataLoader(
            HDF5Dataset(storage_path, y, transformations_for_phase[x]),
            batch_size=batch_size,
            num_workers=1,
        )
        for x, y in {"train": train, "valid": test}.items()
    }
    kwargs["dataloaders"] = dataloaders
    kwargs["train"] = train
    kwargs["valid"] = test
    kwargs["class_weight"] = class_weights
    return kwargs


def get_train_val_generators(FLAGS):
    dataset = read_dataset(FLAGS.csv)
    print(f"len dataset {len(dataset)}")
    train_size = TrainConfig.train_size
    datasets_path = Config.dataset_path
    train_csv = f"{datasets_path}{FLAGS.csv}_train.csv"
    test_csv = f"{datasets_path}{FLAGS.csv}_test.csv"
    is_train_test_exists = os.path.isfile(train_csv)
    is_train_test_exists = is_train_test_exists and os.path.isfile(test_csv)
    if not FLAGS.cached or not is_train_test_exists:
        if "src" in dataset.columns:
            print("stratify on src too")
            stratify_on = dataset[["label", "src"]]
        else:
            stratify_on = dataset["label"]
        train, test = train_test_split(
            dataset,
            stratify=stratify_on,
            train_size=train_size,
            random_state=14,
        )
        train.to_csv(train_csv)
        test.to_csv(test_csv)
    else:
        train, test = pd.read_csv(train_csv), pd.read_csv(test_csv)

    class_weights = class_weight.compute_class_weight(
        "balanced", np.unique(train.label.values), train.label.values
    )
    class_weights = dict(enumerate(class_weights))
    print(class_weights)
    print(f"Train shape : {train.shape}")
    print(f"Test shape : {test.shape}")
    batch_size = FLAGS.batch_size
    return torch_generators(
        train, test, batch_size, FLAGS, class_weights, datasets_path
    )


def get_fit_generator_kwargs(FLAGS):
    return get_train_val_generators(FLAGS)


def train_model(FLAGS):
    kwargs = get_fit_generator_kwargs(FLAGS)
    train_torch(FLAGS, kwargs)
