from __future__ import print_function, division
from sklearn.utils import class_weight
import numpy as np
from sklearn.model_selection import train_test_split
import os
import pandas as pd
from torch.utils.data import DataLoader
from code_src.dataset_hdf5 import HDF5Dataset
from code_src.train_keras import train_keras, DataGenerator
from code_src.augmentation import get_augmentation
from code_src.torch_module import train_torch


def torch_generators(
    train, test, batch_size, cfg, class_weights):
    kwargs = dict()
    transformations_for_phase = get_augmentation()
    dataloaders = {
        x: DataLoader(
            HDF5Dataset(cfg, cfg.dataset.path, y, cfg.training.img_size,
                        transformations_for_phase[x],mode=cfg.training.mode),
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


def keras_generators(
    train, test, batch_size, cfg, class_weights):
    kwargs = dict()
    transformations_for_phase = get_augmentation()
    dataloaders = {
        x: DataGenerator(
            cfg, cfg.dataset.path, y, transform=transformations_for_phase[x],
            batch_size=cfg.training.batch_size,
            num_classes=2, shuffle=False, size=cfg.training.img_size,
            strong_aug=cfg.training.strong_aug,
        )
        for x, y in {"train": train, "valid": test}.items()
    }
    kwargs["dataloaders"] = dataloaders
    kwargs["train"] = train
    kwargs["valid"] = test
    kwargs["class_weight"] = class_weights
    return kwargs


def get_train_val_generators(cfg):
    dataset = pd.read_csv(cfg.dataset.csv_path)
    print(f"len dataset {len(dataset)}")
    train_csv = os.path.join(cfg.dataset.csvs_path, f"{cfg.dataset.csv_name}_train.csv")
    test_csv = os.path.join(cfg.dataset.csvs_path, f"{cfg.dataset.csv_name}_test.csv")
    is_train_test_exists = os.path.isfile(train_csv) and os.path.isfile(test_csv)

    if not cfg.dataset.cached or not is_train_test_exists:
        if "src" in dataset.columns:
            print("stratify on src too")
            stratify_on = dataset[["label", "src"]]
        else:
            stratify_on = dataset["label"]

        train, test = train_test_split(
            dataset,
            stratify=stratify_on,
            train_size=cfg.training.train_size,
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
    if cfg.training.mode == "classification":
        print(class_weights)
    print(f"Train shape : {train.shape}")
    print(f"Test shape : {test.shape}")
    if cfg.general.framework == "torch":
        return torch_generators(
            train, test, cfg.training.batch_size, cfg, class_weights)
    elif cfg.general.framework == "keras":
        return keras_generators(
            train, test, cfg.training.batch_size, cfg, class_weights)
    else:
        raise NotImplementedError


def get_fit_generator_kwargs(cfg):
    return get_train_val_generators(cfg)


def train_model(cfg):
    kwargs = get_fit_generator_kwargs(cfg)
    if cfg.general.framework == "torch":
        train_torch(cfg, kwargs)
    elif cfg.general.framework == "keras":
        train_keras(cfg, kwargs)
    else:
        raise NotImplementedError
