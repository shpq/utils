import argparse
import numpy as np
from train import train_model
from load_pictures import extend_original_pics_storage
from utils import read_dataset, TrainConfig, Config, StorageName, create_folder
import os


def change_cf(FLAGS):
    TrainConfig.batch_size = FLAGS.batch_size
    Config.size_images = (FLAGS.img_size, FLAGS.img_size)
    TrainConfig.input_shape_raw = (FLAGS.img_size, FLAGS.img_size, 3)
    StorageName.storage = f"{FLAGS.storage}_{FLAGS.img_size}x{FLAGS.img_size}"
    FLAGS.storage = f"{FLAGS.storage}_{FLAGS.img_size}x{FLAGS.img_size}"


def create_folders():
    for folder in [
        Config.dataset_path,
        TrainConfig.checkpoints_folder,
        StorageName.storage_path,
        os.path.join(StorageName.storage_path, StorageName.storage, '')
    ]:
        print(folder)
        create_folder(folder)


def main(FLAGS):
    change_cf(FLAGS)
    dataset = read_dataset(FLAGS.csv)
    print("download new images")
    dataset = dataset.drop_duplicates(subset=['url'])
    create_folders()
    extend_original_pics_storage(FLAGS, dataset)
    train_model(FLAGS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    parser.add_argument("--storage", type=str, help="Pics storage name")

    parser.add_argument("--csv", type=str, help="Name of csv in csv folder")

    parser.add_argument(
        "--cached",
        default=False,
        help="Use cached split?",
        action="store_true",
    )

    parser.add_argument(
        "--schedule",
        default=False,
        help="Use lr schedule?",
        action="store_true",
    )

    parser.add_argument(
        "--pretrained",
        type=str,
        default="xception",
        help="Which pretrained model do you wanna use?",
    )

    parser.add_argument(
        "--depth_trainable",
        type=int,
        default=1,
        help="Depth of trainable layers",
    )

    parser.add_argument(
        "--epoch_reduce",
        type=int,
        default=5,
        help="After what epoch we need to reduce lr?",
    )

    parser.add_argument("--lr", type=float, help="Learning rate")

    parser.add_argument(
        "--batch_size", type=int, default=-1, help="Size of the batch"
    )

    parser.add_argument("--img_size", type=int, help="Image size")

    parser.add_argument(
        "--gamma", type=float, default=0.5, help="Step reducing?"
    )

    parser.add_argument(
        "--saved", type=str, default="-", help="Saved model?"
    )

    parser.add_argument(
        "--quantize",
        default=False,
        help="Quantize model?",
        action="store_true",
    )

    parser.add_argument(
        "--qconfig",
        default=None,
        help="fbgemm or qnnpack qconfig for quantization",
    )

    parser.add_argument(
        "--dropout",
        default=0,
        help="Dropout rate to put before last layer",
    )

    parser.add_argument(
        "--src",
        default="timm",
        help="Source where we can get this model",
    )

    FLAGS = parser.parse_args()
    main(FLAGS)
