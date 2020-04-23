import argparse
import numpy as np
from train import train_model
from load_pictures import extend_original_pics_storage
from utils import read_dataset, TrainConfig, Config, StorageName, create_folder


def change_cf(FLAGS):
    TrainConfig.batch_size = FLAGS.batch_size
    Config.size_images = (FLAGS.img_size, FLAGS.img_size)
    TrainConfig.input_shape_raw = (FLAGS.img_size, FLAGS.img_size, 3)
    FLAGS.storage = f"{FLAGS.storage}_{FLAGS.img_size}x{FLAGS.img_size}.hdf5"


def create_folders():
    for folder in [
        Config.dataset_path,
        TrainConfig.checkpoints_folder,
        StorageName.storage_path,
    ]:
        create_folder(folder)


def main(FLAGS):
    dataset = read_dataset(FLAGS.csv)
    urls = np.unique(dataset.url.values)
    print("download new images")
    change_cf(FLAGS)
    create_folders()
    extend_original_pics_storage(FLAGS, urls)
    train_model(FLAGS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    parser.add_argument("--storage", type=str, help="Pics storage name")

    parser.add_argument("--csv", type=str, help="Name of csv in csv folder")

    parser.add_argument(
        "--cat",
        default=False,
        help="Is data categorical?",
        action="store_true",
    )

    parser.add_argument(
        "--parallel_download",
        default=False,
        help="Is data categorical?",
        action="store_true",
    )

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
        "--framework",
        type=str,
        default="keras",
        help="Which framework do you wanna use? (keras / torch)",
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

    FLAGS = parser.parse_args()
    main(FLAGS)
