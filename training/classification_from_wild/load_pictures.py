import requests
import numpy as np
from PIL import Image
import PIL
from tqdm import tqdm
import h5py
from utils import *


def validate(array):
    if (
        len(array.shape) != 3
        or array.shape[2] != 3
        or 0.5 > array.shape[0] / array.shape[1]
        or array.shape[0] / array.shape[1] > 1.3
    ):
        return False
    return True


def change_dataset(FLAGS, to_del):
    df = read_dataset(FLAGS.csv)
    cols = df.columns
    print(f"deleting {len(to_del)}")
    print(f"have {len(df)}")
    df = df[~df["url"].isin(to_del)]
    print(f"result {len(df)}")

    df[[x for x in cols if "Unnamed" not in x and x]].to_csv(
        Config.dataset_path + FLAGS.csv + ".csv"
    )


def picture_url_to_array(url, size):
    try:
        bytearray_picture = requests.get(url, stream=True, timeout=3).raw
        pil_image = Image.open(bytearray_picture)
        array = np.array(pil_image)
        if len(array.shape) != 3 or array.shape[2] != 3:
            raise
    except Exception as e:
        print(e)
        return None

    transformed_pil_image = pil_image.resize(size, resample=PIL.Image.BICUBIC)

    # image_type = 'story' if real_size[0] != real_size[1] else 'feed'

    return transformed_pil_image


def skip_loaded_urls(urls, storage_name, overwrite):
    if overwrite:
        return urls

    already_loaded_urls = set(
        Cipher.decode(x) for x in get_storage_keys(storage_name)
    )
    return list(set(urls).difference(already_loaded_urls))


def extend_original_pics_storage(FLAGS, urls, overwrite=False):
    urls = skip_loaded_urls(
        urls, StorageName.storage_path + FLAGS.storage, overwrite
    )
    if FLAGS.parallel_download:
        return extend_original_pics_storage_parallel(FLAGS, urls, overwrite)
    to_del = []
    with h5py.File(StorageName.storage_path + FLAGS.storage, "a") as h5f:
        for url in tqdm(urls):
            array_image = picture_url_to_array(url, Config.size_images)
            if array_image is None:
                to_del.append(url)
                continue
            h5f.create_dataset(Cipher.encode(url), data=array_image)
    change_dataset(FLAGS, to_del)


def extend_original_pics_storage_parallel(FLAGS, urls, overwrite=False):
    from threading import Thread

    def _load_and_save(url, size_images, h5f):
        array_image = picture_url_to_array(url, size_images)
        if array_image is None:
            return url
        h5f.create_dataset(Cipher.encode(url), data=array_image)

    def load_parallel(func, arg_list):
        to_del = []

        class MyThread(Thread):
            def __init__(self, arg, h5f):
                Thread.__init__(self)
                self.arg = arg
                self.h5f = h5f

            def run(self):
                el = func(self.arg, Config.size_images, self.h5f)
                if el:
                    to_del.append(el)

        with h5py.File(StorageName.storage_path + FLAGS.storage, "a") as h5f:
            threads = [MyThread(arg, h5f) for arg in arg_list]
            tqdm([thr.start() for thr in threads])
            tqdm([thr.join() for thr in threads])
            return to_del

    to_del = load_parallel(_load_and_save, urls)
    change_dataset(FLAGS, to_del)
