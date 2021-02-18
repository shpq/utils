import os
import requests
import numpy as np
from PIL import Image
import PIL
import pandas as pd
from tqdm import tqdm
from multiprocessing.pool import ThreadPool as Pool


def validate(array):
    if (
        len(array.shape) != 3
        or array.shape[2] != 3
        or 0.5 > array.shape[0] / array.shape[1]
        or array.shape[0] / array.shape[1] > 1.3
    ):
        return False
    return True


def change_dataset(cfg, to_del):
    df = pd.read_csv(cfg.dataset.csv_path)
    cols = df.columns
    print(f"deleting {len(to_del)}")
    print(f"have {len(df)}")
    df[[x for x in cols if x and "Unnamed" not in x]].to_csv(
        cfg.dataset.csv_path_back)
    df = df[~df["url"].isin(to_del)].reset_index(drop=True)
    print(f"result {len(df)}")

    df[[x for x in cols if "Unnamed" not in x and x]].to_csv(
        cfg.dataset.csv_path)


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

    # transformed_pil_image = pil_image.resize(size, resample=PIL.Image.BICUBIC)

    return pil_image # transformed_pil_image


def extend_original_pics_storage(cfg, overwrite=False):
    dataset = pd.read_csv(cfg.dataset.csv_path)
    if "name" not in dataset.columns:
        dataset.reset_index(drop=True, inplace=True)
        dataset["name"] = dataset.index.astype("str") + ".jpg"
        dataset.to_csv(cfg.dataset.csv_path)
    dataset = pd.read_csv(cfg.dataset.csv_path)
    urls_skipped = skip_loaded_urls(
        dataset, cfg.dataset.path, overwrite
    )
    return extend_original_pics_storage_parallel(cfg, urls_skipped, dataset, overwrite)


def skip_loaded_urls(dataset, storage_path, overwrite):
    if overwrite:
        return dataset.urls.unique()

    os.makedirs(storage_path, exist_ok=True)
    already_loaded_names = list(set(
        [x for x in os.listdir(storage_path) if '.jpg' in x]
    ))
    print(f"already_loaded_names len {len(already_loaded_names)}")
    print(f'without loaded names len {len(dataset[~dataset.name.isin(already_loaded_names)].url.values)}')
    return dataset[~dataset.name.isin(already_loaded_names)].url.values




def extend_original_pics_storage_parallel(cfg, urls, dataset, overwrite=False):
    def _load_and_save(v):
        url = v['url']
        name_jpg = v['name']
        size_images = tuple(cfg.training.img_size)
        array_image = picture_url_to_array(url, size_images)
        if array_image is None:
            return url
        array_image.save(os.path.join(cfg.dataset.path, name_jpg))


    urls_with_names = dataset[dataset.url.isin(urls)][['url', 'name']]
    urls_with_names = urls_with_names.to_dict(orient='records')
    print(f"urls_with_names len {len(urls_with_names)}")
    with Pool(25) as p:
        to_del = list(
            tqdm(p.imap(_load_and_save, urls_with_names), total=len(urls_with_names)))
    to_del = [x for x in to_del if x]
    change_dataset(cfg, to_del)
