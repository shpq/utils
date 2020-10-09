import requests
import numpy as np
from PIL import Image
import PIL
from tqdm import tqdm
import h5py
from utils import *
from multiprocessing import Pool


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
    df[[x for x in cols if "Unnamed" not in x and x]].to_csv(
        os.path.join(Config.dataset_path, FLAGS.csv + "_backup.csv")
    )
    df = df[~df["url"].isin(to_del)]
    print(f"result {len(df)}")

    df[[x for x in cols if "Unnamed" not in x and x]].to_csv(
        os.path.join(Config.dataset_path, FLAGS.csv + ".csv")
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

    return transformed_pil_image




def extend_original_pics_storage(FLAGS, dataset, overwrite=False):
    urls_skipped = skip_loaded_urls(
        dataset, os.path.join(StorageName.storage_path, FLAGS.storage), overwrite
    )
    return extend_original_pics_storage_parallel(FLAGS, urls_skipped, dataset, overwrite)

def skip_loaded_urls(dataset, storage_name, overwrite):
    if overwrite:
        return dataset.urls.unique()

    already_loaded_names = list(set(
         [x for x in get_storage_keys(storage_name) if '.jpg' in x]
    ))
    print(f"already_loaded_names len {len(already_loaded_names)}")
    print(f'withput loaded names len {len(dataset[~dataset.name.isin(already_loaded_names)].url.values)}')
    return dataset[~dataset.name.isin(already_loaded_names)].url.values


def _load_and_save(v):
    url = v['url']
    name_jpg = v['name']
    size_images=Config.size_images
    name=StorageName.storage_path + StorageName.storage

    array_image = picture_url_to_array(url, size_images)
    if array_image is None:
        return url
    array_image.save(os.path.join(StorageName.storage_path, StorageName.storage, name_jpg))

def extend_original_pics_storage_parallel(FLAGS, urls, dataset, overwrite=False):
    
    urls_with_names = dataset[dataset.url.isin(urls)][['url', 'name']]
    urls_with_names = urls_with_names.to_dict(orient='records')
    print(f"urls_with_names len {len(urls_with_names)}")
    with Pool(25) as p:
        to_del = list(tqdm(p.imap(_load_and_save, urls_with_names), total=len(urls_with_names)))
    to_del = [x for x in to_del if x]
    change_dataset(FLAGS, to_del)
