import os
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm
from PIL import Image
import requests
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    parser.add_argument(
        "--number",
        type=int,
        default=1000,
        help="Number of parallel workers",
    )

    args = parser.parse_args()

    main_directory = os.path.dirname(__file__)
    csvs_dir = main_directory + "wild_ages_csvs"
    dfs_names = os.listdir(csvs_dir)
    df_list = [pd.read_csv(csvs_dir + "/" + df_name) for df_name in dfs_names if '.csv' in df_name]
    df = pd.concat(df_list).reset_index(drop=True)
    df = df[["text", "url", "src", "age"]]
    number = args.number
    print(f"new df len {len(df)}")

    try:
        df_prev = pd.read_pickle("wild_ages_with_path.pk")
        print(f"previous df len: {len(df_prev)}")
        df_prev = df_prev[~pd.isna(df_prev.url)]
        print(f"previous df with not nan path len: {len(df_prev)}")
    except:
        df_prev = None

    if df_prev is not None:
        df_append = df[~df.url.isin(df_prev.url)]
        print(f"df to append len: {len(df_append)}")
        df = df.append(df_append).reset_index(drop=True)

    def get_results(value):
        try:
            index, row = value
            url = row["url"]
            picture_bytearray = requests.get(url, stream=True).raw
            image = Image.open(picture_bytearray)
            name = url.split("/")[-1].split("?")[0]
            path = main_directory + "images_from_the_wild/" + name
            image.save(path)

        except:
            return None
        return path

    with Pool(number) as p:
        r = list(tqdm(p.imap(get_results, df.iterrows()), total=len(df)))
        df["paths"] = r
        df.to_csv("wild_ages_with_path.csv")
        df.to_pickle("wild_ages_with_path.pk")
