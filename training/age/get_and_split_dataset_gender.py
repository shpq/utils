import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from multiprocessing import Pool



if __name__ == "__main__":
    tqdm.pandas()
    main_directory = os.path.dirname(__file__)
    df = pd.read_pickle(
        main_directory + "ages_from_the_wild/wild_gender_with_detections.pk")
    df.dropna(subset=["paths"], inplace=True)
    df.dropna(subset=['detections'], inplace=True)
    df["num_images"] = df["detections"].progress_apply(lambda x : len(x))
    df = df[df['num_images'] > 0] 
    list_files = os.listdir(main_directory + "ages_from_the_wild/cropped_images/")
    rows = []
    def get_age(name):
        file_last = name.split("n_")[0]
        values = df[df.paths.str.contains(file_last)]
        if not values.shape[0]:
            return None
        values = values.iloc[0]
        return values["gender"]

    p = Pool(30)
    genders = list(tqdm(p.imap(get_age, list_files), total=len(list_files)))
    p.close()
    df_new = pd.DataFrame({"name" : list_files, "gender" : genders})

    df_new.to_pickle("gender_from_the_wild.pk")
    df_new.to_csv("gender_from_the_wild.csv")
    stratify_on = df_new["gender"]
    df_train, df_test = train_test_split(df_new, stratify=stratify_on, train_size=0.8,
                                         random_state=14)
    df_train.to_csv("gender_from_the_wild_v2_train.csv")
    df_test.to_csv("gender_from_the_wild_v2_test.csv")
