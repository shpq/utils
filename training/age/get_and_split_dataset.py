import pandas as pd
import os
from sklearn.model_selection import train_test_split


src_to_age = {f"https://www.instagram.com/explore/tags/{i}yearsold/": i for i in range(2, 90)}
src_to_age["https://www.instagram.com/explore/tags/1yearold/"] = 1
src_to_age["https://www.instagram.com/explore/tags/newborn/"] = 0

main_directory = os.path.dirname(__file__)
df = pd.read_pickle(
    main_directory + "ages_from_the_wild/wild_ages_with_path.pk")
df.dropna(subset=["paths"], inplace=True)
list_files = os.listdir(main_directory + "ages_from_the_wild/cropped_images/")
rows = []
for file in list_files:

    file_last = file.split("n_")[0]
    values = df[df.paths.str.contains(file_last)]
    if not values.shape[0]:
        continue

    values = values.iloc[0]
    rows.append({"name": file, "age": src_to_age[values["src"]]})

df_new = pd.DataFrame(rows)
df_new.to_pickle("ages_from_the_wild.pk")
df_new.to_csv("ages_from_the_wild.csv")
stratify_on = df_new["age"]
df_train, df_test = train_test_split(df_new, stratify=stratify_on, train_size=0.8,
                                     random_state=14)
df_train.to_csv("ages_from_the_wild_train.csv")
df_test.to_csv("ages_from_the_wild_test.csv")
