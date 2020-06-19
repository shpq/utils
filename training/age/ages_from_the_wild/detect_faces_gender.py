import cv2
import face_detection
import pandas as pd
import os
from multiprocessing import Pool
from tqdm import tqdm
import argparse
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    parser.add_argument(
        "--save",
        default=False,
        help="Save downloaded images?",
        action="store_true",
    )

    parser.add_argument(
        "--force",
        default=False,
        help="Force rewrite?",
        action="store_true",
    )
    
    parser.add_argument(
        "--batch_size",
        default=100,
        help="Batch size for detection?",
    )
    
    args = parser.parse_args()
    batch_size = args.batch_size
    print(face_detection.available_detectors)
    main_directory = os.path.dirname(__file__)

    detector = face_detection.build_detector(
        "RetinaNetResNet50", confidence_threshold=.95, nms_iou_threshold=.3)
    print(detector.device)
    df = pd.read_pickle("wild_gender_with_path.pk")
    try:
        df_prev = pd.read_pickle("wild_gender_with_detections.pk")
    except:
        df_prev = None

    def get_image(path):

        if path is None:
            return None

        name = path.split("/")[-1].split(".jpg")[0]
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        return cv2.resize(image, (640, 640))

    def get_batched_result(batch_):
        batch_images = np.asarray([np.array(x) for x in batch_.image.values])
        detections_batch = detector.batched_detect(batch_images)
        batch_['detections'] = detections_batch
        for i, row in batch_.iterrows():
            detections = row['detections']
            path = row['paths']
            image = row["image"]
            name = path.split("/")[-1].split(".jpg")[0]
            for ind, result in enumerate(detections):
                x1, y1, x2, y2 = result[0:4].astype(int)
                x1, y1, x2, y2 = max(x1, 0), max(y1, 0), max(x2, 0), max(y2, 0)
                new_path = main_directory + "cropped_images/" + \
                    name + "_" + str(ind + 1) + ".jpg"
                if y2-y1 < 50 or x2-x1 < 50:
                    continue

                cv2.imwrite(new_path, cv2.cvtColor(cv2.resize(
                    image[y1:y2, x1:x2], (128, 128)), cv2.COLOR_BGR2RGB), )
                
        return batch_

    def chunker(seq, size):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))


    def get_results(value):
        index, row = value
        path = row["paths"]
        if path is None:
            return None

        name = path.split("/")[-1].split(".jpg")[0]
        image = cv2.imread(path)
        detections = detector.detect(image)
        new_paths = []
        if args.save:
            for ind, result in enumerate(detections):
                x1, y1, x2, y2 = result[0:4].astype(int)
                x1, y1, x2, y2 = max(x1, 0), max(y1, 0), max(x2, 0), max(y2, 0)
                new_path = main_directory + "cropped_images/" + \
                    name + "_" + str(ind + 1) + ".jpg"
                if y2-y1 < 60 or x2-x1 < 60:
                    continue

                cv2.imwrite(new_path, cv2.resize(
                    image[y1:y2, x1:x2], (128, 128)))
                new_paths.append(new_path)

        return detections, new_paths

    if not args.force and df_prev is not None:
        print(f"len of df : {len(df)}")
        df = df[~df.url.isin(df_prev.url)]
        print(f"len of df after cleaning: {len(df)}")
    df['detections'] = None
    df.dropna(subset=['paths'], inplace=True)
    cols = ['text', 'url', 'src', 'paths']


    for batch in tqdm(chunker(df, batch_size), total = len(list(chunker(df, batch_size)))):
        batch["image"] = batch["paths"].apply(get_image)
        batch = get_batched_result(batch)
        df.loc[df.url.isin(batch.url), 'detections'] = batch['detections']


    if df_prev is not None:
        df_prev = df_prev.append(df).reset_index(drop=True)
        df_prev.to_pickle("wild_gender_with_detections.pk")
        df_prev.to_csv("wild_gender_with_detections.csv")
    else:
        df.to_pickle("wild_gender_with_detections.pk")
        df.to_csv("wild_gender_with_detections.csv")
