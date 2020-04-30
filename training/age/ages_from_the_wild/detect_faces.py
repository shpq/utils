import cv2
import face_detection
import pandas as pd
import os
from multiprocessing import Pool
from tqdm import tqdm
import argparse


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
        help="Save downloaded images?",
        action="store_true",
    )
    args = parser.parse_args()

    print(face_detection.available_detectors)
    main_directory = os.path.dirname(__file__)

    detector = face_detection.build_detector(
        "RetinaNetResNet50", confidence_threshold=.95, nms_iou_threshold=.3)

    df = pd.read_pickle("wild_ages_with_path.pk")
    try:
        df_prev = pd.read_pickle("wild_ages_with_detections.pk")
    except:
        df_prev = None

    def get_results(value):
        index, row = value
        path = row["paths"]
        if path is None:
            return None

        name = path.split("/")[-1].split(".jpg")[0]
        image = cv2.imread(path)
        detections = detector.detect(image)
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

        return detections
    if not args.force and df_prev is not None:
        print(f"len of df : {len(df)}")
        df = df[~df.url.isin(df_prev.url)]
        print(f"len of df after cleaning: {len(df)}")

    r = list(tqdm(map(get_results, df.iterrows()), total=len(df)))
    df["detections"] = r
    df_prev = df_prev.append(df).reset_index(drop=True)
    df.to_pickle("wild_ages_with_detections.pk")
    df.to_csv("wild_ages_with_detections.csv")
