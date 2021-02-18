import albumentations as A
import numpy as np
from random import choice, random


class Augmentation:
    def __init__(self, cfg):
        self.size = tuple(cfg.training.size)

    def augment_saving_beauty(self, image):
        image = np.array(image).astype("uint8")
        compose_list = [
            A.Resize(*self.size),
            A.HorizontalFlip(),
            A.ShiftScaleRotate(
                shift_limit=0.02, scale_limit=0.02, rotate_limit=5
            ),
        ]
        comp = A.Compose(compose_list,  p=1,)
        norm = A.Compose(
            [A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))], p=1)
        image_not_norm = comp(image=image)["image"]
        return image_not_norm, norm(image=image_not_norm)["image"]

    def augment_reducing_beauty(self, image):
        image = np.array(image).astype("uint8")
        strong_augs = [
            # A.RandomBrightness(p=1, limit=(-0.55, -0.1)),
            # A.RandomBrightness(p=1, limit=(0.1, 0.3)),
            A.Blur(p=1, blur_limit=(4, 7)),
            A.MotionBlur(blur_limit=(16, 25), p=1),
            A.JpegCompression(p=1, quality_lower=5, quality_upper=25,),
            A.ImageCompression(p=1, quality_lower=5, quality_upper=25, compression_type=0),
            A.GaussNoise(p=1, var_limit=(1e2, 1e3)),
            A.MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True, per_channel=True, p=1),
            A.GaussNoise(p=1, var_limit=(1e3, 5e3)),
            A.Downscale(p=1, scale_min=0.2, scale_max=0.5),
            # A.RandomContrast(p=1, limit=(-0.6, -0.2)),
            # A.RandomContrast(p=1, limit=(0.2, 0.6)),
            # A.GridDistortion(p=1, num_steps=15, distort_limit=0.7),
        ]
        compose_list = [
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), p=1),
        ]
        compose_list.insert(0, choice(strong_augs))

        # while random() < 0.2:
            # compose_list.insert(0, choice(strong_augs))
        # print(compose_list)
        comp = A.Compose(compose_list, p=1)
        return comp(image=image)["image"]
