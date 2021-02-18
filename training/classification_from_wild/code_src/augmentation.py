import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from random import choice

p_small = 0.02


def additional_augmenation_good(image, size, mode="torch"):
    image = np.array(image).astype("uint8")
    compose_list = [
        A.Resize(*size),
        # A.RandomCrop(120, 120),
        # A.Blur(blur_limit=(4, 6), p=p_small),
        # A.RandomBrightness(p=p_small, limit=(-0.5, 0.5)),
        # A.JpegCompression(quality_lower=35, quality_upper=70, p=p_small),
        # A.GaussNoise(var_limit=1000, p=p_small),
        # A.RandomSunFlare(p=p_small),
        # A.Downscale(p=p_small),
        # A.CLAHE(p=0.05),
        # A.RandomContrast(p=0.05),
        # A.RandomBrightness(p=0.05),
        A.HorizontalFlip(),
        # A.VerticalFlip(),
        # A.RandomRotate90(),
        A.ShiftScaleRotate(
            shift_limit=0.12, scale_limit=0.12, rotate_limit=20
        ),
        # A.Blur(blur_limit=2, p=0.05),
        # A.OpticalDistortion(p=0.05),
        # A.GridDistortion(p=1, num_steps=12, distort_limit=0.7),
        # A.ChannelShuffle(p=0.05),
        # A.HueSaturationValue(p=0.05),
        # A.ElasticTransform(),
        # A.ToGray(p=p_small),
        # A.JpegCompression(p=0.05),
        # A.MedianBlur(p=0.05),
        # A.Cutout(p=0.05),
        # A.RGBShift(p=p_small),
        # A.GaussNoise(var_limit=(0, 50), p=0.05),


    ]
    if mode == "torch":
        compose_list += [A.Normalize(), ToTensorV2()]
    elif mode == "keras":
        compose_list += [A.Normalize(mean=(0.5, 0.5, 0.5),
                                     std=(0.5, 0.5, 0.5), p=1)]
    else:
        raise NotImplementedError
    comp = A.Compose(compose_list,  p=1,)
    return comp(image=image)["image"]


def additional_augmenation_bad(image, size, mode="torch"):
    image = np.array(image).astype("uint8")
    compose_list = [
        A.Resize(*size),
        # A.RandomCrop(120, 120),
        # A.Blur(blur_limit=(4, 6), p=p_small),
        # A.RandomBrightness(p=p_small, limit=(-0.5, 0.5)),
        # A.JpegCompression(quality_lower=35, quality_upper=70, p=p_small),
        # A.GaussNoise(var_limit=1000, p=p_small),
        # A.RandomSunFlare(p=p_small),
        # A.Downscale(p=p_small),
        # A.CLAHE(p=0.05),
        # A.RandomContrast(p=0.05),
        # A.RandomBrightness(p=0.05),
        A.HorizontalFlip(),
        # A.VerticalFlip(),
        # A.RandomRotate90(),
        A.ShiftScaleRotate(
            shift_limit=0.12, scale_limit=0.12, rotate_limit=20
        ),
        # A.Blur(blur_limit=2, p=0.05),
        # A.OpticalDistortion(p=0.05),
        # A.GridDistortion(p=1, num_steps=12, distort_limit=0.7),
        # A.ChannelShuffle(p=0.05),
        # A.HueSaturationValue(p=0.05),
        # A.ElasticTransform(),
        # A.ToGray(p=p_small),
        # A.JpegCompression(p=0.05),
        # A.MedianBlur(p=0.05),
        # A.Cutout(p=0.05),
        # A.RGBShift(p=p_small),
        #A.GaussNoise(var_limit=(0, 50), p=0.05),
    ]
    if mode == "torch":
        compose_list += [A.Normalize(), ToTensorV2()]
    elif mode == "keras":
        compose_list += [A.Normalize(mean=(0.5, 0.5, 0.5),
                                     std=(0.5, 0.5, 0.5), p=1)]
    else:
        raise NotImplementedError
    comp = A.Compose(compose_list, p=1)
    return comp(image=image)["image"]


def additional_augmenation_val(image, size, mode="torch"):
    image = np.array(image).astype("uint8")
    compose_list = [A.Resize(*size),
                    # A.RandomCrop(120, 120),
                    # ToTensorV2(),
                    ]
    if mode == "torch":
        compose_list += [A.Normalize(), ToTensorV2()]
    elif mode == "keras":
        compose_list += [A.Normalize(mean=(0.5, 0.5, 0.5),
                                     std=(0.5, 0.5, 0.5), p=1)]
    else:
        raise NotImplementedError
    comp = A.Compose(compose_list, p=1)
    return comp(image=image)["image"]


def additional_augmenation_strong(image, size, mode="torch"):
    image = np.array(image).astype("uint8")
    strong_augs = [
        A.RandomBrightness(p=1, limit=(-0.7, -0.45)),
        A.RandomBrightness(p=1, limit=(0.35, 0.5)),
        A.Blur(p=1, blur_limit=(5, 7)),
        A.JpegCompression(p=1, quality_lower=8, quality_upper=15,),
        A.GaussNoise(p=1, var_limit=(8e2, 2e3)),
        A.Downscale(p=1, scale_min=0.2, scale_max=0.4),
        A.RandomContrast(p=1, limit=(-0.8, -0.6)),
        A.RandomContrast(p=1, limit=(1, 1.4)),
        A.GridDistortion(p=1, num_steps=15, distort_limit=0.7),
    ]
    compose_list = [
        A.Resize(*size),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.ShiftScaleRotate(
            shift_limit=0.15, scale_limit=0.15, rotate_limit=30
        ),
    ]
    compose_list.append(choice(strong_augs))

    if mode == "torch":
        compose_list += [A.Normalize(), ToTensorV2()]
    elif mode == "keras":
        compose_list += [A.Normalize(mean=(0.5, 0.5, 0.5),
                                     std=(0.5, 0.5, 0.5), p=1)]
    else:
        raise NotImplementedError

    comp = A.Compose(compose_list, p=1)
    return comp(image=image)["image"]


def get_augmentation():
    return {
        # "train": {0: additional_augmenation_bad, 1: additional_augmenation_good, -1: additional_augmenation_strong},
        "train": additional_augmenation_bad,
        "valid": additional_augmenation_val,
    }
