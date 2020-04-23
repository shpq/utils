import albumentations as A


def additional_augmenation(image):
    image = image.astype("uint8")
    comp = A.Compose(
        [
            A.CLAHE(p=0.05),
            A.RandomContrast(p=0.05),
            A.RandomBrightness(p=0.05),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
            A.Transpose(),
            A.ShiftScaleRotate(
                shift_limit=0.08, scale_limit=0.08, rotate_limit=20, p=1
            ),
            # A.Blur(blur_limit=2, p=0.05),
            A.OpticalDistortion(p=0.05),
            A.GridDistortion(p=0.05),
            # A.ChannelShuffle(p=0.05),
            # A.HueSaturationValue(p=0.05),
            # A.ElasticTransform(),
            A.ToGray(p=0.05),
            A.JpegCompression(p=0.05),
            # A.MedianBlur(p=0.05),
            # A.RGBShift(p=0.05),
            A.GaussNoise(var_limit=(0, 50), p=0.05),
            A.Normalize(),
        ],
        p=1,
    )
    return comp(image=image)["image"]


def additional_augmenation_val(image):
    image = image.astype("uint8")
    comp = A.Compose([A.Normalize()], p=1)
    return comp(image=image)["image"]


def get_augmentation():
    return {
        "train": {0: additional_augmenation, 1: additional_augmenation,},
        "valid": additional_augmenation_val,
    }
