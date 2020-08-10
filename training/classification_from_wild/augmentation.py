import albumentations as A


def additional_augmenation_good(image):
    image = image.astype("uint8")
    comp = A.Compose(
        [
            # A.Resize(128, 128),
            # A.RandomCrop(120, 120),
            #A.Blur(blur_limit=(5, 7), p=p_small),
            #A.RandomBrightness(p=p_small, limit=(-1, 1)),
            #A.JpegCompression(quality_lower=2,quality_upper=10,p=p_small),
            #A.GaussNoise(var_limit=10000, p=p_small),
            #A.RandomSunFlare(p=p_small),
            #A.Downscale(p=0.05),
            #A.CLAHE(p=0.05),
            #A.RandomContrast(p=0.05),
            #A.RandomBrightness(p=0.05),
            A.HorizontalFlip(),
            #A.VerticalFlip(),
            #A.RandomRotate90(),
            A.ShiftScaleRotate(
                shift_limit=0.02, scale_limit=0.02, rotate_limit=10, p=0.3
            ),
            #A.Blur(blur_limit=2, p=0.05),
            #A.OpticalDistortion(p=0.05),
            #A.GridDistortion(p=0.05),
            #A.ChannelShuffle(p=0.05),
            #A.HueSaturationValue(p=0.05),
            #A.ElasticTransform(),
            #A.ToGray(p=0.05),
            #A.JpegCompression(p=0.05),
            #A.MedianBlur(p=0.05),
            #A.Cutout(p=0.05),
            #A.RGBShift(p=0.05),
            #A.GaussNoise(var_limit=(0, 50), p=0.05),
            A.Normalize(),
        ],
        p=1,
    )
    return comp(image=image)["image"]

def additional_augmenation_bad(image):
    image = image.astype("uint8")
    p_small = 0.005
    comp = A.Compose(
        [
            # A.Resize(128, 128),
            # A.RandomCrop(120, 120),
            A.Blur(blur_limit=(5, 7), p=p_small),
            A.RandomBrightness(p=p_small, limit=(-1, 1)),
            A.JpegCompression(quality_lower=2,quality_upper=10,p=p_small),
            A.GaussNoise(var_limit=10000, p=p_small),
            #A.RandomSunFlare(p=p_small),
            A.Downscale(p=p_small),
            A.CLAHE(p=0.05),
            #A.RandomContrast(p=0.05),
            #A.RandomBrightness(p=0.05),
            A.HorizontalFlip(),
            #A.VerticalFlip(),
            #A.RandomRotate90(),
            A.ShiftScaleRotate(
                shift_limit=0.08, scale_limit=0.08, rotate_limit=30, p=0.3
            ),
            #A.Blur(blur_limit=2, p=0.05),
            #A.OpticalDistortion(p=0.05),
            #A.GridDistortion(p=0.05),
            #A.ChannelShuffle(p=0.05),
            #A.HueSaturationValue(p=0.05),
            #A.ElasticTransform(),
            A.ToGray(p=p_small),
            #A.JpegCompression(p=0.05),
            #A.MedianBlur(p=0.05),
            #A.Cutout(p=0.05),
            #A.RGBShift(p=0.05),
            #A.GaussNoise(var_limit=(0, 50), p=0.05),
            A.Normalize(),
        ],
        p=1,
    )
    return comp(image=image)["image"]

def additional_augmenation_val(image):
    image = image.astype("uint8")
    comp = A.Compose([#A.Resize(128, 128),
                      #A.RandomCrop(120, 120),
                      A.Normalize()], p=1)
    return comp(image=image)["image"]


def get_augmentation():
    return {
        "train": {0: additional_augmenation_bad, 1: additional_augmenation_good,},
        "valid": additional_augmenation_val,
    }
