import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import sys
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensor
import cv2
import timm


torch.backends.cudnn.deterministic = True
main_directory = os.path.dirname(__file__)
TRAIN_CSV_PATH = main_directory + "ages_from_the_wild_true_cleared_stratify_reduced_train_v5.csv"
TEST_CSV_PATH = main_directory + "ages_from_the_wild_true_cleared_stratify_reduced_test_v5.csv"
IMAGE_PATH = main_directory + "ages_from_the_wild/cropped_images"


# Argparse helper

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", type=int, default=-1)

parser.add_argument("--seed", type=int, default=-1)

parser.add_argument("--numworkers", type=int, default=3)

parser.add_argument("--outpath", type=str, required=True)

parser.add_argument("--imp_weight", type=int, default=0)

parser.add_argument("--batch_size", type=int, default=0)

parser.add_argument("--lr", type=float, default=0)

parser.add_argument("--pretrained", type=str, default="seresnext26t_32x4d")

parser.add_argument("--epoch_reduce", type=int, default=120)

parser.add_argument("--gamma", type=float, default=0.7)

parser.add_argument("--alpha", type=float, default=0.9)

parser.add_argument("--epoch_clean", type=int, default=30)

parser.add_argument("--clean_diff", type=float, default=10.0)

parser.add_argument("--saved", type=str, default="")

args = parser.parse_args()

NUM_WORKERS = args.numworkers

if args.cuda >= 0:
    DEVICE = torch.device("cuda:%d" % args.cuda)
else:
    DEVICE = torch.device("cpu")

if args.seed == -1:
    RANDOM_SEED = None
else:
    RANDOM_SEED = args.seed

IMP_WEIGHT = args.imp_weight

MODEL_NAME = args.pretrained


PATH = args.outpath
if not os.path.exists(PATH):
    os.mkdir(PATH)
LOGFILE = os.path.join(PATH, "training.log")
TEST_PREDICTIONS = os.path.join(PATH, "test_predictions.log")
TEST_ALLPROBAS = os.path.join(PATH, "test_allprobas.tensor")

# Logging

header = []

header.append("PyTorch Version: %s" % torch.__version__)
header.append("CUDA device available: %s" % torch.cuda.is_available())
header.append("Using CUDA device: %s" % DEVICE)
header.append("Random Seed: %s" % RANDOM_SEED)
header.append("Task Importance Weight: %s" % IMP_WEIGHT)
header.append("Output Path: %s" % PATH)
header.append("Script: %s" % sys.argv[0])

with open(LOGFILE, "w") as f:
    for entry in header:
        print(entry)
        f.write("%s\n" % entry)
        f.flush()


##########################
# SETTINGS
##########################

# Hyperparameters
learning_rate = args.lr
num_epochs = 200

# Architecture
BATCH_SIZE = args.batch_size
GRAYSCALE = False

df = pd.read_csv(TRAIN_CSV_PATH, index_col=0)
ages = df["gender"].values
del df
ages = torch.tensor(ages, dtype=torch.float)


def task_importance_weights(label_array):
    uniq = torch.unique(label_array)
    num_examples = label_array.size(0)

    m = torch.zeros(uniq.shape[0])

    for i, t in enumerate(torch.arange(torch.min(uniq), torch.max(uniq))):
        m_k = torch.max(
            torch.tensor(
                [
                    label_array[label_array > t].size(0),
                    num_examples - label_array[label_array > t].size(0),
                ]
            )
        )
        m[i] = torch.sqrt(m_k.float())

    imp = m / torch.max(m)
    return imp


# Data-specific scheme
if not IMP_WEIGHT:
    imp = torch.ones(NUM_CLASSES - 1, dtype=torch.float)
elif IMP_WEIGHT == 1:
    imp = task_importance_weights(ages)
    imp = imp[0: NUM_CLASSES - 1]
else:
    raise ValueError("Incorrect importance weight parameter.")
    
train_imp = torch.ones(NUM_CLASSES - 1, dtype=torch.float)
imp = imp.to(DEVICE)
train_imp = train_imp.to(DEVICE)



class UTKFaceDataset(Dataset):
    """Custom Dataset for loading UTKFace face images"""

    def __init__(self, csv_path, img_dir, transform=None, clean=False):

        df = pd.read_csv(csv_path, index_col=0)
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.img_names = df["name"].values
        self.y = df["age"].values
        self.transform = transform

    def __getitem__(self, index):
        # img = Image.open(os.path.join(self.img_dir, self.img_names[index]))
        image = cv2.imread(os.path.join(self.img_dir, self.img_names[index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            # img = self.transform(image=np.swapaxes(
            #    np.swapaxes(np.array(img), 2, 0), 1, 2))["image"]
            augmented = self.transform(image=image)
            img = augmented["image"]

        label = self.y[index]
        levels = [1] * label + [0] * (NUM_CLASSES - 1 - label)
        levels = torch.tensor(levels, dtype=torch.float32)
        return img, label, levels

    def __len__(self):
        return self.y.shape[0]


custom_transform = A.Compose(
    [
        A.Resize(128, 128),
        A.RandomCrop(120, 120),
        # A.CLAHE(p=0.05),
        A.RandomContrast(p=0.5, limit=0.05),
        A.RandomBrightness(p=0.5, limit=0.05),
        A.HorizontalFlip(),
        A.Transpose(),
        A.ShiftScaleRotate(
            shift_limit=0.12, scale_limit=0.12, rotate_limit=30, p=1
        ),
        A.Blur(p=0.05, blur_limit=3),
        #A.OpticalDistortion(p=0.05),
        #A.GridDistortion(p=0.05),
        #A.HueSaturationValue(p=0.1),
        # A.ElasticTransform(),
        A.ToGray(p=0.1),
        A.JpegCompression(quality_lower=45,quality_upper=100,p=0.05),
        #A.MedianBlur(p=0.5),
        #A.RGBShift(p=0.1),
        A.GaussNoise(var_limit=(100, 3000), p=0.05),
        A.Normalize(),
        ToTensor(),
    ]
)

train_dataset = UTKFaceDataset(
    csv_path=TRAIN_CSV_PATH, img_dir=IMAGE_PATH, transform=custom_transform
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)

custom_transform2 = A.Compose(
    [
        A.Resize(128, 128),
        A.CenterCrop(120, 120),
        A.Normalize(),
        ToTensor(),
    ]
)

test_dataset = UTKFaceDataset(
    csv_path=TEST_CSV_PATH, img_dir=IMAGE_PATH, transform=custom_transform2
)


test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)


##########################
# MODEL
##########################


class CustomModel(nn.Module):
    def __init__(self, model_name, num_classes):
        super(CustomModel, self).__init__()
        back_model = timm.create_model(
            model_name, pretrained=True, num_classes=1
        )
        self.num_classes = num_classes
        self.back = nn.Sequential(*list(back_model.children())[:-1])
        self.fc = list(back_model.children())[-1]
        self.linear_1_bias = nn.Parameter(
            torch.zeros(self.num_classes - 1).float()
        )

    def forward(self, x):
        x = self.back(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        logits = logits + self.linear_1_bias
        probas = torch.sigmoid(logits)
        return logits, probas


###########################################
# Initialize Cost, Model, and Optimizer
###########################################


def cost_fn(logits, levels, imp):
    val = -torch.sum(
        (
            F.logsigmoid(logits) * levels
            + (F.logsigmoid(logits) - logits) * (1 - levels)
        )
        * imp,
        dim=1,
    )
    return torch.mean(val)


torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
if args.saved != "":
    print(f"opening saved: {args.saved}")
    model = torch.load(PATH + "/" + args.saved)
else:
    model = CustomModel(MODEL_NAME, NUM_CLASSES)

model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.StepLR(
    optimizer, step_size=args.epoch_reduce, gamma=args.gamma
)


def compute_mae_and_mse(model, data_loader, device):
    mae, mse, num_examples = 0, 0, 0
    list_preds = []
    for i, (features, targets, levels) in enumerate(data_loader):

        features = features.to(device)
        targets = targets.to(device)

        logits, probas = model(features)
        predict_levels = probas > 0.5
        predicted_labels = torch.sum(predict_levels, dim=1)
        list_preds += [int(x) for x in predicted_labels]
        num_examples += targets.size(0)
        predicted_labels = torch.log1p(predicted_labels.float())
        targets  = torch.log1p(targets.float())
        mae += torch.sum(torch.abs(predicted_labels - targets))
        mse += torch.sum((predicted_labels - targets) ** 2)
    mae = mae.float() / num_examples
    mse = mse.float() / num_examples
    return mae, mse, list_preds


def clear_columns(cols, init=False):
    if init:
        return [x for x in cols if "unname" not in x.lower() and "pred" not in x.lower()]
    return [x for x in cols if "unname" not in x.lower()]


df_train = pd.read_csv(TRAIN_CSV_PATH)
df_train = df_train[clear_columns(df_train.columns, init=True)]
df_train[f"age_pred_0"] = df_train["age"]
df_train.to_csv(TRAIN_CSV_PATH)
df_test = pd.read_csv(TEST_CSV_PATH)
df_test = df_test[clear_columns(df_test.columns, init=True)]
df_test[f"age_pred_0"] = df_test["age"]
df_test.to_csv(TEST_CSV_PATH)
loaders = {"train": train_loader, "test": test_loader}
start_time = time.time()
alpha = args.alpha
for epoch in range(num_epochs):
    for phase in ["train", "test"]:
        if phase == "train":
            model.train()
        else:
            model.eval()
        running_cost = 0.0
        loader = loaders[phase]
        pbar = tqdm(enumerate(loader), total=len(loader))

        for batch_idx, (features, targets, levels) in pbar:
            scale_value = 1 / BATCH_SIZE / max(batch_idx, 1)
            pbar.set_description(
                (
                    "Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f"
                    % (
                        epoch + 1,
                        num_epochs,
                        batch_idx,
                        len(train_dataset) // BATCH_SIZE,
                        running_cost * scale_value,
                    )
                )
            )
            features = features.to(DEVICE)
            targets = targets
            targets = targets.to(DEVICE)
            levels = levels.to(DEVICE)
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == "train"):
                # FORWARD AND BACK PROP
                logits, probas = model(features)
                if phase == "train":
                    cost = cost_fn(logits, levels, imp)
                else:
                    cost = cost_fn(logits, levels, train_imp)

                if phase == "train":
                    cost.backward()

                    # UPDATE MODEL PARAMETERS
                    optimizer.step()

                running_cost += cost.item() * features.size(0)
        if phase == "train":
            scheduler.step()
    with torch.set_grad_enabled(False):  # save memory during inference

        train_mae, train_mse, list_preds_train = compute_mae_and_mse(
            model, train_loader, device=DEVICE
        )
        test_mae, test_mse, list_preds_test = compute_mae_and_mse(
            model, test_loader, device=DEVICE)

        s = "MAE/RMSE: | Train: %.2f/%.2f | Test: %.2f/%.2f" % (
            train_mae,
            torch.sqrt(train_mse),
            test_mae,
            torch.sqrt(test_mse),
        )
        df_train = pd.read_csv(TRAIN_CSV_PATH)
        df_train[f"age_pred_{epoch+1}"] = None
        df_train[f"age_true_pred_{epoch+1}"] = None
        print(f"len np.array(list_preds_train) : {np.array(list_preds_train).shape}")
        if epoch > args.epoch_clean:
            df_train.loc[clean_dataset(df_train["age"], df_train[f"age_pred_{epoch}"]), f"age_pred_{epoch+1}"] = alpha*df_train.loc[clean_dataset(df_train["age"], df_train[f"age_pred_{epoch}"]), f"age_pred_{epoch}"] + (1-alpha)*np.array(list_preds_train)
            df_train.loc[clean_dataset(df_train["age"], df_train[f"age_pred_{epoch}"]), f"age_true_pred_{epoch+1}"] = np.array(list_preds_train)
            df_train[clear_columns(df_train.columns)].to_csv(TRAIN_CSV_PATH)
            df_test = pd.read_csv(TEST_CSV_PATH)

            ages = torch.tensor(df_train.loc[clean_dataset(df_train["age"], df_train[f"age_pred_{epoch+1}"]), f"age"].values, dtype=torch.float)
        else:
            df_train.loc[:, f"age_pred_{epoch+1}"] = alpha*df_train.loc[:, f"age_pred_{epoch}"] + (1-alpha)*np.array(list_preds_train)
            df_train.loc[:, f"age_true_pred_{epoch+1}"] = np.array(list_preds_train)
            df_train[clear_columns(df_train.columns)].to_csv(TRAIN_CSV_PATH)
            df_test = pd.read_csv(TEST_CSV_PATH)

            ages = torch.tensor(df_train.loc[:, f"age"].values, dtype=torch.float)
        if not IMP_WEIGHT:
            imp = torch.ones(NUM_CLASSES - 1, dtype=torch.float)
        elif IMP_WEIGHT == 1:
            imp = task_importance_weights(ages)
            imp = imp[0: NUM_CLASSES - 1]
        else:
            raise ValueError("Incorrect importance weight parameter.")
        imp = imp.to(DEVICE)
        df_test[f"age_pred_{epoch+1}"] = list_preds_test
        
        df_test[clear_columns(df_test.columns)].to_csv(TEST_CSV_PATH)
        if epoch + 1 > args.epoch_clean:
            train_dataset = UTKFaceDataset(
                csv_path=TRAIN_CSV_PATH, img_dir=IMAGE_PATH, transform=custom_transform, clean=True
            )

            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
            )
            loaders["train"] = train_loader
        print(s)
        train_mae = round(float(train_mae), 4)
        test_mae = round(float(test_mae), 4)
        name = f"age_wild_{epoch + 1}_{MODEL_NAME}_{train_mae}_vs_{test_mae}.h5"
        name = PATH + "/" + name
        torch.save(model, name)
