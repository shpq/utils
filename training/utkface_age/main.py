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

from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensor
import cv2


torch.backends.cudnn.deterministic = True
main_directory = os.path.dirname(__file__)
TRAIN_CSV_PATH = main_directory + "utkface_train.csv"
TEST_CSV_PATH = main_directory + "utkface_test.csv"
IMAGE_PATH = main_directory + "utkface/UTKFace_128x128"


# Argparse helper

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", type=int, default=-1)

parser.add_argument("--seed", type=int, default=-1)

parser.add_argument("--numworkers", type=int, default=3)

parser.add_argument("--outpath", type=str, required=True)

parser.add_argument("--imp_weight", type=int, default=0)

parser.add_argument("--batch_size", type=int, default=0)

parser.add_argument("--lr", type=float, default=0)

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
NUM_CLASSES = 91
BATCH_SIZE = args.batch_size
GRAYSCALE = False

df = pd.read_csv(TRAIN_CSV_PATH, index_col=0)
ages = df["age"].values
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
imp = imp.to(DEVICE)


###################
# Dataset
###################


class UTKFaceDataset(Dataset):
    """Custom Dataset for loading UTKFace face images"""

    def __init__(self, csv_path, img_dir, transform=None):

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
        A.CLAHE(p=0.05),
        A.RandomContrast(p=0.05),
        A.RandomBrightness(p=0.05),
        A.HorizontalFlip(),
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
        ToTensor(),
    ]
)

train_dataset = UTKFaceDataset(
    csv_path=TRAIN_CSV_PATH, img_dir=IMAGE_PATH, transform=custom_transform
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


train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
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


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, grayscale):
        self.num_classes = num_classes
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_dim, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1, padding=2)
        self.fc = nn.Linear(2048 * block.expansion, 1, bias=False)
        self.linear_1_bias = nn.Parameter(
            torch.zeros(self.num_classes - 1).float()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2.0 / n) ** 0.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        logits = logits + self.linear_1_bias
        probas = torch.sigmoid(logits)
        return logits, probas


def resnet34(num_classes, grayscale):
    """Constructs a ResNet-34 model."""
    model = ResNet(
        block=BasicBlock,
        layers=[3, 4, 6, 3],
        num_classes=num_classes,
        grayscale=grayscale,
    )
    return model


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
model = resnet34(NUM_CLASSES, GRAYSCALE)

model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def compute_mae_and_mse(model, data_loader, device):
    mae, mse, num_examples = 0, 0, 0
    for i, (features, targets, levels) in enumerate(data_loader):

        features = features.to(device)
        targets = targets.to(device)

        logits, probas = model(features)
        predict_levels = probas > 0.5
        predicted_labels = torch.sum(predict_levels, dim=1)
        num_examples += targets.size(0)
        mae += torch.sum(torch.abs(predicted_labels - targets))
        mse += torch.sum((predicted_labels - targets) ** 2)
    mae = mae.float() / num_examples
    mse = mse.float() / num_examples
    return mae, mse


loaders = {"train": train_loader, "test": test_loader}
start_time = time.time()
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
                cost = cost_fn(logits, levels, imp)

                if phase == "train":
                    cost.backward()

                    # UPDATE MODEL PARAMETERS
                    optimizer.step()
                running_cost += cost.item() * features.size(0)
    with torch.set_grad_enabled(False):  # save memory during inference

        train_mae, train_mse = compute_mae_and_mse(
            model, train_loader, device=DEVICE
        )
        test_mae, test_mse = compute_mae_and_mse(
            model, test_loader, device=DEVICE)

        s = "MAE/RMSE: | Train: %.2f/%.2f | Test: %.2f/%.2f" % (
            train_mae,
            torch.sqrt(train_mse),
            test_mae,
            torch.sqrt(test_mse),
        )

        print(s)

    all_pred = []
    all_probas = []
    with torch.set_grad_enabled(False):
        for batch_idx, (features, targets, levels) in enumerate(test_loader):

            features = features.to(DEVICE)
            logits, probas = model(features)
            all_probas.append(probas)
            predict_levels = probas > 0.5
            predicted_labels = torch.sum(predict_levels, dim=1)
            lst = [str(int(i)) for i in predicted_labels]
            all_pred.extend(lst)

    with open(TEST_PREDICTIONS, "w") as f:
        all_pred = ",".join(all_pred)
        f.write(all_pred)
