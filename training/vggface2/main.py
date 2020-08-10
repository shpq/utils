import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset
import timm
import face_detection
import glob
import os
import cv2
from torch.utils import data
from PIL import Image
import matplotlib.image as mpimg
from tqdm import tqdm
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2
from random import shuffle
torch.nn.Module.dump_patches = False
class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, device, s=30.0, m=0.50):
        super(ArcFace, self).__init__()
        self.device = device
        self.s = s
        self.m = m
        self.cos_m = np.cos(m)
        self.sin_m = np.sin(m)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features), requires_grad=True).cuda()
        nn.init.xavier_uniform_(self.weight)

    def forward(self, feature, label):
        feature_normalized = F.normalize(feature.view(feature.shape[0], -1))
        cos_th = F.linear(feature_normalized, F.normalize(self.weight))
        sin_th = torch.sqrt((1.0 - torch.pow(cos_th, 2)).clamp(0, 1))
        cos_th_plus_m = cos_th * self.cos_m - sin_th * self.sin_m
        one_hot = torch.zeros(cos_th.size(), device=self.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * cos_th_plus_m) + ((1.0 - one_hot) * cos_th)
        output *= self.s
        return output

def load_model(saved, checkpoint_path):
    #model = torchvision.models.resnet18()
    if saved is None:
        model = timm.create_model('efficientnet_b0', pretrained=True)
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
    else:
        print(f"open saved: {os.path.join(checkpoint_path, saved)}")
        model = torch.load(os.path.join(checkpoint_path, saved))
    return model

def extract_faces(data_path, face_path, batch_size_det):
    detector = face_detection.build_detector('RetinaNetMobileNetV1', confidence_threshold=0.83, nms_iou_threshold=0.08 )
    train_dir = 'train'
    test_dir = 'test'
    max_size = 500.0
    print('Create directories for faces...')
    for dir_name in [train_dir, test_dir]:
        print(dir_name)
        class_names = os.listdir(os.path.join(data_path, dir_name))
        for class_name in tqdm(class_names, total=len(class_names)):
            if ".ipynb_checkpoints" in class_name:
                continue
            #if '666' not in class_name:
            #    continue
            os.makedirs(os.path.join(face_path, dir_name, class_name), exist_ok=True)
    def split_list(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]
    print('Start face extracting and saving into new directory...')
    for dir_name in [train_dir, test_dir]:
        print(dir_name)
        class_names = os.listdir(os.path.join(data_path, dir_name))
        for class_name in tqdm(class_names, total=len(class_names)):
            file_names = os.listdir(os.path.join(data_path, dir_name, class_name))
            file_names = [x for x in file_names if ".ipynb_checkpoints" not in x]
            for batch in split_list(file_names, n=batch_size_det):
                paths = []
                pics = []
                pics_numpy = []
                batch_pics = []
                for infile in batch:
                    file, ext = os.path.splitext(infile)
                    pic_path = os.path.join(data_path, dir_name, class_name, infile)
                    pic = Image.open(pic_path)
                    paths.append(pic_path)
                    pics.append(pic)
                    pics_numpy.append(np.array(pic))
                batch_h = int(min(np.quantile([pic.shape[0] for pic in pics_numpy], q=0.9), 500))
                batch_w = int(min(np.quantile([pic.shape[1] for pic in pics_numpy], q=0.9), 500))
                pics_resized = [pic.resize((batch_h, batch_w)) for pic in pics]
                batch_pics = [np.array(pic) for pic in pics_resized]
                batch_pics = np.array(batch_pics)
                detections = detector.batched_detect(batch_pics)
                for detection, pic, path in zip(detections, pics_resized, paths):
                    try:
                        #x1, y1, x2, y2 = [max(int(x), 0) for x in detector.detect(np.array(pic))[0][0:4]]
                        bounds = [[max(int(x), 0) for x in res[0:4]] for res in detection]
                        x1, y1, x2, y2 = max(bounds, key=lambda b: (b[3] - b[1]) * (b[2] - b[0]))
                    except (IndexError, ValueError) as e:
                        print(e)
                        continue
                    face = np.array(pic)[y1:y2, x1:x2]
                    face = Image.fromarray(face)
                    file_parsed = path.split(os.sep)
                    face_name = file_parsed[-1]
                    person_name = file_parsed[-2]
                    save_path = os.path.join(face_path, dir_name, person_name, 'face_' + face_name)
                    face.save(save_path, "JPEG")
class AlbuDataset(data.Dataset):
    def __init__(self, img_dir, transforms=None,):
        self.img_dir = img_dir
        self.targets = os.listdir(img_dir)
        self.instances = []
        for cl in self.targets:
            path_to_image = os.listdir(os.path.join(img_dir, cl))
            # print(print(path_to_image[:4]))
            for n in path_to_image:
                if ".ipynb" in n:
                    continue
                instance = os.path.join(img_dir, cl, n), cl
                self.instances.append(instance)
        shuffle(self.instances)
        self.targets_to_idx = {t : i for i, t in enumerate(sorted(self.targets))}
        self.transform = transforms
    def __getitem__(self, index):
        try:
            img_path, label = self.instances[index]
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = image
        except Exception as e:
            print(img_path)
            raise
        if self.transform is not None:
            # img = self.transform(image=np.swapaxes(
            #    np.swapaxes(np.array(img), 2, 0), 1, 2))["image"]
            if isinstance(self.transform, dict):
                augmented = self.transform[label](image=image)["image"]
            else:
                augmented = self.transform(image=image)["image"]
        return augmented, torch.tensor(self.targets_to_idx[label])
    def __len__(self):
        return len(self.instances)
def train_model(model, loss, metric, optimizer, scheduler, device, train_dataloader, val_dataloader, batch_size, num_epochs, checkpoint_path):
    pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    for epoch in range(num_epochs):
        print('Epoch {}/{}:'.format(epoch, num_epochs - 1), flush=True)
        #pbar.set_description('Epoch {}/{}:'.format(epoch, num_epochs - 1))
        for phase in ['train', 'val']:
            if phase == 'train':
                dataloader = train_dataloader
                scheduler.step()
                model.train() 
            else:
                dataloader = val_dataloader
                model.eval()   
            pbar = tqdm(enumerate(dataloader), total=len(dataloader))
            running_loss = 0.
            running_acc = 0.
            scale_value = 0.
            for idx, (inputs, labels) in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    feature = model(inputs)
                    preds = metric(feature, labels)
                    loss_value = loss(preds, labels)
                    preds_class = preds.argmax(dim=1)
                    if phase == 'train':
                        loss_value.backward()
                        optimizer.step()
                running_loss += loss_value.item()
                running_acc += (preds_class == labels.data).float().mean()
                scale_value = 1 / (idx + 1)
                running_norm_loss = running_loss * scale_value
                running_norm_acc = running_acc * scale_value
                pbar.set_description("{} Loss: {:.4f} Acc: {:.4f}".format(phase, running_norm_loss, running_norm_acc))
            if phase == 'train':
                pass
            else:
                model_name = 'vggface2_{}_{:.4f}_{:.4f}.h5'.format(epoch, running_norm_loss, running_norm_acc)
                torch.save(model, os.path.join(checkpoint_path, model_name))

if __name__ == "__main__":
    #python main.py --data_path "C:\\Users\\Mnfst\\Age Prediction Expansion\\feature_extractor\\vggface2" --face_path "C:\\Users\\Mnfst\\Age Prediction Expansion\\feature_extractor\\faces" --checkpoint_path "C:\\Users\\Mnfst\\Age Prediction Expansion\\feature_extractor\\checkpoint" --batch_size 100
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('--data_path', help='Image directory')
    parser.add_argument('--face_path', help='New empty directory for extracted faces')
    parser.add_argument('--checkpoint_path', help='Checkpoints directory')
    parser.add_argument('--batch_size_det', help='Batch size for face detection')
    parser.add_argument('--saved', help='Use saved checkpoint', default=None)
    args = parser.parse_args()
    data_path = args.data_path
    face_path = args.face_path
    checkpoint_path = args.checkpoint_path
    batch_size_det = int(args.batch_size_det)
    saved = args.saved
    if not os.path.isdir(face_path):
        extract_faces(data_path, face_path, batch_size_det)

    resize = 128
    crop_size = 120
    p_small = 0.005
    p_big = 0.5
    transforms = A.Compose(
        [
            A.Resize(resize, resize),
            A.RandomCrop(crop_size, crop_size),
            A.Blur(blur_limit=7, p=p_small),
            A.RandomBrightness(p=p_small, limit=(-0.2, 0.2)),
            A.JpegCompression(quality_lower=10,quality_upper=25,p=p_small),
            A.GaussNoise(var_limit=1000, p=p_small),
            #A.RandomSunFlare(p=p_small),
            A.Downscale(p=p_small, scale_min=0.25, scale_max=0.8),
            #A.CLAHE(p=p_small),
            #A.RandomContrast(p=0.05),
            #A.RandomBrightness(p=0.05),
            #A.HorizontalFlip(),
            #A.VerticalFlip(p=p_small),
            #A.RandomRotate90(),
            A.ShiftScaleRotate(shift_limit = 0,scale_limit=0, rotate_limit=180, p=p_big),
            #A.Blur(blur_limit=2, p=0.05),
            #A.OpticalDistortion(p=0.05),
            #A.GridDistortion(p=0.05),
            #A.ChannelShuffle(p=0.05),
            #A.HueSaturationValue(p=0.05),
            #A.ElasticTransform(),
            A.ToGray(p=p_small),
            #A.JpegCompression(p=0.05),
            #A.MedianBlur(p=0.05),
            A.Cutout(max_h_size=10,num_holes=2, max_w_size=10, p=p_small),
            A.Normalize(),
            ToTensorV2(),
                ],
                p=1,
        )
    """transforms = T.Compose([
            T.Resize((resize, resize)),
            T.RandomCrop((crop_size, crop_size)),
            #T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
    ])"""
    dataset_train = AlbuDataset(os.path.join(face_path, 'train'), transforms)
    dataset_test = AlbuDataset(os.path.join(face_path, 'test'), transforms)
    batch_size = 512
    train_dataloader = torch.utils.data.DataLoader(
    dataset_train, shuffle=True, batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(
    dataset_test, shuffle=True, batch_size=batch_size)
    model = load_model(saved, checkpoint_path)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    class_num = len(set(dataset_train.targets))
    loss = torch.nn.CrossEntropyLoss()
    metric = ArcFace(in_features=1280, out_features=class_num, device=device)    
    params = list(model.parameters()) + list(metric.parameters())
    optimizer = torch.optim.Adam(params, lr=1.0e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    model = model.to(device)
    num_epochs = 100
    train_model(model, loss, metric, optimizer, scheduler, device,
                train_dataloader, test_dataloader,
                batch_size, num_epochs, checkpoint_path)
