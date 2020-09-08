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
from mobilenetv2_quantized import MobileNetV2Q
from facenet_pytorch import MTCNN
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from skimage import transform as trans
import numpy as np
import os


torch.nn.Module.dump_patches = False


class FaceAligner:
    def __init__(self, mtcnn=MTCNN(select_largest=False, post_process=False, device='cuda', min_face_size=70, thresholds=[0.85, 0.9, 0.95],), desiredLeftEye=(0.3, 0.4),
                 desiredFaceWidth=120, desiredFaceHeight=180, max_size=500):
        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        self.mtcnn = mtcnn
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight

        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth

    def align(self, images):
        # convert the landmark (x, y)-coordinates to a NumPy array
        if not len(images):
            return []
        boxes, probs, landmarks = self.mtcnn.detect(images, landmarks=True)
        outputs = []
        for image, landmark, box, prob in zip(images, landmarks, boxes, probs):
            max_x, max_y, _ = image.shape
            if box is None:
                outputs.append(None)
                continue
            # add int round
            for b in box:
                b = [min(max(b[0], 0), max_x), min(max(b[1], 0), max_x), min(
                    max(b[2], 0), max_y), min(max(b[3], 0), max_y)]

            try:
                landmark = [x for _, x, _ in sorted(zip(box, landmark, prob), key=lambda x: np.abs(
                    (x[0][0] - x[0][2]) * (x[0][1] - x[0][3])) * x[2], reverse=True)]
            except Exception as e:
                print(e)
                outputs.append(None)
                continue

            leftEyeCenter = landmark[0][1]
            rightEyeCenter = landmark[0][0]

            # compute the angle between the eye centroids
            dY = rightEyeCenter[1] - leftEyeCenter[1]
            dX = rightEyeCenter[0] - leftEyeCenter[0]
            angle = np.degrees(np.arctan2(dY, dX)) - 180

            desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

            dist = np.sqrt((dX ** 2) + (dY ** 2))
            desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
            desiredDist *= self.desiredFaceWidth
            scale = desiredDist / dist

            # compute center (x, y)-coordinates (i.e., the median point)
            # between the two eyes in the input image
            eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
                          (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

            # grab the rotation matrix for rotating and scaling the face
            M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

            # update the translation component of the matrix
            tX = self.desiredFaceWidth * 0.5
            tY = self.desiredFaceHeight * self.desiredLeftEye[1]
            M[0, 2] += (tX - eyesCenter[0])
            M[1, 2] += (tY - eyesCenter[1])

            # apply the affine transformation
            (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
            output = cv2.warpAffine(np.array(image), M, (w, h),
                                    flags=cv2.INTER_CUBIC)
            outputs.append(output)

            # return the aligned face
        return outputs


class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, device, s=30.0, m=0.50):
        super(ArcFace, self).__init__()
        self.device = device
        self.s = s
        self.m = m
        self.cos_m = np.cos(m)
        self.sin_m = np.sin(m)
        self.weight = nn.Parameter(torch.FloatTensor(
            out_features, in_features), requires_grad=True).cuda()
        nn.init.xavier_uniform_(self.weight)

    def forward(self, feature, label):
        feature = feature.to(self.device)
        label = label.to(self.device)
        feature_normalized = F.normalize(feature.view(feature.shape[0], -1))
        cos_th = F.linear(feature_normalized, F.normalize(self.weight))
        sin_th = torch.sqrt((1.0 - torch.pow(cos_th, 2)).clamp(0, 1))
        cos_th_plus_m = cos_th * self.cos_m - sin_th * self.sin_m
        one_hot = torch.zeros(cos_th.size(), device=self.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * cos_th_plus_m) + ((1.0 - one_hot) * cos_th)
        output *= self.s
        return output


def load_model(saved, checkpoint_path, custom=False):
    #model = torchvision.models.resnet18()
    if saved is None and custom:
        model = MobileNetV2Q()
        # model = timm.create_model('efficientnet_b0', pretrained=True)
        pretrained_model = torch.hub.load(
            'pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
        pretrained_model.classifier[1] = nn.Linear(
            pretrained_model.classifier[1].in_features, 2)
        model.load_state_dict(pretrained_model.state_dict())
        del model.classifier
    elif saved is None and not custom:
        model = timm.create_model('tf_efficientnet_b1_ns', pretrained=True)
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
    else:
        print(f"open saved: {os.path.join(checkpoint_path, saved)}")
        model = timm.create_model('tf_efficientnet_b1_ns', pretrained=False)
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        model.load_state_dict(torch.load(os.path.join(checkpoint_path, saved)))

    return model


def extract_faces(data_path, face_path, batch_size_det):
    align = FaceAligner()
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

            # if '666' not in class_name:
            #    continue
            os.makedirs(os.path.join(face_path, dir_name,
                                     class_name), exist_ok=True)

    def split_list(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]
    print('Start face extracting and saving into new directory...')
    for dir_name in [train_dir, test_dir]:
        print(dir_name)
        class_names = os.listdir(os.path.join(data_path, dir_name))
        if dir_name == train_dir:
            class_names = [x for x in class_names if int(x[1:]) > 8346]

        for class_name in tqdm(class_names, total=len(class_names)):
            file_names = os.listdir(os.path.join(
                data_path, dir_name, class_name))
            file_names = [
                x for x in file_names if ".ipynb_checkpoints" not in x]
            for batch in split_list(file_names, n=batch_size_det):
                paths = []
                pics = []
                pics_numpy = []
                batch_pics = []
                for infile in batch:
                    file, ext = os.path.splitext(infile)
                    pic_path = os.path.join(
                        data_path, dir_name, class_name, infile)
                    pic = Image.open(pic_path)
                    paths.append(pic_path)
                    pics.append(pic)
                    pics_numpy.append(np.array(pic))
                batch_h = int(
                    min(np.quantile([pic.shape[0] for pic in pics_numpy], q=0.9), 500))
                batch_w = int(
                    min(np.quantile([pic.shape[1] for pic in pics_numpy], q=0.9), 500))
                pics_resized = [pic.resize((batch_h, batch_w)) for pic in pics]
                batch_pics = [np.array(pic) for pic in pics_resized]
                batch_pics = np.array(batch_pics)
                detections = align.align(batch_pics)
                for detection in detections:
                    continue
                    if detection is not None:
                        plt.imshow(detection)
                        plt.show()

                for detection, path in zip(detections, paths):
                    if detection is not None:
                        file_parsed = path.split(os.sep)
                        face_name = file_parsed[-1]
                        person_name = file_parsed[-2]
                        save_path = os.path.join(
                            face_path, dir_name, person_name, 'face_' + face_name)
                        Image.fromarray(detection).save(save_path, "JPEG")


class AlbuDataset(data.Dataset):
    def __init__(self, img_dir, transforms=None):
        self.img_dir = img_dir
        self.targets = sorted(os.listdir(img_dir))
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
        self.targets_to_idx = {t: i for i,
                               t in enumerate(sorted(self.targets))}
        # print(self.targets_to_idx)
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


def train_model(model, loss, metric, optimizer, scheduler, device, train_dataloader, val_dataloader, batch_size, num_epochs, checkpoint_path, quantize=False):
    pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    for epoch in range(num_epochs):
        print('Epoch {}/{}:'.format(epoch, num_epochs - 1), flush=True)
        #pbar.set_description('Epoch {}/{}:'.format(epoch, num_epochs - 1))
        for phase in ['train', 'val']:
            if phase == 'train':
                dataloader = train_dataloader
                scheduler.step()
                model.train()
                model = model.to(device)
            else:
                dataloader = val_dataloader
                model.eval()
            pbar = tqdm(enumerate(dataloader), total=len(dataloader))
            running_loss = 0.
            running_acc = 0.
            scale_value = 0.
            if phase == "val" and quantize:
                if epoch > 0:
                    # Freeze quantizer parameters
                    model.apply(torch.quantization.disable_observer)
                if epoch > 0:
                    # Freeze batch norm mean and variance estimates
                    model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
                model = model.to("cpu")
            for idx, (inputs, labels) in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    if phase == "val" and quantize:
                        quantized_model = torch.quantization.convert(
                            model.eval(), inplace=False)
                        quantized_model.eval()
                        inputs = inputs.to("cpu")
                        feature = quantized_model(inputs)
                        labels = labels.to(device)
                    else:
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
                pbar.set_description("{} Loss: {:.4f} Acc: {:.4f}".format(
                    phase, running_norm_loss, running_norm_acc))
            if phase == 'train':
                pass
            else:
                model_name = 'vggface2_{}_{:.4f}_{:.4f}.h5'.format(
                    epoch, running_norm_loss, running_norm_acc)
                torch.save(model.state_dict(), os.path.join(
                    checkpoint_path, model_name))


if __name__ == "__main__":
    # python main.py --data_path "." --face_path ".\faces_cropped" --checkpoint_path ".\model_checkpoints" --batch_size 200 --saved vggface2_0_17.0799_0.0326.h5
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('--data_path', help='Image directory')
    parser.add_argument(
        '--face_path', help='New empty directory for extracted faces')
    parser.add_argument('--checkpoint_path', help='Checkpoints directory')
    parser.add_argument('--batch_size_det',
                        help='Batch size for face detection')
    parser.add_argument('--saved', help='Use saved checkpoint', default=None)
    args = parser.parse_args()
    data_path = args.data_path
    face_path = args.face_path
    checkpoint_path = args.checkpoint_path
    batch_size_det = int(args.batch_size_det)
    saved = args.saved

    if not os.path.isdir(face_path):
        extract_faces(data_path, face_path, batch_size_det)

    # extract_faces(data_path, face_path, batch_size_det)
    resize = 128
    crop_size = 120
    p_small = 0.005
    p_big = 0.5
    transforms = A.Compose(
        [
            # A.Resize(resize, resize),
            #A.RandomCrop(crop_size, crop_size),
            A.Blur(blur_limit=5, p=p_small),
            #A.RandomBrightness(p=p_small, limit=(-0.2, 0.2)),
            A.JpegCompression(quality_lower=50, quality_upper=100, p=p_small),
            A.GaussNoise(var_limit=200, p=p_small),
            # A.RandomSunFlare(p=p_small),
            #A.Downscale(p=p_small, scale_min=0.25, scale_max=0.8),
            # A.CLAHE(p=p_small),
            A.RandomContrast(p=0.05),
            A.RandomBrightness(p=0.05),
            # A.HorizontalFlip(),
            # A.VerticalFlip(p=p_small),
            # A.RandomRotate90(),
            # A.ShiftScaleRotate(shift_limit=0, scale_limit=0,
            #                   rotate_limit=180, p=p_big),
            #A.Blur(blur_limit=2, p=0.05),
            # A.OpticalDistortion(p=0.05),
            # A.GridDistortion(p=0.05),
            # A.ChannelShuffle(p=0.05),
            # A.HueSaturationValue(p=0.05),
            # A.ElasticTransform(),
            A.ToGray(p=p_small),
            # A.JpegCompression(p=0.05),
            # A.MedianBlur(p=0.05),
            A.Cutout(max_h_size=10, num_holes=2, max_w_size=10, p=p_small),
            A.Normalize(),
            ToTensorV2(),
        ],
        p=1,
    )
    transforms_test = A.Compose(
        [
            # T.Resize((resize, resize)),
            # T.RandomCrop((crop_size, crop_size)),
            # T.RandomHorizontalFlip(),
            A.Normalize(),
            ToTensorV2(),
        ],
        p=1,)
    dataset_train = AlbuDataset(os.path.join(face_path, 'train'), transforms)
    dataset_test = AlbuDataset(os.path.join(
        face_path, 'test'), transforms_test)
    batch_size = 64
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
    optimizer = torch.optim.Adam(params, lr=3.0e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=7, gamma=0.1)
    """
    print("preparing qat")
    model = model.to("cpu")
    model.eval()
    model.fuse_model()
    model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
    torch.quantization.prepare_qat(model, inplace=True)
    print("start training")"""
    model = model.to(device)
    num_epochs = 100
    train_model(model, loss, metric, optimizer, scheduler, device,
                train_dataloader, test_dataloader,
                batch_size, num_epochs, checkpoint_path)
