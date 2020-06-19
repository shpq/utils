import pandas as pd
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import cv2
import matplotlib.image as mpimg
import random
import os
from tqdm import tqdm
import argparse
import albumentations as A
from albumentations.pytorch import ToTensor
import random
from sklearn.metrics import r2_score


torch.nn.Module.dump_patches = False

class DatasetCustom(Dataset):
    def __init__(self, csv_file, root_dir, transform, size, mode):
        self.csv_file = csv_file
        self.root_dir = root_dir
        self.transform = transform
        self.size = size
        self.mode = mode
        
        
    def __len__(self):
        return len(self.csv_file)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.csv_file.at[idx, 'image_name'])
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        target = self.csv_file.at[idx, 'er_rel_norm']
        if random.random()<0.05:
            augmented = self.transform(image=image, bad=True, size=self.size, mode = self.mode)
            target = -1
        else:
            augmented = self.transform(image=image, bad=False, size=self.size, mode = self.mode)
        image = augmented["image"]
        sample = [idx, image, target]
        return sample
    
    
def stratified_split(y, split, csv_file, n_bins=50):
    dataset_size = len(y)
    bins = np.linspace(0, dataset_size, n_bins) / dataset_size
    y_binned = np.digitize(y, bins)
    train_csv, test_csv = train_test_split(csv_file, test_size=split,\
                                                        shuffle=True, stratify=y_binned,\
                                                       random_state=14)
    return train_csv, test_csv


def get_split(csv_name):
    split = 0.1
    csv_file = pd.read_csv(csv_name, index_col=0)
    creative_ids = csv_file.creative_id.unique()[:1]
    csv_file_test = csv_file[csv_file.creative_id.isin(creative_ids)]
    csv_file_split = csv_file[~csv_file.creative_id.isin(creative_ids)]
    y = csv_file['er']
    y = y.to_numpy().reshape(len(y), 1)
    ss = StandardScaler()
    y = ss.fit_transform(y).reshape(1, len(y))[0]
    csv_file['target'] = y
    dataset_size = len(y)
    train_csv, test_csv = None, None
    for n_bins in [50, 40, 30, 20]:
        try:
            train_csv, test_csv = stratified_split(y, split, csv_file_split, n_bins)
            print('stratified split')
        except Exception as e:
            continue
    if train_csv is None:
        print('non-stratified split')
        train_csv, test_csv = train_test_split(csv_file_split, test_size=split,\
                                                        shuffle=True, random_state=14)
    test_csv = pd.concat([
            test_csv,
            csv_file_test
        ], axis=0)
    train_csv = train_csv.reset_index(drop=True)
    test_csv = test_csv.reset_index(drop=True)
    test_csv['test_ids'] = str(creative_ids.to_list())
    return train_csv, test_csv


def load_model(model_path, model_name):
    path = os.path.join(model_path, model_name)
    model = torch.load(path, map_location=torch.device('cpu'))
    for param in model.parameters():
        param.requires_grad = False
    model.last_linear = torch.nn.Linear(2048, 1, bias=True)
    return model


def train_model(model, loss, optimizer, scheduler, device, train_dataloader, val_dataloader, model_path, split_path, batch_size, num_epochs):
    train_loss_list, val_loss_list = [],[]
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
            scale_value = 0.
            score = 0.
            r2_scores = []
            df_pred = pd.DataFrame([0]*len(dataloader) * batch_size, columns=['target'])
            for idx, (idxs, inputs, labels) in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    preds = model(inputs).view(-1)
                    loss_value = loss(preds, labels)
                    if phase == 'train':
                        loss_value.backward()
                        optimizer.step()
                running_loss += loss_value.item()
                scale_value = 1 / (idx + 1)
                true_vals = torch.Tensor.tolist(labels)
                idxs = torch.Tensor.tolist(idxs)
                preds = torch.Tensor.tolist(preds)
                r2 = r2_score(np.array(true_vals), np.array(preds))
                r2_scores.append(r2)
                score = np.mean(r2_scores)
                running_norm_loss = running_loss * scale_value

                pbar.set_description("{} Loss: {:.4f}, r2 score {:.4f}".format(phase, running_norm_loss, score))

                
                df_pred.loc[idxs, 'target'] = preds
            if phase == 'train':
                score_train = round(score, 3)
                train_loss_list.append(running_norm_loss)
            else:
                val_loss_list.append(running_norm_loss)
                model_name = f'er_{epoch}_{round(train_loss_list[-1], 3)}_vs_{round(val_loss_list[-1], 3)}_r2_{score_train}_vs_{round(score, 3)}.h5'
                torch.save(model, os.path.join(model_path, model_name))
                csv_test_name = os.path.join(split_path, 'test.csv')
                test_csv = pd.read_csv(csv_test_name, index_col=0)
                test_csv[f'target_{epoch}'] = df_pred['target']
                test_csv.to_csv(csv_test_name)
    return train_loss_list, val_loss_list

def additional_augmenation(image, bad=False, size=(600, 600), mode='val'):
    if not bad:
        transforms = A.Compose(
            [
                A.Resize(size[0], size[1]),
                #A.RandomCrop(crop_size, crop_size),
                A.RandomBrightness(p=0.5, limit=0.2),
                A.HorizontalFlip(),
                A.ShiftScaleRotate(
                    shift_limit=0.0, scale_limit=0.08, rotate_limit=10, p=0.5
                ),
                A.ChannelShuffle(p=0.01), 
                A.HueSaturationValue(p=0.01), 
                A.ToGray(p=0.05),
                A.Normalize(),
                ToTensor(),
            ]
        )
    else:
        bad_transform = random.choice([A.Blur(blur_limit=(14, 15), p=1.0),
                                       A.RandomBrightness(p=1, limit=(0.8, 1)),
                                       A.RandomBrightness(p=1, limit=(-1, -0.8)),
                                       A.JpegCompression(quality_lower=2,quality_upper=10,p=1.0),
                                       A.GaussNoise(var_limit=(10000, 10000), p=1.0),
                                       ])
        transforms = A.Compose(
            [
                A.Resize(size[0], size[1]),
                #A.RandomCrop(crop_size, crop_size),
                A.RandomBrightness(p=0.5, limit=0.2),
                A.HorizontalFlip(),
                A.ShiftScaleRotate(
                    shift_limit=0.0, scale_limit=0.08, rotate_limit=10, p=0.5
                ),
                bad_transform,
                A.ChannelShuffle(p=0.01), 
                A.HueSaturationValue(p=0.01), 
                A.ToGray(p=0.05),
                A.Normalize(),
                ToTensor(),
            ]
        )
    if mode == 'val':
        transforms = A.Compose(
            [
                A.Resize(size[0], size[1]),
                A.Normalize(),
                ToTensor(),
            ]
        )
    return transforms(image=image)



if __name__ == "__main__":
    #python main.py --image_path test_im --csv_name test_data.csv --model_path model --model_name csv_from_toloka_clear_unappr_torch_seresnext50_32x4d__epoch_7_val_acc_0.979_val_loss_0.07.h5
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('--image_path', help='Image directory')
    parser.add_argument('--train_name', help='Path to train csv file')
    parser.add_argument('--test_name', help='Path to test csv file')
    parser.add_argument('--train_size', help='Train size of image')
    parser.add_argument('--test_size', help='Test size of image')
    parser.add_argument('--model_path', help='Model and checkpoints directory')
    parser.add_argument('--model_name', help='Model file name')
    parser.add_argument('--batch_size', help='Batch size')
    parser.add_argument('--lr', help='Learning rate')
    args = parser.parse_args()
    image_path = args.image_path
    # csv_name = args.csv_name
    model_path = args.model_path
    lr = float(args.lr)
    model_name = args.model_name
    batch_size = int(args.batch_size)
    size_train = (int(args.train_size), int(args.train_size))
    size_test = (int(args.test_size), int(args.test_size))
    train_csv, test_csv = pd.read_csv(args.train_name), pd.read_csv(args.test_name)
    split_path = 'split'
    train_csv.to_csv(os.path.join(split_path, 'train.csv'))
    test_csv.to_csv(os.path.join(split_path, 'test.csv'))
    
    dataset_train = DatasetCustom(train_csv, image_path, additional_augmenation, size_train, mode='train')
    dataset_test = DatasetCustom(test_csv, image_path, additional_augmenation, size_test, mode='val')
    
    train_dataloader = torch.utils.data.DataLoader(
    dataset_train, batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(
    dataset_test, batch_size=batch_size)
    model = load_model(model_path, model_name)
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    num_epochs = 100
    train_model(model, loss, optimizer, scheduler, device,
                train_dataloader, test_dataloader,
                model_path, split_path,
                batch_size, num_epochs)