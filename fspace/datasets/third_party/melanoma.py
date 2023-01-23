import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms

import os
import cv2
from PIL import Image

image_key = 'image_name'
label_key = 'target'

class MelanomaDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, train=True, transform=None, metadata=False):
        np.random.seed(23)
        data_new = os.path.join(data_path, 'melanoma-2020')
        data_old = os.path.join(data_path, 'melanoma-2019')
        dfs = []
        datapaths = [data_new, data_old]
        for data_path in datapaths:
            train_path = os.path.join(data_path, 'train')
            train_csv = os.path.join(data_path, 'train.csv')
            df = pd.read_csv(train_csv)
            df['path'] = train_path
            folds = set(df['tfrecord'].values)
            folds = [f for f in folds if f >= 0]
            train_splits = np.random.choice(folds, size=int(0.8*len(folds)), replace=False).tolist()
            val_splits = [fold for fold in folds if fold not in train_splits]
            if train:
                df = df.loc[df['tfrecord'].isin(train_splits)].reset_index()
            else:
                df = df.loc[df['tfrecord'].isin(val_splits)].reset_index()
            dfs.append(df)
        self.df = pd.concat(dfs).reset_index()
        self.file_names = self.df[image_key].values
        self.labels = self.df[label_key].values
        self.transform = transform
        self.data_path = self.df['path'].values
        self.metadata = None
        if metadata:
            self.metadata = False

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = os.path.join(self.data_path[idx], str(file_name) + '.jpg')
        image = cv2.imread(file_path)
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx]).long()
        return image, label

def get_melanoma_orig(root=None, img_size=256, normalize=None, **_):
    metadata = False

    transform = transforms.Compose([
                    transforms.Resize(img_size),
                    transforms.CenterCrop(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(*normalize)])
    testset = MelanomaDataset(root, train=False, transform=transform, metadata=metadata)

    transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(img_size, scale=(0.65, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(*normalize)])
    trainset = MelanomaDataset(root, transform=transform_train, metadata=metadata)

    return trainset, None, testset