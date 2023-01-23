'''
Modified from: https://github.com/mlfoundations/imagenet-applications-transfer/blob/main/datasets/cassava.py
'''

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from pathlib import Path

import os
import cv2
from PIL import Image


class LeafDataset(torch.utils.data.Dataset):
    def __init__(self, df, data_path, transform=None):
        self.df = df
        self.file_names = df['image_id'].values
        self.labels = df['label'].values
        self.transform = transform
        self.data_path = data_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = os.path.join(self.data_path, file_name)
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx]).long()
        return image, label


def get_cassava_orig(root=None, n_classes=5, img_size=256, seed=42, val_size=.2, normalize=None, **_):
    metadata = False
    root = Path(root) / 'cassava'
    train_path = os.path.join(root, 'train_images')
    train_csv = os.path.join(root, 'train.csv')
    train_df = pd.read_csv(train_csv)

    np.random.seed(seed)
    train_size = len(train_df.values)
    indices_by_label = []
    validation_idx = []
    for i in range(n_classes):
        label_idx = [idx for idx, l in enumerate(train_df['label'].values) if l == i]
        validation_idx.extend(np.random.choice(label_idx, int(val_size * len(label_idx)), replace=False).tolist())

    train_idx = [i for i in range(train_size) if i not in validation_idx]
    train_keys = train_df['image_id'].values[train_idx]
    validation_keys = train_df['image_id'].values[validation_idx]
    train_split_df = train_df[train_df['image_id'].isin(train_keys.tolist())]
    validation_split_df = train_df[train_df['image_id'].isin(validation_keys.tolist())]

    transform = transforms.Compose([
                    transforms.Resize(img_size),
                    transforms.CenterCrop(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(*normalize)])
    testset = LeafDataset(df=validation_split_df, data_path=train_path, transform=transform)

    transform_train = transforms.Compose([
                    transforms.RandomResizedCrop(img_size, scale=(0.65, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(*normalize)])
    trainset = LeafDataset(df=train_split_df, data_path=train_path, transform=transform_train)

    return trainset, testset