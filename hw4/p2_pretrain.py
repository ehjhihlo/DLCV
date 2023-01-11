import os
import sys
import numpy as np
import pandas as pd
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torchvision.datasets import DatasetFolder

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.transforms import AutoAugmentPolicy
from torchvision import datasets
from torch import optim
from tqdm import tqdm

import matplotlib.pyplot as plt
from PIL import Image
import random
from byol_pytorch import BYOL
from torchvision import models


def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


same_seeds(1126)


def filenameToPILImage(x):
    return Image.open(x)


train_tfm = transforms.Compose([
    filenameToPILImage,
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class p2Data(Dataset):
    def __init__(self, csv_path, data_dir, transforms=train_tfm):
        super(p2Data).__init__()
        self.data_dir = data_dir
        self.df = pd.read_csv(csv_path).set_index("id")
        self.transforms = train_tfm

    def __getitem__(self, idx):
        path = self.df.loc[idx, "filename"]
        label = self.df.loc[idx, "label"]
        image = self.transforms(os.path.join(self.data_dir, path))
        return image, label

    def __len__(self):
        return len(self.df)

# class p2_dataset(Dataset):
#     def __init__(self, fnames, transform):
#         self.transform = transform
#         self.fnames = fnames
#         self.num_samples = len(self.fnames)

#     def __getitem__(self,idx):
#         fname = self.fnames[idx]
#         # 1. Load the image
#         img = Image.open(fname).convert('RGB')
#         # 2. Resize and normalize the images using torchvision.
#         img = self.transform(img)
#         # if(img.shape[0]==4):
#         return img

#     def __len__(self):
#         return self.num_samples

# def get_dataset(root):
#     fnames = glob.glob(os.path.join(root, '*'))
#     fnames.sort()
#     # 1. Resize the image to (64, 64)
#     # 2. Linearly map [0, 1] to [-1, 1]
#     train_compose = [
#       transforms.Resize((128, 128)),
#       # transforms.RandomChoice(transform_0),
#       transforms.ToTensor(),
#       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ]
#     train_transform = transforms.Compose(train_compose)
#     dataset = p2_dataset(fnames, train_transform)
#     return dataset


train_dir = "./hw4_data/mini/train/"
train_csv_dir = "./hw4_data/mini/train.csv"
train_dataset = p2Data(train_csv_dir, train_dir, train_tfm)
train_size = int(len(train_dataset)*0.9)
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = data.random_split(
    train_dataset, [train_size, val_size])
print("Images in train set: ", len(train_dataset))
print("Images in val set: ", len(val_dataset))

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

resnet = models.resnet50(pretrained=False).to(device)
# checkpoint = torch.load('p2_resnet50.ckpt')
# resnet.load_state_dict(checkpoint)

learner = BYOL(
    resnet,
    image_size=128,
    hidden_layer='avgpool',
    # use_momentum=False       # turn off momentum in the target encoder
).to(device)

opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

n_epochs = 250
best_loss = 10000
stale = 0
patience = 100
SSL_model_path_best = './p2_resnet50_best.ckpt'
SSL_model_path = './p2_resnet50.ckpt'
learner_path = './p2_learner.ckpt'
SSL_optimizer_path = './p2_optimizer.ckpt'


for epoch in range(n_epochs):
    # ---------- Training ----------
    train_loss = []
    for batch in tqdm(train_loader):
        imgs, _ = batch
        loss = learner(imgs.to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        learner.update_moving_average()  # update moving average of target encoder
        train_loss.append(loss.item())

    train_loss = sum(train_loss) / len(train_loss)

    # Print the information.
    print(
        f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}")

    # ---------- Validation ----------
    valid_loss = []
    for batch in tqdm(valid_loader):
        imgs, _ = batch
        with torch.no_grad():
            loss = learner(imgs.to(device))
        valid_loss.append(loss.item())
    valid_loss = sum(valid_loss) / len(valid_loss)

    # save models
    if valid_loss < best_loss:
        print(f"Best model found at epoch {epoch+1}, saving model")
        # only save best to prevent output memory exceed error
        torch.save(resnet.state_dict(), SSL_model_path_best)
        best_loss = valid_loss
        stale = 0
    else:
        stale += 1
        if stale > patience:
            print(f"No improvment for {patience} epochs, stop training")
            break

    torch.save(opt.state_dict(), SSL_optimizer_path)
    torch.save(resnet.state_dict(), SSL_model_path)
    torch.save(learner.state_dict(), learner_path)
    print(
        f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}")
