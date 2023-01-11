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

# import matplotlib.pyplot as plt
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


def label_transfer_str2int(label):
    if label == "Fork":
        return 0
    elif label == "Radio":
        return 1
    elif label == "Glasses":
        return 2
    elif label == "Webcam":
        return 3
    elif label == "Speaker":
        return 4
    elif label == "Keyboard":
        return 5
    elif label == "Sneakers":
        return 6
    elif label == "Bucket":
        return 7
    elif label == "Alarm_Clock":
        return 8
    elif label == "Exit_Sign":
        return 9
    elif label == "Calculator":
        return 10
    elif label == "Folder":
        return 11
    elif label == "Lamp_Shade":
        return 12
    elif label == "Refrigerator":
        return 13
    elif label == "Pen":
        return 14
    elif label == "Soda":
        return 15
    elif label == "TV":
        return 16
    elif label == "Candles":
        return 17
    elif label == "Chair":
        return 18
    elif label == "Computer":
        return 19
    elif label == "Kettle":
        return 20
    elif label == "Monitor":
        return 21
    elif label == "Marker":
        return 22
    elif label == "Scissors":
        return 23
    elif label == "Couch":
        return 24
    elif label == "Trash_Can":
        return 25
    elif label == "Ruler":
        return 26
    elif label == "Telephone":
        return 27
    elif label == "Hammer":
        return 28
    elif label == "Helmet":
        return 29
    elif label == "ToothBrush":
        return 30
    elif label == "Fan":
        return 31
    elif label == "Spoon":
        return 32
    elif label == "Calendar":
        return 33
    elif label == "Oven":
        return 34
    elif label == "Eraser":
        return 35
    elif label == "Postit_Notes":
        return 36
    elif label == "Mop":
        return 37
    elif label == "Table":
        return 38
    elif label == "Laptop":
        return 39
    elif label == "Pan":
        return 40
    elif label == "Bike":
        return 41
    elif label == "Clipboards":
        return 42
    elif label == "Shelf":
        return 43
    elif label == "Paper_Clip":
        return 44
    elif label == "File_Cabinet":
        return 45
    elif label == "Push_Pin":
        return 46
    elif label == "Mug":
        return 47
    elif label == "Bottle":
        return 48
    elif label == "Knives":
        return 49
    elif label == "Curtains":
        return 50
    elif label == "Printer":
        return 51
    elif label == "Drill":
        return 52
    elif label == "Toys":
        return 53
    elif label == "Mouse":
        return 54
    elif label == "Flowers":
        return 55
    elif label == "Desk_Lamp":
        return 56
    elif label == "Pencil":
        return 57
    elif label == "Sink":
        return 58
    elif label == "Batteries":
        return 59
    elif label == "Bed":
        return 60
    elif label == "Screwdriver":
        return 61
    elif label == "Backpack":
        return 62
    elif label == "Flipflops":
        return 63
    elif label == "Notebook":
        return 64


autoAugment = [transforms.AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
               transforms.AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
               transforms.AutoAugment(policy=AutoAugmentPolicy.SVHN)]

train_tfm1 = transforms.Compose([
    filenameToPILImage,
    # transforms.RandomChoice(transform_set),
    transforms.Resize((128, 128)),
    transforms.AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
    transforms.CenterCrop((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_tfm2 = transforms.Compose([
    filenameToPILImage,
    # transforms.RandomChoice(transform_set),
    transforms.Resize((128, 128)),
    transforms.AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
    transforms.CenterCrop((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_tfm3 = transforms.Compose([
    filenameToPILImage,
    # transforms.RandomChoice(transform_set),
    transforms.Resize((128, 128)),
    transforms.AutoAugment(policy=AutoAugmentPolicy.SVHN),
    transforms.CenterCrop((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_tfm = transforms.Compose([
    filenameToPILImage,
    transforms.Resize((128, 128)),
    transforms.CenterCrop((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# class p2DataOffice(Dataset):
#     def __init__(self, root_dir, fnames, labels, transforms):
#         super(p2DataOffice).__init__()
#         self.root_dir = root_dir
#         self.fnames = fnames
#         self.num_samples = len(self.fnames)
#         self.labels = labels
#         self.transforms = transforms

#     def __getitem__(self, idx):
#         fname = self.fnames[idx]
#         # path = self.df.loc[idx, "filename"]
#         # label = self.df.loc[idx, "label"]
#         image = Image.open(os.path.join(self.root_dir, fname)).convert('RGB')
#         image = self.transforms(image)
#         label = label_transfer_str2int(self.labels[idx])
#         return image, label

#     def __len__(self):
#         return len(self.num_samples)


# transform_0 = [transforms.RandomAffine(degrees=(-20, 20), translate=(0.1, 0.1), scale=(1, 1.5)), transforms.RandomRotation((-15, 15)), transforms.RandomHorizontalFlip(
#     p=0.5), transforms.ColorJitter(contrast=(1, 1.5), saturation=(1, 2)), transforms.RandomPerspective(distortion_scale=0.3, p=1, interpolation=2), ]


# def get_dataset(csv_dir, root_dir):
#     csvfile = pd.read_csv(csv_dir)
#     fnames = csvfile['filename']
#     labels = csvfile['label']
#     train_compose = [
#         transforms.Resize((128, 128)),
#         # transforms.RandomChoice(transform_0),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225]),
#     ]
#     train_transform = transforms.Compose(train_compose)
#     dataset = p2DataOffice(root_dir, fnames, labels, train_transform)
#     return dataset

# workspace_dir = './hw4_data/'
# train_dataset = get_dataset(os.path.join(
#     workspace_dir, 'office/train.csv'), os.path.join(workspace_dir, 'office/train/'))
# val_dataset = get_dataset(os.path.join(
#     workspace_dir, 'office/val.csv'), os.path.join(workspace_dir, 'office/val/'))

train_dir = "./hw4_data/office/train/"
train_csv_dir = "./hw4_data/office/train.csv"
val_dir = "./hw4_data/office/val/"
val_csv_dir = "./hw4_data/office/val.csv"


class p2DataOffice(Dataset):
    def __init__(self, csv_path, data_dir, transform):
        super(p2DataOffice).__init__()
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")
        self.transform = transform

    def __getitem__(self, idx):
        path = self.data_df.loc[idx, "filename"]
        str_label = self.data_df.loc[idx, "label"]
        int_label = label_transfer_str2int(str_label)
        image = self.transform(os.path.join(self.data_dir, path))
        return image, int_label

    def __len__(self):
        return len(self.data_df)


train_set1 = p2DataOffice(train_csv_dir, train_dir, train_tfm1)
train_set2 = p2DataOffice(train_csv_dir, train_dir, train_tfm2)
train_set3 = p2DataOffice(train_csv_dir, train_dir, train_tfm3)
train_set4 = p2DataOffice(train_csv_dir, train_dir, test_tfm)
train_set = ConcatDataset([train_set1, train_set2, train_set3, train_set4])
val_set = p2DataOffice(val_csv_dir, val_dir, test_tfm)

print('Train set:', len(train_set))
print('Val set:', len(val_set))

# Construct data loaders.
batch_size = 128
train_loader = DataLoader(train_set, batch_size=batch_size,
                          shuffle=True, num_workers=0, pin_memory=True)
valid_loader = DataLoader(val_set, batch_size=batch_size,
                          shuffle=False, num_workers=0, pin_memory=True)


class p2Model(nn.Module):
    def __init__(self, resnet):
        super(p2Model, self).__init__()
        self.resnet = resnet
        self.fc_layers = nn.Sequential(
            nn.Linear(1000, 520),
            nn.BatchNorm1d(520),
            nn.LeakyReLU(0.2),

            nn.Linear(520, 130),
            nn.BatchNorm1d(130),
            nn.LeakyReLU(0.2),

            nn.Linear(130, 65)
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc_layers(x)
        return x


device = "cuda" if torch.cuda.is_available() else "cpu"
# device = 'cuda'
print(device)
backbone_model = 'p2_resnet50_246.ckpt'
# backbone_model = './hw4_data/pretrain_model_SL.pt'
checkpoint = torch.load(backbone_model)

model_path = 'p2_model_2nd.ckpt'
# model_path = './p2_model_TA_freeze.ckpt'
# model_path_best = './log/p2_model_2nd_best.ckpt'

# Initialize a model, and put it on the device specified.
resnet = models.resnet50(pretrained=False).to(device)
resnet.load_state_dict(checkpoint)

# freeze backbone model
# for param in resnet.parameters():   
#     param.requires_grad = False

model = p2Model(resnet).to(device)

print("Device: {}".format(device))

n_epochs = 40
patience = 100
stale = 0
best_acc = 0
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)
optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

for epoch in range(n_epochs):
    # ---------- Training ----------
    model.train()

    train_loss = []
    train_accs = []

    for batch in tqdm(train_loader):
        imgs, labels = batch
        logits = model(imgs.to(device))
        loss = criterion(logits, labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        # grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        optimizer.step()
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
        train_loss.append(loss.item())
        train_accs.append(acc)

    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)

    # Print the information.
    print(
        f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

    # ---------- Validation ----------
    model.eval()

    valid_loss = []
    valid_accs = []

    for batch in tqdm(valid_loader):
        imgs, labels = batch
        with torch.no_grad():
            logits = model(imgs.to(device))

        loss = criterion(logits, labels.to(device))
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        valid_loss.append(loss.item())
        valid_accs.append(acc)

    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)

    print(
        f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
    torch.save(model.state_dict(), model_path)
    # if epoch+1 > 150 and epoch % 10 == 0:
    #     torch.save(model.state_dict(), f"./log/p2_model_2nd_{epoch+1}.ckpt")
    # save models
    if valid_acc > best_acc:
        print(f"Best model found at epoch {epoch+1}, saving model")
        # torch.save(model.state_dict(), model_path_best)
        best_acc = valid_acc
        stale = 0
    else:
        stale += 1
        if stale > patience:
            print(f"No improvment for {patience} epochs, stop training")
            break