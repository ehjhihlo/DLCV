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
from torchvision import datasets
from tqdm import tqdm

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


def label_transfer_int2str(num):
    if (int)(num) == 0:
        return "Fork"
    elif (int)(num) == 1:
        return "Radio"
    elif (int)(num) == 2:
        return "Glasses"
    elif (int)(num) == 3:
        return "Webcam"
    elif (int)(num) == 4:
        return "Speaker"
    elif (int)(num) == 5:
        return "Keyboard"
    elif (int)(num) == 6:
        return "Sneakers"
    elif (int)(num) == 7:
        return "Bucket"
    elif (int)(num) == 8:
        return "Alarm_Clock"
    elif (int)(num) == 9:
        return "Exit_Sign"
    elif (int)(num) == 10:
        return "Calculator"
    elif (int)(num) == 11:
        return "Folder"
    elif (int)(num) == 12:
        return "Lamp_Shade"
    elif (int)(num) == 13:
        return "Refrigerator"
    elif (int)(num) == 14:
        return "Pen"
    elif (int)(num) == 15:
        return "Soda"
    elif (int)(num) == 16:
        return "TV"
    elif (int)(num) == 17:
        return "Candles"
    elif (int)(num) == 18:
        return "Chair"
    elif (int)(num) == 19:
        return "Computer"
    elif (int)(num) == 20:
        return "Kettle"
    elif (int)(num) == 21:
        return "Monitor"
    elif (int)(num) == 22:
        return "Marker"
    elif (int)(num) == 23:
        return "Scissors"
    elif (int)(num) == 24:
        return "Couch"
    elif (int)(num) == 25:
        return "Trash_Can"
    elif (int)(num) == 26:
        return "Ruler"
    elif (int)(num) == 27:
        return "Telephone"
    elif (int)(num) == 28:
        return "Hammer"
    elif (int)(num) == 29:
        return "Helmet"
    elif (int)(num) == 30:
        return "ToothBrush"
    elif (int)(num) == 31:
        return "Fan"
    elif (int)(num) == 32:
        return "Spoon"
    elif (int)(num) == 33:
        return "Calendar"
    elif (int)(num) == 34:
        return "Oven"
    elif (int)(num) == 35:
        return "Eraser"
    elif (int)(num) == 36:
        return "Postit_Notes"
    elif (int)(num) == 37:
        return "Mop"
    elif (int)(num) == 38:
        return "Table"
    elif (int)(num) == 39:
        return "Laptop"
    elif (int)(num) == 40:
        return "Pan"
    elif (int)(num) == 41:
        return "Bike"
    elif (int)(num) == 42:
        return "Clipboards"
    elif (int)(num) == 43:
        return "Shelf"
    elif (int)(num) == 44:
        return "Paper_Clip"
    elif (int)(num) == 45:
        return "File_Cabinet"
    elif (int)(num) == 46:
        return "Push_Pin"
    elif (int)(num) == 47:
        return "Mug"
    elif (int)(num) == 48:
        return "Bottle"
    elif (int)(num) == 49:
        return "Knives"
    elif (int)(num) == 50:
        return "Curtains"
    elif (int)(num) == 51:
        return "Printer"
    elif (int)(num) == 52:
        return "Drill"
    elif (int)(num) == 53:
        return "Toys"
    elif (int)(num) == 54:
        return "Mouse"
    elif (int)(num) == 55:
        return "Flowers"
    elif (int)(num) == 56:
        return "Desk_Lamp"
    elif (int)(num) == 57:
        return "Pencil"
    elif (int)(num) == 58:
        return "Sink"
    elif (int)(num) == 59:
        return "Batteries"
    elif (int)(num) == 60:
        return "Bed"
    elif (int)(num) == 61:
        return "Screwdriver"
    elif (int)(num) == 62:
        return "Backpack"
    elif (int)(num) == 63:
        return "Flipflops"
    elif (int)(num) == 64:
        return "Notebook"


class p2Data(Dataset):
    def __init__(self, root, fnames, transform):
        super(p2Data).__init__()
        self.transform = transform
        self.fnames = fnames
        self.n_samples = len(self.fnames)
        self.root = root

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = Image.open(os.path.join(self.root, fname)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return self.n_samples


def get_dataset(csv_path, root):
    csv_file = pd.read_csv(csv_path)
    fnames = csv_file['filename']
    test_tfm = transforms.Compose([
        transforms.Resize((128, 128)),
        # transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = p2Data(root=root, fnames=fnames, transform=test_tfm)
    return dataset, fnames


# test_csv_dir = "./hw4_data/office/val.csv"
test_csv_dir = sys.argv[1]
# test_dir = "./hw4_data/office/val/"
test_dir = sys.argv[2]
# output_dir = "p2_output_second.csv"
output_dir = sys.argv[3]

batch_size = 128
test_set, fnames = get_dataset(test_csv_dir, test_dir)
test_loader = DataLoader(test_set, batch_size=batch_size,
                         shuffle=False, num_workers=0, pin_memory=True)
print('Test set:', len(test_set))


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

# checkpoint = torch.load('./p2_model_best.ckpt') #0.42364
# checkpoint = torch.load('./p2_model_111.ckpt') #0.42857
# checkpoint = torch.load('./p2_model_121.ckpt') #0.41379
# checkpoint = torch.load('./p2_model_131.ckpt') #0.40640
# checkpoint = torch.load('./p2_model_141.ckpt') #0.43596
# checkpoint = torch.load('./p2_model_151.ckpt') #0.42118
# checkpoint = torch.load('./p2_model_161.ckpt') #0.41379
# checkpoint = torch.load('./p2_model_171.ckpt') #0.44335
# checkpoint = torch.load('./p2_model_181.ckpt') #0.43103
# checkpoint = torch.load('./p2_model_191.ckpt') #0.44335
# checkpoint = torch.load('./p2_model_201.ckpt') #0.45567
# checkpoint = torch.load('./p2_model_211.ckpt') #0.43596

checkpoint = torch.load('p2_model_2nd.ckpt')  # 0.50985

# checkpoint = torch.load('./log/p2_model_2nd_171.ckpt') #0.50246
# checkpoint = torch.load('./log/p2_model_2nd_211.ckpt') #0.49753
# checkpoint = torch.load('./log/p2_model_2nd_231.ckpt') #0.49753
# checkpoint = torch.load('./log/p2_model_2nd_best.ckpt') #0.43596
# E
# checkpoint = torch.load('./p2_model_freeze.ckpt') #0.35467

# B
# checkpoint = torch.load('./p2_model_TA.ckpt') #0.32758
# D
# checkpoint = torch.load('p2_model_TA_freeze.ckpt') #0.20690
# A
# checkpoint = torch.load('p2_model_A.ckpt') #0.38177
# C 60
# checkpoint = torch.load('p2_model_C_60.ckpt') #0.34236

resnet = models.resnet50(pretrained=False).to(device)
model = p2Model(resnet).to(device)
model.load_state_dict(checkpoint)
print("Device: {}".format(device))

model.eval()
predictions = []
for batch in tqdm(test_loader):
    imgs = batch
    with torch.no_grad():
        logits = model(imgs.to(device))
    predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

with open(output_dir, "w") as f:
    f.write("id,filename,label\n")
    for i, pred in enumerate(predictions):
        f.write(f"{i},{fnames[i]},{label_transfer_int2str(pred)}\n")
