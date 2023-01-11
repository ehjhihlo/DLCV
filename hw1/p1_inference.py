# Import necessary packages.
import glob
import numpy as np
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.transforms import AutoAugmentPolicy
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset
import sys
import random
from tqdm.auto import tqdm

# Model A
# !gdown --id 1mruBACy42xGqltXi5xoPCTSYMLZfjpM9 --output "model.ckpt"
# Model B
# !gdown --id 1PzXqnm6hfFju-HV0nzSSOchE2Hyy4-PQ --output "model.ckpt"
# !gdown --id 10GF-5MJpun0aTMQjfX2CFwhYCJxrv0M0 --output "model.ckpt"

def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
same_seeds(87)

test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class p1Data(Dataset):
    def __init__(self, imgs, labels, train):
        super(p1Data, self).__init__()
        self.labels = labels
        self.imgs = imgs
        self.train = train

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.imgs)

batch_size = 64
img_dir = sys.argv[1]
# img_dir = "../hw1_data/p1_data/val_50"
output_dir = sys.argv[2]
imgs = []
file_list = []

def read_img(filepath):
    '''
    Read masks from directory and tranform to categorical
    '''
    file_list = [file for file in os.listdir(filepath) if file.endswith('.png')]
    file_list.sort()
    images = []

    for i, file in enumerate(file_list):
        img = Image.open(os.path.join(filepath, file))
        images.append(test_tfm(img))
    # print(file_list)
    return images, file_list

images, file_list = read_img(img_dir)
test_set = p1Data(imgs=images, labels=[0]*len(images), train=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)


device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.efficientnet_b4(pretrained=False, weights=models.efficientnet.EfficientNet_B2_Weights).to(device)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.4, inplace=True),
    nn.Linear(in_features=1792, out_features=50, bias=True),
    ).to(device)

# model = resnet34().to(device)
# checkpoint = torch.load("./p1_modelB.ckpt", map_location=torch.device('cpu'))
checkpoint = torch.load("p1_modelB.ckpt")
model.load_state_dict(checkpoint)

model.eval()
predictions = []

for batch in tqdm(test_loader):
    imgs, labels = batch
    with torch.no_grad():
        logits = model(imgs.to(device))
    predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

# with open('output.csv', "w") as f:
with open(output_dir, "w") as f:
    f.write("filename, label\n")
    for i, pred in enumerate(predictions):
        f.write(f"{file_list[i]},{pred}\n")
