import os
import sys
import glob
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision import datasets
import torchvision.models as models
from torchvision.datasets import DatasetFolder
from torchvision.models import vgg16
from torch import optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from PIL import Image
import numpy as np
import random
import imageio
from tqdm.auto import tqdm

torch.cuda.empty_cache()

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

def read_masks(filepath):
    '''
    Read masks from directory and tranform to categorical
    '''
    file_list = [file for file in os.listdir(filepath) if file.endswith('.png')]
    file_list.sort()
    n_masks = len(file_list)
    masks = np.empty((n_masks, 512, 512))

    for i, file in enumerate(file_list):
        mask = imageio.imread(os.path.join(filepath, file))
        mask = (mask >= 128).astype(int)
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        masks[i, mask == 3] = 0  # (Cyan: 011) Urban land 
        masks[i, mask == 6] = 1  # (Yellow: 110) Agriculture land 
        masks[i, mask == 5] = 2  # (Purple: 101) Rangeland 
        masks[i, mask == 2] = 3  # (Green: 010) Forest land 
        masks[i, mask == 1] = 4  # (Blue: 001) Water 
        masks[i, mask == 7] = 5  # (White: 111) Barren land 
        masks[i, mask == 0] = 6  # (Black: 000) Unknown 

    return masks

def read_sats(filepath):
    '''
    Read masks from directory and tranform to categorical
    '''
    file_list = [file for file in os.listdir(filepath) if file.endswith('.jpg')]
    file_list.sort()
    n_masks = len(file_list)
    file_list2 = []
    sats = []

    for i, file in enumerate(file_list):
        sat = Image.open(os.path.join(filepath, file))
        sats.append(test_tfm(sat))
        file_list2.append(file.split('.')[0] + '.png')
        sat.close()

    return sats, file_list2

def mean_iou_score(pred, labels):
    '''
    Compute mean IoU score over 6 classes
    '''
    mean_iou = 0
    for i in range(6):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        iou = tp / (tp_fp + tp_fn - tp)
        mean_iou += iou / 6
        print('class #%d : %1.5f'%(i, iou))
    print('\nmean_iou: %f\n' % mean_iou)

    return mean_iou

# class p2Data(Dataset):
#     def __init__(self, root, images, labels, transform=None):
#         super(p2Data, self).__init__()
#         self.transform = transform
#         self.images = images
#         self.labels = labels
#         self.filenames = []
#         filenames = glob.glob(os.path.join(root,'*'))
#         for fn in filenames:
#             self.filenames.append(fn)
#         self.len = len(self.filenames)
#     def __getitem__(self, index):
#         image_fn = self.images[index]
#         image = Image.open(image_fn)
#         if self.transform is not None:
#             image = self.transform(image)
#         label = self.labels[index]
#         return image, label
#     def __len__(self):
#         return self.len

class p2Data(Dataset):
    def __init__(self, images, labels):
        super(p2Data, self).__init__()
        # self.transform = transform
        self.images = images
        self.labels = labels
        self.len = len(self.labels)
    def __getitem__(self, index):
        image = self.images[index]
        # image = Image.open(image)
        label = self.labels[index]
        # if self.transform is not None:
        #     image = self.transform(image)
        return image, label
    def __len__(self):
        return self.len

train_tfm = transforms.Compose([
	transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.6),
    transforms.ToTensor(),
  # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_tfm = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Set path, batch size
batch_size = 32
# test_path = "../hw1_data/p2_data/validation/"
test_path = sys.argv[1]
# output_path = "/p2_output/"
output_path = sys.argv[2]
test_images = []

# Construct datasets
test_images = read_sats(test_path)[0]
file_list = read_sats(test_path)[1]
test_label = np.zeros((len(test_images), 512, 512))
# test_label = read_masks(test_path)

# for i in range(len(file_list)):
# 	file_list[i] = file_list[i].replace("_sat", "")

test_set = p2Data(images=test_images, labels=test_label)
print('# images in testset:', len(test_set))

# Construct data loaders.
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

class fcn4(nn.Module):
    """
    ref:https://foundationsofdl.com/2021/03/03/segmentation-model-implementation/
    """
    def __init__(self, backbone_model=models.vgg16(pretrained=True)):
        super(fcn4, self).__init__()
        self.features = backbone_model.features
        self.conv1to2 = nn.Sequential(*list(self.features.children())[:11])
        self.poo12 = self.features[10]
        self.conv3 = nn.Sequential(*list(self.features.children())[11:17]) 
        self.pool3 = self.features[16]
        self.conv4 = nn.Sequential(*list(self.features.children())[17:24])
        self.pool4 = self.features[23]
        self.conv5 = nn.Sequential(*list(self.features.children())[24:])
        self.pool5 = self.features[30]
        self.conv6 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv7 = nn.Conv2d(512, 512, 3, 1, 1)
        self.upsample2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, bias=False)
        self.upsample2_fuse = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, bias=False)
        self.upsample4 = nn.ConvTranspose2d(512, 256, kernel_size=8, stride=4, bias=False)
        # self.upsample8 = nn.ConvTranspose2d(512, 256, kernel_size=16, stride=8, bias=False)
        self.upsample8 = nn.ConvTranspose2d(256, 7, kernel_size=16, stride=8, bias=False)
        # self.upsample16 = nn.ConvTranspose2d(256, 7, 32, 16, 0, bias=False)
        # self.upsample32 = nn.ConvTranspose2d(512, 7, 64, 32, 0, bias=False)
        self.dropout = nn.Dropout2d()
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        x = self.conv1to2(x)
        pooling2 = x
        x = self.conv3(x)
        pooling3 = x
        x = self.conv4(x)
        pooling4 = x
        pooling4 = self.upsample2(pooling4)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv7(x)
        conv_7 = x
        # print('conv7',conv_7.shape, conv_7.dtype)
        conv_7 = self.upsample4(conv_7)
        # print('conv7',conv_7.shape, conv_7.dtype)
        fusion_1 = pooling3 + pooling4[:, :, 1:pooling4.shape[2]-1, 1:pooling4.shape[3]-1] + conv_7[:, :, 2:conv_7.shape[2]-2, 2:conv_7.shape[3]-2]
        # print('pool3', pooling3.shape, pooling3.dtype)
        fusion_1 = self.upsample2_fuse(fusion_1)
        fusion_2 = pooling2 + fusion_1[:, :, 1:fusion_1.shape[2]-1, 1:fusion_1.shape[3]-1]
        # print("fusion_2",fusion_2.shape, fusion_2.dtype)
        fusion = self.upsample8(fusion_2)
        x_out = fusion[:, :, 4:fusion.shape[2]-4, 4:fusion.shape[3]-4]
        # print(x_out.shape, x_out.dtype)
        return x_out

# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True

device = "cuda" if torch.cuda.is_available() else "cpu"
# model = fcn32().to(device)
model = fcn4().to(device)
# model_path = './model_p2_fcn8.ckpt'
model_path = 'model_p2_fcn4.ckpt'

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5, betas=(0.9, 0.999))

# checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['net'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

model.eval()
predictions = []
for batch in tqdm(test_loader):
    imgs, labels = batch
    with torch.no_grad():
        logits = model(imgs.to(device))
    predictions.extend(logits.argmax(dim=1).cpu().detach().numpy())
# predicted_mask_val = np.array(predictions, dtype=np.float16)
# valid_miou = mean_iou_score(predicted_mask_val, test_label)
# print(valid_miou)
masks_pred = np.empty((len(predictions), 512, 512, 3))

for i, p in enumerate(predictions):
    masks_pred[i, p == 0] = [0, 255, 255]  # (Cyan: 011) Urban land 
    masks_pred[i, p == 1] = [255, 255, 0]  # (Yellow: 110) Agriculture land 
    masks_pred[i, p == 2] = [255, 0, 255]  # (Purple: 101) Rangeland 
    masks_pred[i, p == 3] = [0, 255, 0]  # (Green: 010) Forest land 
    masks_pred[i, p == 4] = [0, 0, 255]  # (Blue: 001) Water 
    masks_pred[i, p == 5] = [255, 255, 255]  # (White: 111) Barren land 
    masks_pred[i, p == 6] = [0, 0, 0] # (Black: 000) Unknown 

masks_pred = masks_pred.astype(np.uint8)
for i, photo in enumerate(masks_pred):
    # imageio.imsave(output_path + '/' + file_list[i], photo)
    imageio.imsave(os.path.join(output_path, file_list[i]), photo)
    # imageio.imsave(file_list[i], photo)
