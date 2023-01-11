# -*- coding: utf-8 -*-
# from IPython.display import Image as display_image
import torch
import clip

import random
import numpy as np
import os
import torch.nn as nn
# import torchvision
# import torchvision.transforms as transforms
# import torchvision.models as models
# from torchvision.transforms import AutoAugmentPolicy
from torch.utils.data import DataLoader, Dataset
import json
# import pandas as pd
from tqdm import tqdm
from PIL import Image
import sys
"""DLCVHW3_p1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1T4iACXMWVc9Qp99WdZpAdYNYHhybVkEf
"""

# !gdown --id 1hJCiOeuLoICaKaFMsXMByl89b6jNmgz9 --output "data.zip"
# ! unzip data.zip

# ! pip install git+https://github.com/openai/CLIP.git
# ! pip install ftfy regex tqdm


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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model, preprocess = clip.load("ViT-B/32", device=device)


class p1Data(Dataset):
    def __init__(self, fnames, transform=preprocess):
        self.transform = transform
        self.fnames = fnames
        self.file_list = [file for file in os.listdir(
            fnames) if file.endswith('.png')]
        self.file_list.sort()
        # self.labels = labels
        self.num_samples = len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        # label = self.labels[idx]
        filepath = os.path.join(self.fnames, fname)
        # img = torchvision.io.read_image(fname)
        img = Image.open(filepath)
        img = self.transform(img)
        return img, fname

    def __len__(self):
        return self.num_samples


batch_size = 125
# image_path = "./hw3_data/p1_data/val/"
image_path = sys.argv[1]
test_set = p1Data(image_path, transform=preprocess)
test_dataloader = DataLoader(
    test_set, batch_size=batch_size, shuffle=False, num_workers=0)


def get_json_dataset(json_file):
    with open(json_file) as jf:
        json_data = json.load(jf)
    json_values = list(json_data.values())
    json_keys = list(json_data.keys())
    return json_keys, json_values


# json_path = "./hw3_data/p1_data/id2label.json"
json_path = sys.argv[2]
json_keys, json_values = get_json_dataset(json_path)

result = []
for i, (images, fnames) in enumerate(tqdm(test_dataloader)):
    image_input = images.to(device)
    text_inputs = torch.cat([clip.tokenize(
        f"A photo of a {c}.") for c in json_values]).to(device)  # Acc = 0.7124
    # text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}.") for c in json_values]).to(device) # Acc = 0.7124
    # text_inputs = torch.cat([clip.tokenize(f"This is a photo of {c}.") for c in json_values]).to(device) # Acc = 0.6076
    # text_inputs = torch.cat([clip.tokenize(f"This is a {c} image.") for c in json_values]).to(device) # Acc = 0.682
    # text_inputs = torch.cat([clip.tokenize(f"No {c}, no score.") for c in json_values]).to(device) # Acc = 0.5628
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)
    # Pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    for j, p in enumerate(probs):
        result.append([fnames[j], p.argmax().item()])

# output_path = 'p1_output.csv'
output_path = sys.argv[3]

with open(output_path, "w") as f:
    # correct = 0
    f.write("filename,label\n")
    for i, r in enumerate(result):
        f.write(str(r[0])+','+str(r[1]))
        f.write('\n')
    #     true_label = int(r[0].split("/")[-1].split("_")[0])
    #     pred_label = int(r[1])
    #     if (true_label == pred_label):
    #         correct += 1
    # acc = correct / len(result)
    # print(f"Acc = {round(acc, 4)}")
