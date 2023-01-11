import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from torchvision import models, datasets, transforms

import numpy as np
import time
import sys
import os
import json
import tqdm
from PIL import Image

from models import utils, caption
from configuration import Config

# MAX_DIM = 384
MAX_DIM = 320 # use this to beat CLIP strong baseline

config = Config()

class p2Data(Dataset):
    def __init__(self, fnames, transform=None):
        self.transform = transform
        self.fnames = fnames
        self.file_list = [file for file in os.listdir(
            fnames) if file.endswith('.jpg')]
        self.file_list.sort()
        self.num_samples = len(self.file_list)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        filepath = os.path.join(self.fnames, fname)
        img = Image.open(filepath)
        img = self.transform(img)
        return img, fname


class dim:
    def __init__(self):
        self.dim = 3

    def __call__(self, x):
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        return x


test_transform = transforms.Compose([
    transforms.ToTensor(),
    dim(),
    transforms.Resize((MAX_DIM, MAX_DIM)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# test_image_dir = '../hw3_data/p2_data/images/val'
test_image_dir = sys.argv[1]
# output_dir = 'output_34_225_320.json'
output_dir = sys.argv[2]
test_set = p2Data(test_image_dir, transform=test_transform)
test_dataloader = DataLoader(test_set, batch_size=32, shuffle=False)
device = torch.device(config.device)
print(f'Initializing Device: {device}')

seed = config.seed + utils.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)

_, criterion = caption.build_model(config)
model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=False)
model.to(device)
model.eval()

config.checkpoint = '27_model_best_225.pth'

# if os.path.exists(config.checkpoint):
print("Loading Checkpoint...")
# checkpoint = torch.load(config.checkpoint, map_location='cpu')
checkpoint = torch.load(config.checkpoint)
model.load_state_dict(checkpoint['model'])

# print(f"Valid: {len(test_set)}")


def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros(
        (imgs.shape[0], max_length), dtype=torch.long)
    mask_template = torch.ones((imgs.shape[0], max_length), dtype=torch.bool)
    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template, mask_template


tokenizer = Tokenizer.from_file("caption_tokenizer.json")
total = len(test_dataloader)
start_token = 2
end_token = 3
max_len = 30
result_dict = {}
with tqdm.tqdm(total=total) as pbar:
    with torch.no_grad():
        for k, (imgs, fnames) in enumerate(test_dataloader):
            imgs = imgs.to(device)

            cap2, cap_mask2 = create_caption_and_mask(
                start_token, config.max_position_embeddings)
            cap2 = cap2.to(device)
            cap_mask2 = cap_mask2.to(device)

            for i in range(max_len):
                predictions = model(imgs, cap2, cap_mask2)[:, i, :]
                predicted_id = torch.argmax(predictions, axis=-1)
                for j in range(imgs.shape[0]):
                    if predicted_id[j] != 3:
                        cap2[j, i + 1] = predicted_id[j]
                        cap_mask2[j, i + 1] = False

            for r in range(imgs.shape[0]):
                string = tokenizer.decode(
                    cap2[r].tolist(), skip_special_tokens=True).capitalize().split('.')[0]
                string = string[:-1] + '.'
                basename = fnames[r][:-4]
                print(f'{basename}: {string}')
                result_dict[basename] = string
            pbar.update(1)

json_object = json.dumps(result_dict, indent=4)
with open(output_dir, "w") as outfile:
    outfile.write(json_object)
