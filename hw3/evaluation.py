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
from datasets import coco
from configuration import Config
import sys
# import re

MAX_DIM = 384


def main(config):
    def under_max(image):
        if image.mode != 'RGB':
            image = image.convert("RGB")

        shape = np.array(image.size, dtype=np.float)
        long_dim = max(shape)
        scale = MAX_DIM / long_dim

        new_shape = (shape * scale).astype(int)
        image = image.resize(new_shape)

        return image

    test_transform = transforms.Compose([
        transforms.Lambda(under_max),
        transforms.ToTensor(),
        transforms.Resize((384, 384)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    class p2Data(Dataset):
        def __init__(self, fnames, transform=None):
            self.transform = transform
            self.fnames = fnames
            self.file_list = [file for file in os.listdir(
                fnames) if file.endswith('.jpg')]
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

    # test_json_path = 'hw3_data/p2_data/val.json'
    test_image_dir = '../hw3_data/p2_data/images/val/'
    # test_image_dir = sys.argv[1]
    # output_json = sys.argv[2]
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

    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print(f"Number of params: {n_parameters}")

    print("Loading Checkpoint...")
    # checkpoint = torch.load(config.checkpoint, map_location='cpu')
    checkpoint = torch.load('6_model_best.pth')
    model.load_state_dict(checkpoint['model'])

    # tokenizer = Tokenizer.from_file("../../../hw3_data/caption_tokenizer.json")
    print(f"Valid: {len(test_set)}")

    print("Start Evaluating..")
    # tokenizer = Tokenizer.from_file("../hw3_data/caption_tokenizer.json")
    tokenizer = Tokenizer.from_file("caption_tokenizer.json")
    model.eval()
    total = len(test_dataloader)
    print(total)

    start_token = 2
    end_token = 3

    def create_caption_and_mask(start_token, max_length):
        caption_template = torch.zeros(
            (imgs.shape[0], max_length), dtype=torch.long)
        # print('caption_template = ', caption_template)
        mask_template = torch.ones(
            (imgs.shape[0], max_length), dtype=torch.bool)
        # print('mask_template = ', mask_template)
        caption_template[:, 0] = start_token
        mask_template[:, 0] = False

        return caption_template, mask_template

    max_len = 30
    result_dict = {}
    with tqdm.tqdm(total=total) as pbar:
        with torch.no_grad():
            for k, (imgs, fnames) in enumerate(test_dataloader):
                imgs = imgs.to(device)
        #         print(imgs.shape)

                cap, cap_mask = create_caption_and_mask(
                    start_token, config.max_position_embeddings)
                cap = cap.to(device)
                cap_mask = cap_mask.to(device)
        #         print(cap.shape)

                for i in range(max_len):
                    predictions = model(imgs, cap, cap_mask)[:, i, :]
                    predicted_id = torch.argmax(predictions, axis=-1)
                    for j in range(imgs.shape[0]):
                        if predicted_id[j] != 3:
                            cap[j, i + 1] = predicted_id[j]
                            cap_mask[j, i + 1] = False

                for r in range(imgs.shape[0]):
                    s = tokenizer.decode(
                        cap[r].tolist(), skip_special_tokens=True).capitalize().split('.')[0]
                    s = s[:-1]+'.'
                    name = fnames[r][:-4]
                    print(f'{name}: {s}')
                    result_dict[name] = s
    #             input()
                pbar.update(1)

    json_object = json.dumps(result_dict, indent=4)
    # with open(output_json, "w") as outfile:
    with open("p2_output_pretrained_6_384.json", "w") as outfile:
        outfile.write(json_object)


if __name__ == "__main__":
    config = Config()
    main(config)
