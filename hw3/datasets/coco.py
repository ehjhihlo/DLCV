from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision as tv

from PIL import Image
import numpy as np
import random
import os

# from transformers import BertTokenizer
from tokenizers import Tokenizer
from .utils import nested_tensor_from_tensor_list, read_json

MAX_DIM = 384


def under_max(image):
    if image.mode != 'RGB':
        image = image.convert("RGB")

    shape = np.array(image.size, dtype=np.float)
    long_dim = max(shape)
    scale = MAX_DIM / long_dim

    new_shape = (shape * scale).astype(int)
    image = image.resize(new_shape)

    return image


class RandomRotation:
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle, expand=True)


train_transform = tv.transforms.Compose([
    RandomRotation(),
    tv.transforms.Lambda(under_max),
    tv.transforms.ColorJitter(brightness=[0.5, 1.3], contrast=[
                              0.8, 1.5], saturation=[0.2, 1.5]),
    tv.transforms.RandomHorizontalFlip(),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

val_transform = tv.transforms.Compose([
    tv.transforms.Lambda(under_max),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


class CocoCaption(Dataset):
    def __init__(self, root, ann, max_length, limit, transform=train_transform, mode='training'):
        super().__init__()

        self.root = root
        self.transform = transform
        self.annot = []
        total_annot_count = 0
        for i, img_dict in enumerate(ann['images']):
            # print(f'loading: {i}')
            count = 0
            img_name = img_dict['file_name']
            # self.img_name_list.append(img_name)
            img_id = img_dict['id']
            # self.img_id_list.append(img_id)
            # cap_list = []
            # id_list = []
            for anno_dict in ann['annotations']:
                img_idd = anno_dict['image_id']
                img_cap = anno_dict['caption']
                # cap_id = anno_dict['id']
                if img_idd == img_id:
                    count += 1
                    total_annot_count += 1
                    # cap_list.append(img_cap)
                    # id_list.append(cap_id)
                    self.annot.append((img_name, img_cap))
                    print(
                        f'loading: {i}, {count} annotations of image {i} loaded, Total annotations: {total_annot_count}', end='\r')

        if mode == 'validation':
            self.annot = self.annot
        if mode == 'training':
            self.annot = self.annot[: limit]

        self.tokenizer = Tokenizer.from_file(
            "caption_tokenizer.json")
        self.max_length = max_length + 1

    # def _process(self, image_id):
    #     val = str(image_id).zfill(12)
    #     return val + '.jpg'

    def __len__(self):
        return len(self.annot)

    def __getitem__(self, idx):
        image_id, caption = self.annot[idx]
        image = Image.open(os.path.join(self.root, image_id))

        if self.transform:
            image = self.transform(image)
        image = nested_tensor_from_tensor_list(image.unsqueeze(0))

        # caption_encoded = self.tokenizer.encode_plus(
        #     caption, max_length=self.max_length, pad_to_max_length=True, return_attention_mask=True, return_token_type_ids=False, truncation=True)
        caption_encoded = self.tokenizer.encode(caption)

        # caption = np.array(caption_encoded['input_ids'])
        # cap_mask = (1 - np.array(caption_encoded['attention_mask'])).astype(bool)

        caption = np.zeros((self.max_length,), dtype=int)
        caption[:len(caption_encoded.ids)] = caption_encoded.ids
        cap_mask = (caption == 0)

        # cap_mask = np.full((self.max_length,), True)
        # for i in range(len(caption_encoded.ids)):
        #     caption[i] = caption_encoded.ids[i]
        #     cap_mask[i] = False

        # print(caption, cap_mask)
        return image.tensors.squeeze(0), image.mask.squeeze(0), caption, cap_mask


def build_dataset(config, mode='training'):
    if mode == 'training':
        # train_dir = os.path.join(config.dir, 'train2017')
        train_dir = os.path.join(config.dir, 'images', 'train')
        train_file = os.path.join(config.dir, 'train.json')
        data = CocoCaption(train_dir, read_json(
            train_file), max_length=config.max_position_embeddings, limit=config.limit, transform=train_transform, mode='training')
        return data

    elif mode == 'validation':
        # val_dir = os.path.join(config.dir, 'val2017')
        val_dir = os.path.join(config.dir, 'images', 'val')
        val_file = os.path.join(config.dir, 'val.json')
        data = CocoCaption(val_dir, read_json(
            val_file), max_length=config.max_position_embeddings, limit=config.limit, transform=val_transform, mode='validation')
        return data

    else:
        raise NotImplementedError(f"{mode} not supported")
