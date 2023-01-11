
import torch

from tokenizers import Tokenizer
from PIL import Image
# import argparse

from models import caption
from datasets import coco, utils
from configuration import Config
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np


# image_path = "../hw3_data/p3_data/images/bike.jpg"
# image_path = "../hw3_data/p3_data/images/girl.jpg"
# image_path = "../hw3_data/p3_data/images/ski.jpg"
# image_path = "../hw3_data/p3_data/images/sheep.jpg"
image_path = "../hw3_data/p3_data/images/umbrella.jpg"

# image_path = "../hw3_data/p2_data/images/val/000000406755.jpg"
# image_path = "../hw3_data/p2_data/images/val/000000084157.jpg"

config = Config()
checkpoint_path = config.checkpoint

model, _ = caption.build_model(config)
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model'])

# print(model)

tokenizer = Tokenizer.from_file("../hw3_data/caption_tokenizer.json")

start_token = 2
end_token = 3
# print('start_token = ', start_token)
# print('end_token = ', end_token)

image = Image.open(image_path)
image = coco.val_transform(image)
image = image.unsqueeze(0)

# print("image shape = ", image.shape)


def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    # print('caption_template = ', caption_template)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)
    # print('mask_template = ', mask_template)
    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template, mask_template


caption, cap_mask = create_caption_and_mask(
    start_token, config.max_position_embeddings)
# print('caption = ', caption)
# print('cap_mask = ', cap_mask)

# print("caption shape = ", caption.shape)
w = 24
img_size = 384
# x_list = []
mask_list = []
@torch.no_grad()
def evaluate():
    model.eval()
    for i in range(60):
        predictions = model(image, caption, cap_mask)
        predictions = predictions[:, i, :]
        scores = model.transformer.decoder.layers[5].att[0]
        # attn_map = scores.mean(dim=0)[i][1:].reshape(w, w).detach().cpu().numpy()
        attn_map = scores.squeeze()[i].reshape(w, w).detach().cpu().numpy()
        # print("scores = ", scores)
        predicted_id = torch.argmax(predictions, axis=-1)

        if predicted_id[0] == 3:
            return caption

        caption[:, i+1] = predicted_id[0]
        cap_mask[:, i+1] = False

        x = attn_map
        x = cv2.resize(x, (img_size, img_size), interpolation=cv2.INTER_CUBIC)

        normed_mask = x / x.max()
        normed_mask = (normed_mask * 255).astype('uint8')
        mask_list.append(normed_mask)

    return caption


output = evaluate()
print('output = ', output)
result = tokenizer.decode(
    output[0].tolist(), skip_special_tokens=True).capitalize()
print(result)

result_list = result.split()
print(result_list)
print(len(mask_list))
ground_truth_img = image[0]
ground_truth_img = ground_truth_img.permute(1, 2, 0)
ground_truth_img = np.array(ground_truth_img)
fig, ax = plt.subplots()
ax.imshow(ground_truth_img)
ax.axis('off')
ax.set_title('<BOS>')
fig.savefig(os.path.join('../p3_result', 'start'+image_path.split('/')[-1]))
for i in range(len(mask_list)+1):
    fig, ax = plt.subplots()
    ax.imshow(ground_truth_img, alpha=1)
    ax.axis('off')
    if i == len(mask_list):
        ax.set_title("<EOS>")
    else:
        ax.set_title(result_list[i])
    ax.imshow(mask_list[i], alpha=0.5, interpolation='nearest', cmap="jet")
    fig.savefig(os.path.join('../p3_result', str(i) +
                             "_"+image_path.split('/')[-1]))
