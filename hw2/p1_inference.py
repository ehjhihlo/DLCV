# -*- coding: utf-8 -*-
import numpy as np
from torch.autograd import Variable
# from torch import optim
import torchvision
from torchvision.utils import save_image
import torch.nn.functional as F
import torch.nn as nn
import torch
import random
# import glob
import os
from PIL import Image
import sys


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
# same_seeds(777)

# workspace_dir = './hw2_data/face/'
# output_dir = './p1_output'
output_dir = sys.argv[1]

"""## Model
Here, we use DCGAN as the model structure. Feel free to modify your own model structure.

Note that the `N` of the input/output shape stands for the batch size.
"""


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    """
    Input shape: (N, in_dim)
    Output shape: (N, 3, 64, 64)
    """

    def __init__(self, in_dim, dim=64):
        super(Generator, self).__init__()

        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                # nn.ConvTranspose2d(in_dim, out_dim, 5, 2,
                #                    padding=2, output_padding=1, bias=False),
                nn.Upsample(scale_factor=2),
                nn.ReflectionPad2d(2),
                nn.Conv2d(in_dim, out_dim, 5),
                nn.BatchNorm2d(out_dim),
                nn.ReLU()
            )
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.ReLU()
        )
        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 8, dim * 4),
            dconv_bn_relu(dim * 4, dim * 2),
            dconv_bn_relu(dim * 2, dim),
            # nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1),
            nn.Upsample(scale_factor=2),
            nn.ReflectionPad2d(2),
            nn.Conv2d(dim, 3, 5),
            nn.Tanh()
        )
        self.apply(weights_init)

    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2_5(y)
        return y


class Discriminator(nn.Module):
    """
    Input shape: (N, 3, 64, 64)
    Output shape: (N, )
    """

    def __init__(self, in_dim, dim=64):
        super(Discriminator, self).__init__()

        def conv_bn_lrelu(in_dim, out_dim):
            return nn.Sequential(
                # nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                nn.utils.spectral_norm(nn.Conv2d(in_dim, out_dim, 5, 2, 2)),
                # nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.1),
            )

        """ Medium: Remove the last sigmoid layer for WGAN. """
        self.ls = nn.Sequential(
            # nn.Conv2d(in_dim, dim, 5, 2, 2),
            nn.utils.spectral_norm(nn.Conv2d(in_dim, dim, 5, 2, 2)),
            nn.LeakyReLU(0.1),
            conv_bn_lrelu(dim, dim * 2),
            conv_bn_lrelu(dim * 2, dim * 4),
            conv_bn_lrelu(dim * 4, dim * 8),
            # nn.Conv2d(dim * 8, 1, 4),
            nn.utils.spectral_norm(nn.Conv2d(dim * 8, 1, 4)),
            nn.Sigmoid(),
        )
        self.apply(weights_init)

    def forward(self, x):
        y = self.ls(x)
        y = y.view(-1)
        return y


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Training hyperparameters
batch_size = 128
z_dim = 150
z_sample = Variable(torch.randn(150, z_dim)).to(device)


G = Generator(z_dim)
D = Discriminator(3)
G.load_state_dict(torch.load('G_SNGAN2270.pth'))
D.load_state_dict(torch.load('D_SNGAN2270.pth'))

G.eval()
D.eval()
G.to(device)
D.to(device)

"""### Generate and show some images.

"""
n_output, n_gen = 1000, 2000
z_sample = Variable(torch.randn(n_gen, z_dim)).to(device)
imgs_sample = []
imgs_vec = []
imgs_score = []
with torch.no_grad():
    for i in range(0, n_gen, 50):
        img = G(z_sample[i:i+50])
        imgs_sample.append(img)
        for j in D.ls[:5]:
            img = j(img)  # apply until before last conv
        imgs_vec.append(img)
        # by_score
        for j in D.ls[5:]:
            img = j(img)
        imgs_score.append(img.view(-1))
imgs_sample = (torch.cat(imgs_sample) + 1) / 2.0
imgs_vec = torch.cat(imgs_vec)
# by_score
imgs_score = torch.cat(imgs_score)

idx = torch.sort(imgs_score)[1][:n_output].cpu().tolist()
imgs_sample = imgs_sample[list(idx)]

imgs_sample = (imgs_sample.permute(0, 2, 3, 1) *
               255).cpu().numpy().astype('uint8')

for i in range(n_output):
    im = Image.fromarray(imgs_sample[i])
    im.save(os.path.join(output_dir, f"{i+1}.jpg"), quality=86, subsampling=0)
    # im.save(f"output_SNGAN2/{i+1}.jpg", quality=86, subsampling=0)
