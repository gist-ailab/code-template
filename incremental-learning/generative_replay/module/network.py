import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm

def split_model(model):
    pass


class Identity_Layer(nn.Module):
    def __init__(self):
        super(Identity_Layer, self).__init__()

    def forward(self, x):
        return x

#### ICaRL Wrapper (Basis)
class Incremental_Wrapper(nn.Module):
    def __init__(self, option, model_enc, model_fc, feature_out=False, device='cuda'):
        super(Incremental_Wrapper, self).__init__()
        self.option = option
        self.model_enc = model_enc
        self.model_fc = model_fc

        self.device = device
        self.feature_out = feature_out

    def forward(self, image):
        x = self.model_enc(image)
        out = self.model_fc(x)

        if self.feature_out:
            return out, x
        else:
            return out

    def feature_extractor(self, image):
        out1 = self.model_enc(image)
        return out1

# Generator
class Generator(nn.Module):
    def __init__(self, option):
        super(Generator, self).__init__()
        self.init_size = option.result['data']['img_size'] // 4
        self.l1 = nn.Sequential(nn.Linear(option.result['train']['latent_dim'], 128*self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(128),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(3, affine=False)
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks2(img)
        return img