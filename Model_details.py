import os, glob, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import shutil
from scipy.io import loadmat
import torch
from torchvision.utils import make_grid
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from fastonn import SelfONN2d as SelfONN2dlayer
  

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=1, padding=1, dropout=True,q=3):
        super(Upsample, self).__init__()
        self.dropout = dropout
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ReflectionPad2d(1),
            SelfONN2dlayer(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,bias=nn.InstanceNorm1d, q=q),
            nn.Tanh(),
        )
        self.dropout_layer = nn.Dropout2d(0.5)

    def forward(self, x, shortcut=None):
        x = self.block(x)
        if self.dropout:
            x = self.dropout_layer(x)

        if shortcut is not None:
            x = torch.cat([x, shortcut], dim=1)

        return x



class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, apply_instancenorm=True,q=3):
        super(Downsample, self).__init__()
        self.conv = SelfONN2dlayer(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,bias=nn.InstanceNorm1d, q=q)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.Tanh()
        self.apply_norm = apply_instancenorm

    def forward(self, x):
        x = self.conv(x)
        if self.apply_norm:
            x = self.norm(x)
        x = self.relu(x)

        return x






class Apprentice_Unet_Regressor(nn.Module):
    def __init__(self, filter=16):
        super(Apprentice_Unet_Regressor, self).__init__()
        self.downsamples = nn.ModuleList([
            Downsample(3, filter, kernel_size=5, padding=2,apply_instancenorm=False),  # (b, filter, 128, 128)
            Downsample(filter, filter * 2, kernel_size=5,padding=2),  # (b, filter * 2, 64, 64)
            Downsample(filter * 2, filter * 4, kernel_size=5,padding=2),  # (b, filter * 4, 32, 32)
            Downsample(filter * 4, filter * 8, kernel_size=5,padding=2),  # (b, filter * 8, 16, 16)
            Downsample(filter * 8, filter * 8, kernel_size=5,padding=2), # (b, filter * 8, 8, 8)
        ])

        self.upsamples = nn.ModuleList([
            Upsample(filter * 8, filter * 8, kernel_size=7,padding=2),
            Upsample(filter * 16, filter * 4, dropout=False, kernel_size=7,padding=2),
            Upsample(filter * 8, filter * 2, dropout=False, kernel_size=7,padding=2),
            Upsample(filter * 4, filter, dropout=False, kernel_size=7,padding=2)
        ])

        self.last = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ReflectionPad2d(1),
            # SelfONN2dlayer(filter * 2, 3, kernel_size=5, stride=1, padding=1,q=3),
            SelfONN2dlayer(filter * 2, 3, kernel_size=5, stride=1, padding=1,q=3),

            nn.Tanh()
        )

    def forward(self, x):
        skips = []
        for l in self.downsamples:
            x = l(x)
            skips.append(x)
            # print(x.size())
        skips = reversed(skips[:-1])
        for l, s in zip(self.upsamples, skips):
            x = l(x, s)
            # print(x.size())
        out = self.last(x)

        return out






class Master_Regressor(nn.Module):
    def __init__(self, filter=16):
        super(Master_Regressor, self).__init__()

        self.block = nn.Sequential(
            Downsample(3, filter, kernel_size=4, stride=4, apply_instancenorm=False,q=2),
            Downsample(filter, filter * 4, kernel_size=4, stride=4,q=2),
            Downsample(filter * 4, filter * 4, kernel_size=4, stride=4,q=2),
            Downsample(filter * 4, filter * 4, kernel_size=4, stride=2,q=2),
            # Downsample(filter * 4, filter * 4, kernel_size=4, stride=2),
            # Downsample(filter * 4, filter * 4, kernel_size=4, stride=2),
            # Downsample(filter * 4, filter * 4, kernel_size=4, stride=2),

        )

        self.last = SelfONN2dlayer(filter * 4, 8, kernel_size=4, stride=1, padding=1,q=2)
        self.flat= nn.Flatten(1)
        self.flat2= nn.Flatten(2)
       
        self.fc1 = nn.Linear(8, 16)  
        self.fc2 = nn.Linear(16, 1)  
        self.tanh = nn.ReLU()


    def forward(self, x):
        x = self.block(x)
        x = self.last(x)
        x=self.flat(x)
        x=self.tanh(self.fc1(x))
        x=self.tanh(self.fc2(x))
        x = x.unsqueeze(2)

        return x

