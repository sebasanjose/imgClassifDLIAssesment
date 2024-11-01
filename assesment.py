import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms
import torchvision.io as tv_io

import glob
from PIL import Image

import utils

device = torch.device("mps" if torch.mps.is_available() else "cpu")
print("device is: ", device)

from torchvision.models import vgg16
from torchvision.models import VGG16_Weights

weights = VGG16_Weights.DEFAULT
vgg_model = vgg16(weights=weights)

# Freeze base model
vgg_model.requires_grad_(False)
next(iter(vgg_model.parameters())).requires_grad

vgg_model.classifier[0:3]

N_CLASSES = 1

my_model = nn.Sequential(
    vgg_model.features,
    vgg_model.avgpool,
    nn.Flatten(),
    vgg_model.classifier[0:3],
    nn.Linear(4096, 500),
    nn.ReLU(),
    nn.Linear(500, N_CLASSES)
)
my_model

loss_function = nn.BCEWithLogitsLoss
optimizer = Adam(my_model.parameters())
my_model = torch.compile(my_model.to(device))

pre_trans = weights.transforms()

IMG_WIDTH, IMG_HEIGHT = (224, 224)

random_trans = transforms.Compose([
    FIXME
])

trans = transforms.Compose([
transforms.RandomResizedCrop((IMG_WIDTH, IMG_HEIGHT), scale=(.7, 1), ratio=(1,
])