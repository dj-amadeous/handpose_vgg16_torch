

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torchvision.models import vgg16, get_model_weights
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory

data_dir = "/Volumes/Evan_Samsung_HP_data/nyu_dataset/data"

a = get_model_weights("vgg16")


class HandPoseVGG16(nn.Module):
    def __init__(self, out_dim=63, p=0.5):
        super().__init__()
        self.model = vgg16(weights="IMAGENET1K_V1")
        for param in self.model.parameters():
            param.requires_grad = False
        in_features = self.model.classifier[-1].in_features
        old_cls = list(self.model.classifier.children())
        self.model.classifier = nn.Sequential(
            *old_cls[:-1],
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(256, out_dim) # number of output features for regression of 21 joints
        )
        for m in self.model.classifier[-7:]:
            if isinstance(m, nn.Linear):
                for p_ in m.parameters():
                    p_.requires_grad = True

    def forward(self, x):
        return self.model(x)


#model = vgg16(weights="IMAGENET1K_V1")

#hpm = HandPoseVGG16()

#print(hpm.model.p())
#print(model)


#print(hpm.model.classifier[6])