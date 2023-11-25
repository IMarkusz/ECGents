import os
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset


class ScalogramData(Dataset):
    def __init__(self, scalograms_dir, channels=["I", "II", "III"], width=600):
        self.channels = channels
        self.width = width
        self.paths = self._scan_paths(scalograms_dir, channels)
        self.base_fnames = list(self.paths.keys())

    def __len__(self):
        return len(self.base_fnames)

    def __getitem__(self, index):
        base_fname = self.base_fnames[index]

        x = []
        for channel in self.channels:
            img = Image.open(self.paths[base_fname][channel])
            img = img.convert("L")
            img = T.ToTensor()(img)
            img = F.pad(img, (0, 0, 0, 247 - img.shape[1]))
            x.append(img)

        x = torch.concat(x)
        x = F.pad(x, (0, self.width - x.shape[-1], 0, 0))
        x = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x)

        return x

    def _scan_paths(self, dirpath, channels):
        paths = {}
        for fname in sorted(os.listdir(dirpath)):
            channel = fname[fname.rfind("_") + 1 :].replace(".png", "")
            if channel not in channels:
                continue

            base_fname = fname.replace(f"_{channel}.png", "")
            paths[base_fname] = paths.get(base_fname, {})
            paths[base_fname][channel] = os.path.join(dirpath, fname)

        return paths


class ResNetModel(nn.Module):
    def __init__(self):
        super().__init__()

        base_model = torchvision.models.resnet152(pretrained=True)
        self.model = torch.nn.Sequential(*(list(base_model.children())[:-1]))
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.model(x)
        x = self.flatten(x)

        return x


class AlexNetModel(nn.Module):
    def __init__(self):
        super().__init__()

        base_model = torchvision.models.alexnet(pretrained=True)
        self.model = torch.nn.Sequential(*(list(base_model.children())[:-1]))
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.model(x)
        x = self.flatten(x)

        return x
