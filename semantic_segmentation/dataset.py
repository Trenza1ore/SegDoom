from typing import Callable

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data: torch.Tensor, transform: Callable=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        image = self.data[idx, :3, :, :]  # RGB image
        label = self.data[idx, 3, :, :]   # Segmentation Labels

        sample = (image, label)

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensorNormalize:
    def __init__(self, device: str="cpu") -> None:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        self.device = device
        self.transform = transforms.Normalize(mean, std, inplace=True).to(self.device)

    def __call__(self, sample):
        image, label = sample
        # image = image.float() / 255.0
        image = image.to(device=self.device, dtype=torch.float)
        # label = (label / 20).long()
        label = label.to(device=self.device, dtype=torch.long)
        image /= 255.0
        self.transform(image)
        label //= 20
        return image, label

class ToTensorNormalizeSingleImg:
    def __init__(self, device: str="cpu") -> None:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        self.device = device
        self.transform = transforms.Normalize(mean, std, inplace=True).to(self.device)

    def __call__(self, image):
        image = image.to(device=self.device, dtype=torch.float)
        image /= 255.0
        self.transform(image)
        return image