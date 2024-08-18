import gc
from time import time, sleep

import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import trange, tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.models.segmentation as models

from semantic_segmentation.dataset import CustomDataset, ToTensorNormalize

def calculate_iou(pred, target, num_classes):
    ious = []
    if len(pred.shape) == 4:
        pred = torch.argmax(pred, dim=1)
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(intersection / union)
    return np.nanmean(ious)

# model_cls, model_weight, mobile_name, tqdm_miniters = {
#     "mobile"    : (models.deeplabv3_mobilenet_v3_large, 
#                    models.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT,
#                    "DeepLabV3-MobileNet", 20),
#     "res101"    : (models.deeplabv3_resnet101, 
#                    models.DeepLabV3_ResNet101_Weights.DEFAULT,
#                    "DeepLabV3-ResNet101", 5)
# }[model_type]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
res101 = mobilenetv3 = None

def init_res101(checkpoint: str | dict = "./semantic_segmentation/resnet101/ss_resnet101_18.pt",
                jit: bool = False, force: bool = False, verbose: int=0):
    global res101

    if force or (not isinstance(res101, models.DeepLabV3)):
        if isinstance(checkpoint, str):
            checkpoint = torch.load(checkpoint)["model"]
        res101 = models.deeplabv3_resnet101(weights=models.DeepLabV3_ResNet101_Weights.DEFAULT)

        prev = res101.classifier[-1]
        res101.classifier[-1] = nn.Conv2d(prev.in_channels, 13, kernel_size=prev.kernel_size, stride=prev.stride)
        del prev
        gc.collect()

        res101.load_state_dict(checkpoint)
        res101 = res101.to(device=device)
        if jit:
            res101 = torch.jit.script(res101)
        res101.eval()

        if verbose > 0:
            print("Initialized one copy of res101", flush=True)

    return res101