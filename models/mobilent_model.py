import torch.nn as nn
import torchvision.models as models

def get_model():
    model = models.mobilnet_v2(num_classes=4)