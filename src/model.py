# src/model.py
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


def get_model(num_classes=6, pretrained=True):
    model = EfficientNet.from_pretrained('efficientnet-b0') if pretrained else EfficientNet.from_name('efficientnet-b0')
    in_features = model._fc.in_features
    model._fc = nn.Linear(in_features, num_classes)
    return model
