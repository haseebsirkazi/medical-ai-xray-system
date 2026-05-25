import torch
import torch.nn as nn
from torchvision import models

def get_model():
    model = models.densenet121(pretrained=True)

    # Replace final layer for 2 classes
    model.classifier = nn.Linear(model.classifier.in_features, 2)

    return model