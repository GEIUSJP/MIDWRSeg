import torch
import torchvision
from model.MIDWRSeg_12 import MIDWRSeg


def build_model(model_name, num_classes):
    if model_name == 'MIDWRSeg':
        return MIDWRSeg(classes=num_classes)
    else:
        raise NotImplementedError
