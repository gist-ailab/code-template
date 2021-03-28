import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18

def load_model(option):
    model = resnet18(pretrained=False)
    return model

def load_optimizer(option, param):
    optim = torch.optim.SGD(param, lr=option.result['train']['lr'], momentum=0.9, weight_decay=5e-4)
    return optim

def load_scheduler(option, optimizer):
    if option.result['train']['scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=option.result['train']['total_epoch'])
    elif option.result['train']['scheduler'] is None:
        scheduler = None
    else:
        raise ('select proper scheduler')

    return scheduler

def load_loss(option):
    criterion = nn.CrossEntropyLoss()
    return criterion