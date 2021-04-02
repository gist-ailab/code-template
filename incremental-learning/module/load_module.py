import torch
import torch.nn as nn
from .network import Incremental_Wrapper, Icarl_Wrapper, Identity_Layer
from .loss import icarl_loss
from .resnet import resnet18, resnet34, resnet50, resnet152
from copy import deepcopy
import torch

def load_model(option, num_class, num_feature=2048):
    if option.result['train']['train_type'] == 'icarl':
        model_cls = resnet34(num_classes=num_class)

        model_enc = deepcopy(model_cls)
        model_enc.fc = nn.Linear(model_enc.fc.in_features, num_feature)
        model_fc = nn.Linear(num_feature, num_class)

        model = Icarl_Wrapper(option, model_enc=model_enc, model_fc=model_fc)

    else:
        model_cls = resnet34(num_classes=num_class)

        model_enc = deepcopy(model_cls)
        model_enc.fc = Identity_Layer()
        model_fc = model_cls.fc

        model = Incremental_Wrapper(option, model_enc=model_enc, model_fc=model_fc)

    return model

def load_optimizer(option, param):
    if option.result['train']['optimizer'] == 'adam':
        optim = torch.optim.Adam(param, lr=option.result['train']['lr'], weight_decay=0.00001)
    else:
        optim = torch.optim.SGD(param, lr=option.result['train']['lr'], momentum=0.9, weight_decay=5e-4)
    return optim

def load_scheduler(option, optimizer):
    if option.result['train']['scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=option.result['train']['total_epoch'])
    elif option.result['train']['scheduler'] is None:
        scheduler = None
    else:
        raise('select proper scheduler')

    return scheduler

def load_loss(option):
    train_type = option.result['train']['train_type']
    if train_type == 'naive':
        criterion = nn.CrossEntropyLoss()
    elif train_type == 'icarl':
        criterion = icarl_loss()
    else:
        raise('select proper train_type')

    return criterion