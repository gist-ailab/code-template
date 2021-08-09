import torch
import torch.nn as nn
from .network.resnet import ResidualNet
import json
import os
from utility.utils import config
import timm

class model_manager(object):
    def __init__(self, config_module):
        self.config = config_module
        self.network_type = config_module.result['network']['network_type']

    def load_network(self):
        # Load Pre-trained Models
        if self.network_type == 'resnet18':
            self.model = ResidualNet(self.config, self.config.result['data']['data_type'], 18, self.config.result['data']['num_class'], self.config.result['train']['attn_type'])
        elif self.network_type == 'resnet34':
            self.model = ResidualNet(self.config, self.config.result['data']['data_type'], 34, self.config.result['data']['num_class'], self.config.result['train']['attn_type'])
        elif self.network_type == 'resnet50':
            self.model = ResidualNet(self.config, self.config.result['data']['data_type'], 50, self.config.result['data']['num_class'], self.config.result['train']['attn_type'])
        elif self.network_type == 'mobilenetv3_small_075':
            self.model = timm.create_model(self.network_type, num_classes=self.config.result['data']['num_class'], pretrained=False)
        else:
            raise ('Select Proper Network Type')

    def load_weight(self, merge_path):
        self.model.load_state_dict(torch.load(merge_path)['model'][0])

def load_model(option):
    manager = model_manager(option)
    manager.load_network()
    model = manager.model
    
    model_list = [model]
    return model_list

def load_optimizer(option, model_list):
    param = model_list[0].parameters()
    
    if option.result['train']['optimizer'] == 'sgd':
        optim = torch.optim.SGD(param, lr=option.result['train']['lr'], momentum=0.9, weight_decay=option.result['train']['weight_decay'])
    elif option.result['train']['optimizer'] == 'adam':
        optim = torch.optim.Adam(param, lr=option.result['train']['lr'])
    else:
        raise('Select Proper Optimizer')
    
    optimizer_list = [optim]
    return optimizer_list

def load_scheduler(option, optimizer_list):
    optimizer = optimizer_list[0]
    
    if 'cifar' in option.result['data']['data_type']:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    elif 'imagenet' in option.result['data']['data_type']:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
    else:
        raise ('select proper scheduler')

    return [scheduler]

def load_loss(option):
    criterion = nn.CrossEntropyLoss()
    
    criterion_list = [criterion]
    return criterion_list

