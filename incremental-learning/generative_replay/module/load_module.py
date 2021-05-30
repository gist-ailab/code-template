import torch
import torch.nn as nn
from .network import Incremental_Wrapper, Identity_Layer, Generator
from .loss import dafl_loss
import torch

def load_model(option, num_class, device):
    if 'cifar' in option.result['data']['data_type']:
        from .resnet_cifar import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
    else:
        from .resnet_imagenet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

    # Load Network Architecture
    if option.result['network']['network_type'] == 'resnet18':
        model_enc = ResNet18()
    elif option.result['network']['network_type'] == 'resnet34':
        model_enc = ResNet34()
    else:
        raise('Select Proper Network Type')

    # Training Type
    if option.result['train']['train_type'] == 'dafl':
        model_fc = nn.Linear(model_enc.num_feature, num_class, bias=True)
        model = Incremental_Wrapper(option, model_enc=model_enc, model_fc=model_fc, feature_out=True, device=device)
        generator = Generator(option)
        return [model, generator]

    elif option.result['train']['train_type'] == 'dream':
        model_fc = nn.Linear(model_enc.num_feature, num_class, bias=True)
        model = Incremental_Wrapper(option, model_enc=model_enc, model_fc=model_fc, feature_out=False, device=device)
        return [model]

    else:
        raise('Select Proper training type')


def load_optimizer(option, model_list, task_id):
    train_type = option.result['train']['train_type']

    if task_id == 0:
        params = model_list[0].parameters()
        optimizer = torch.optim.SGD(params, lr=option.result['train']['lr'], momentum=0.9, weight_decay=5e-4)
        return [optimizer]

    else:
        if train_type == 'dafl':
            assert len(model_list) == 2
            optimizer_S = torch.optim.SGD(model_list[0].parameters(), lr=option.result['train']['lr_s'], momentum=0.9, weight_decay=5e-4)
            optimizer_G = torch.optim.Adam(model_list[1].parameters(), lr=option.result['train']['lr_g'])
            return [optimizer_S, optimizer_G]

        elif train_type == 'dream':
            assert len(model_list) == 1
            optimizer = torch.optim.SGD(model_list[0].parameters(), lr=option.result['train']['lr'], momentum=0.9, weight_decay=5e-4)
            return [optimizer]

        else:
            raise('Select Proper Train Types')


def load_scheduler(option, optimizer_list, task_id):
    train_type = option.result['train']['train_type']

    if task_id == 0:
        if option.result['train']['scheduler'] == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_list[0], T_max=option.result['train']['total_epoch'])
        elif option.result['train']['scheduler'] == 'step':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_list[0], milestones=[48, 63, 80], gamma=0.2)
        elif option.result['train']['scheduler'] is None:
            scheduler = None
        else:
            raise('select proper scheduler')

        return [scheduler]

    else:
        if train_type == 'dafl':
            assert len(optimizer_list) == 2

            if option.result['train']['scheduler'] == 'cosine':
                scheduler_S = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_list[0], T_max=option.result['train']['total_epoch'])
            elif option.result['train']['scheduler'] == 'step':
                scheduler_S = torch.optim.lr_scheduler.MultiStepLR(optimizer_list[0], milestones=[48, 63, 80], gamma=0.2)
            elif option.result['train']['scheduler'] is None:
                scheduler_S = None
            else:
                raise ('select proper scheduler')

            scheduler_SG = torch.optim.lr_scheduler.MultiStepLR(optimizer_list[0], milestones=[800, 1600], gamma=0.1)
            return [scheduler_S, scheduler_SG]

        elif train_type == 'dream':
            if option.result['train']['scheduler'] == 'cosine':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_list[0], T_max=option.result['train']['total_epoch'])
            elif option.result['train']['scheduler'] == 'step':
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_list[0], milestones=[48, 63, 80], gamma=0.2)
            elif option.result['train']['scheduler'] is None:
                scheduler = None
            else:
                raise ('select proper scheduler')

            return [scheduler]


def load_loss(option, old_class, new_class, task_id):
    train_type = option.result['train']['train_type']

    if task_id == 0:
        criterion = nn.CrossEntropyLoss()
    else:
        if train_type == 'dafl':
            criterion = dafl_loss(old_class, new_class)
            # criterion = nn.CrossEntropyLoss()
        elif train_type == 'dream':
            criterion = nn.CrossEntropyLoss()
        else:
            raise('select proper train_type')

    return criterion