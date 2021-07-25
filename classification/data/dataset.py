from torchvision.transforms import transforms
import torchvision
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from random import shuffle
import torch
import random

def load_cifar10(option):
    tr_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    tr_dataset = torchvision.datasets.CIFAR10(root=option.result['data']['data_dir'], train=True, download=True, transform=tr_transform)
    val_dataset = torchvision.datasets.CIFAR10(root=option.result['data']['data_dir'], train=False, download=True, transform=val_transform)
    return tr_dataset, val_dataset

def load_cifar100(option):
    tr_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    tr_dataset = torchvision.datasets.CIFAR100(root=option.result['data']['data_dir'], train=True, download=True, transform=tr_transform)
    val_dataset = torchvision.datasets.CIFAR100(root=option.result['data']['data_dir'], train=False, download=True, transform=val_transform)
    return tr_dataset, val_dataset

def load_imagenet(option):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    tr_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    tr_dataset = torchvision.datasets.ImageFolder(os.path.join(option.result['data']['data_dir'], 'train'), transform=tr_transform)
    val_dataset = torchvision.datasets.ImageFolder(os.path.join(option.result['data']['data_dir'], 'val'), transform=val_transform)
    return tr_dataset, val_dataset

def load_tiny_imagenet(option):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    tr_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Scale(64),
        transforms.ToTensor(),
        normalize,
    ])

    tr_dataset = torchvision.datasets.ImageFolder(os.path.join(option.result['data']['data_dir'], 'train'), transform=tr_transform)
    val_dataset = torchvision.datasets.ImageFolder(os.path.join(option.result['data']['data_dir'], 'val'), transform=val_transform)
    return tr_dataset, val_dataset


def load_data(option, data_type='train'):
    if option.result['data']['data_type'] == 'cifar10':
        tr_d, val_d = load_cifar10(option)
    elif option.result['data']['data_type'] == 'cifar100':
        tr_d, val_d = load_cifar100(option)
    elif option.result['data']['data_type'] == 'imagenet':
        tr_d, val_d = load_imagenet(option)
    elif option.result['data']['data_type'] == 'tiny_imagenet':
        tr_d, val_d = load_tiny_imagenet(option)
    else:
        raise('select appropriate dataset')

    if data_type == 'train':
        return tr_d
    else:
        return val_d


class IncrementalSet(Dataset):
    def __init__(self, dataset, target_list, shuffle_label=False, prop=1.):
        self.dataset = dataset
        self.dataset_label = np.array(self.dataset.targets)

        # Select Target Index
        self.target_index = []
        for ix in target_list:
            ix_index = np.where(self.dataset_label == ix)[0]

            np.random.seed(100)
            select_num = int(len(ix_index) * prop)
            ix_index = np.random.choice(ix_index, select_num, replace=False)
            self.target_index.append(ix_index)

        self.target_index = np.concatenate(self.target_index, axis=0)

        # For Matching Class ID sequentially (0, 1, ... N)
        self.target_dict = {}
        for ix, target in enumerate(target_list):
            self.target_dict[target] = ix
        self.index_list = list(range(len(self.target_index)))

        # Shuffle
        self.shuffle = shuffle_label

        random.seed(100)
        if self.shuffle:
            shuffle(self.index_list)

    def get_image_class(self, label):
        self.target_label_index = np.where(self.dataset_label == label)[0]
        return [self.dataset_exemplar.__getitem__(index) for index in self.target_label_index]

    def __len__(self):
        return len(self.target_index)

    def __getitem__(self, index):
        index = self.index_list[index]
        image, label = self.dataset.__getitem__(self.target_index[index])
        label = torch.Tensor([self.target_dict[int(label)]]).long().item()
        return image, label