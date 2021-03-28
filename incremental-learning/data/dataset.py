from torchvision.transforms import transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

def load_cifar10(option):
    tr_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
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
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
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

def load_data(option, data_type='train'):
    if option.result['data']['data_type'] == 'cifar10':
        tr_d, val_d = load_cifar10(option)
    elif option.result['data']['data_type'] == 'cifar100':
        tr_d, val_d = load_cifar10(option)
    elif option.result['data']['data_type'] == 'imagenet':
        tr_d, val_d = load_cifar10(option)
    else:
        raise('select appropriate dataset')

    if data_type == 'train':
        return tr_d
    else:
        return val_d


class IncrementalSet(Dataset):
    def __init__(self, dataset, start, target_list):
        self.dataset = dataset
        self.start = start

        self.dataset_label = np.array(self.dataset.targets)
        self.target_index = np.concatenate([np.where(self.dataset_label == ix)[0] for ix in target_list], axis=0)

    def __len__(self):
        return len(self.target_index)

    def __getitem__(self, index):
        image, label = self.dataset.__getitem__(self.target_index[index])
        return image, label