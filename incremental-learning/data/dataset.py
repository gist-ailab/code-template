from torchvision.transforms import transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np

def load_cifar():
    transform = transforms.Compose(
        [transforms.Resize((128,128)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    tr_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    return tr_dataset, val_dataset


def load_data(option, data_type='train'):
    tr_d, val_d = load_cifar()
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
        label += self.start
        return image, label