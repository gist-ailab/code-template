import argparse
import os
import neptune
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from module.load_module import load_model, load_loss, load_optimizer, load_scheduler
from module.layers import CosineLinear
from utility.utils import config, train_module
from utility.earlystop import EarlyStopping

from data.dataset import load_data, IncrementalSet, transform_module

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from utility.distributed import apply_gradient_allreduce, reduce_tensor
import numpy as np

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main(rank, option, task_id, save_folder):
    # Basic Options
    if task_id == 0:
        resume = False
    else:
        resume = True
        resume_path = os.path.join(save_folder, 'task_%d_dict.pt' %(task_id-1))

    num_gpu = len(option.result['train']['gpu'].split(','))

    total_epoch = option.result['train']['total_epoch']
    multi_gpu = len(option.result['train']['gpu'].split(',')) > 1
    if multi_gpu:
        ddp = option.result['train']['ddp']
    else:
        ddp = False

    scheduler = option.result['train']['scheduler']
    batch_size, pin_memory = option.result['train']['batch_size'], option.result['train']['pin_memory']

    # Early Stop
    early_stop = option.result['train']['early']
    if early_stop == False:
        option.result['train']['patience'] = 100000

    # Hook
    hook = False

    # Load Model
    def calc_num_class(task_id):
        if task_id == 0:
            num_class = option.result['train']['num_init_segment']
        else:
            num_class = option.result['train']['num_init_segment'] + option.result['train']['num_segment'] * task_id
        return num_class

    if task_id == 0:
        new_class = calc_num_class(0)
        old_class = calc_num_class(0)
    else:
        new_class = calc_num_class(task_id)
        old_class = calc_num_class(task_id - 1)

    transform = transform_module(option)
    old_model = load_model(option, num_class=old_class, old_class=old_class, new_class=new_class, transform=transform, device=rank)
    criterion = load_loss(option, old_class, new_class)

    if resume:
        save_module = train_module(total_epoch, old_model, criterion, multi_gpu)
        save_module.import_module(resume_path)
        old_model.load_state_dict(save_module.save_dict['model'][0])

    # New Model
    if option.result['train']['pretrain_new_model'] and task_id > 0:
        new_model = deepcopy(old_model)

        if option.result['train']['train_type'] == 'rebalance':
            new_model.model_fc = CosineLinear(new_model.model_enc.num_feature, new_class)
        else:
            new_model.model_fc = nn.Linear(new_model.model_fc.in_features, new_class, bias=True)

        new_model.model_fc.weight.data[:old_class] = old_model.model_fc.weight.data
        if new_model.model_fc.bias is not None:
            new_model.model_fc.bias.data[:old_class] = old_model.model_fc.bias.data

        if option.result['train']['train_type'] == 'rebalance':
            hook = True
            new_model.register_hook()

    else:
        new_model = load_model(option, num_class=new_class, old_class=old_class, new_class=new_class, transform=transform, device=rank)

    save_module = train_module(total_epoch, new_model, criterion, multi_gpu)


    # Load Old Exemplary Samples
    if (option.result['exemplar']['num_exemplary'] > 0) and (task_id > 0):
        new_model.exemplar_list = torch.load(os.path.join(save_folder, 'task_%d_exemplar.pt' %(task_id-1)))
    else:
        new_model.exemplar_list = []


    # Multi-Processing GPUs
    if ddp:
        setup(rank, num_gpu)
        torch.manual_seed(0)
        torch.cuda.set_device(rank)

        old_model.to(rank)
        new_model.to(rank)

        old_model = DDP(old_model, device_ids=[rank])
        new_model = DDP(new_model, device_ids=[rank])

        old_model = apply_gradient_allreduce(old_model)
        new_model = apply_gradient_allreduce(new_model)

        criterion.to(rank)

    else:
        if multi_gpu:
            old_model = nn.DataParallel(old_model).to(rank)
            new_model = nn.DataParallel(new_model).to(rank)
        else:
            old_model = old_model.to(rank)
            new_model = new_model.to(rank)


    # Optimizer and Scheduler
    optimizer = load_optimizer(option, new_model.parameters())
    if scheduler is not None:
        scheduler = load_scheduler(option, optimizer)

    # Early Stopping
    early = EarlyStopping(patience=option.result['train']['patience'])

    # Dataset and DataLoader
    if task_id == 0:
        start = 0
        end = option.result['train']['num_init_segment']
    else:
        start = option.result['train']['num_init_segment'] + option.result['train']['num_segment'] * (task_id - 1)
        end = start + option.result['train']['num_segment']


    # Training Set
    tr_target_list = list(range(start, end))
    val_target_list = list(range(0, end))

    tr_dataset = load_data(option, data_type='train')
    ex_dataset = load_data(option, data_type='exemplar')
    tr_dataset = IncrementalSet(tr_dataset, ex_dataset, start, target_list=tr_target_list, shuffle_label=True)


    # Update the image size as a sample
    d_ex, _ = tr_dataset.__getitem__(0)
    new_model.update_datasize(d_ex.size())

    # Merge exemplar set into training dataset
    if (task_id > 0) and (option.result['exemplar']['num_exemplary'] > 0):
        if multi_gpu:
            new_model.module.get_aug_exemplar()
            tr_dataset.update_exemplar(new_model.module.exemplar_aug_list)
        else:
            new_model.get_aug_exemplar()
            tr_dataset.update_exemplar(new_model.exemplar_aug_list)

    tr_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=4*num_gpu)


    # Adversarial Attack










if __name__=='__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--save_dir', type=str, default='/HDD1/sung/checkpoint/')
    parser.add_argument('--exp_name', type=str, default='imagenet_norm')
    parser.add_argument('--exp_num', type=int, default=1)
    parser.add_argument('--log', type=lambda x: x.lower()=='true', default=True)
    args = parser.parse_args()

    # Configure
    save_folder = os.path.join(args.save_dir, args.exp_name, str(args.exp_num))
    os.makedirs(save_folder, exist_ok=True)
    option = config(save_folder)
    option.get_all_config()


    # Resume Configuration
    resume = option.result['train']['resume']

    if resume:
        resume_task_id = option.result['train']['resume_task_id']
    else:
        resume_task_id = 0

    resume_path = os.path.join(save_folder, 'task_%d_dict.pt' %(resume_task_id-1))
    config_path = os.path.join(save_folder, 'task_%d_config.json' %(resume_task_id-1))

    if resume:
        if (os.path.isfile(resume_path) == False) or (os.path.isfile(config_path) == False):
            resume = False
            resume_task_id = 0
        else:
            option = config(save_folder)
            option.import_config(config_path)


    # GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = option.result['train']['gpu']
    num_gpu = len(option.result['train']['gpu'].split(','))
    multi_gpu = num_gpu > 1
    if multi_gpu:
        ddp = option.result['train']['ddp']
    else:
        ddp = False


    # RUN
    if option.result['train']['only_init']:
        resume_task_id = 0
        num_task = 1
    else:
        num_task = 1 + int((option.result['data']['num_class'] - option.result['train']['num_init_segment']) / option.result['train']['num_segment'])


    # Task
    for task_id in range(resume_task_id, num_task):
        if ddp:
            mp.spawn(main, args=(option, task_id, save_folder, ), nprocs=num_gpu, join=True)
        else:
            main('cuda', option, task_id, save_folder)