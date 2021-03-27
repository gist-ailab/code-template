import json
import numpy as np
import pickle
import argparse
import os
import neptune
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18

from module import trainer
from module.load_module import load_model, load_loss, load_optimizer, load_scheduler
from utility.utils import config, train_module

from data.dataset import load_data, IncrementalSet

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main(rank, option, task_id, resume, save_folder):
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


    # Logger
    if (rank == 0) or (rank == 'cuda'):
        neptune.init('sunghoshin/imp', api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiYzdlYWFkMjctOWExMS00YTRlLWI0MWMtY2FhNmIyNzZlYTIyIn0=')
        exp_name, exp_num = save_folder.split('/')[-2], save_folder.split('/')[-1]
        neptune.create_experiment(params={'exp_name':exp_name, 'exp_num':exp_num}, tags=[])


    # Load Model
    old_model = load_model(option)
    criterion = load_loss(option)

    if resume:
        save_module = train_module(total_epoch, old_model, criterion, multi_gpu)
        save_module.import_module(resume_path)
        old_model.load_state_dict(save_module.save_dict['model'][0])


    # New Model
    if option.result['network']['pretrain_new_model']:
        new_model = deepcopy(old_model)
    else:
        new_model = load_model(option)

    save_module = train_module(total_epoch, new_model, criterion, multi_gpu)


    # Multi-Processing GPUs
    if ddp:
        setup(rank, num_gpu)
        torch.manual_seed(0)
        torch.cuda.set_device(rank)

        old_model.to(rank)
        new_model.to(rank)

        old_model = DDP(old_model, device_ids=[rank])
        new_model = DDP(new_model, device_ids=[rank])

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
        scheduler = load_scheduler(option)


    # Dataset and DataLoader
    if task_id == 0:
        start = 0
        end = option.result['train']['num_init_segment']
    else:
        start = option.result['train']['num_init_segment'] + option.result['train']['num_segment'] * (task_id - 1)
        end = start + option.result['train']['num_segment']

    target_list = list(range(start, end))

    tr_dataset = load_data(option, data_type='train')
    tr_dataset = IncrementalSet(tr_dataset, start, target_list=target_list)
    val_dataset = load_data(option, data_type='val')
    val_dataset = IncrementalSet(val_dataset, start, target_list=target_list)

    if ddp:
        tr_sampler = torch.utils.data.distributed.DistributedSampler(dataset=tr_dataset,
                                                                     num_replicas=num_gpu, rank=rank)
        val_sampler = torch.utils.data.distributed.DistributedSampler(dataset=val_dataset,
                                                                     num_replicas=num_gpu, rank=rank)

        tr_loader = torch.utils.data.DataLoader(dataset=tr_dataset, batch_size=batch_size,
                                                  shuffle=False, num_workers=4*num_gpu, pin_memory=pin_memory,
                                                  sampler=tr_sampler)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size,
                                                  shuffle=False, num_workers=4*num_gpu, pin_memory=pin_memory,
                                                  sampler=val_sampler)
    else:
        tr_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=4*num_gpu)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=4*num_gpu)


    # Mixed Precision
    mixed_precision = option.result['train']['mixed_precision']
    if mixed_precision:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None


    # Training
    old_model.eval()
    for epoch in range(0, save_module.total_epoch):
        new_model.train()
        new_model, optimizer, save_module = trainer.train(option, rank, epoch, task_id, new_model, old_model, criterion, optimizer, tr_loader, scaler, save_module, neptune)

        new_model.eval()
        result = trainer.validation(option, rank, epoch, task_id, new_model, old_model, criterion, val_loader, neptune)

        if scheduler is not None:
            scheduler.step()
            save_module.save_dict['scheduler'] = [scheduler.state_dict()]
        else:
            save_module.save_dict['scheduler'] = None


    # Save the Result
    save_module_path = os.path.join(save_folder, 'task_%d_dict.pt' %task_id)
    save_module.export_module(save_module_path)

    save_config_path = os.path.join(save_folder, 'task_%d_config.json' %task_id)
    option.export_config(save_config_path)

    if ddp:
        cleanup()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--save_dir', type=str, default='/HDD1/sung/checkpoint/')
    parser.add_argument('--exp_name', type=str, default='imagenet_norm')
    parser.add_argument('--exp_num', type=int, default=1)
    args = parser.parse_args()

    # Configure
    save_folder = os.path.join(args.save_dir, args.exp_name, str(args.exp_num))
    os.makedirs(save_folder, exist_ok=True)
    option = config(save_folder)
    option.get_config_data()
    option.get_config_network()
    option.get_config_train()

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

    for task_id in range(resume_task_id, num_task):
        if ddp:
            mp.spawn(main, args=(option, task_id, resume, save_folder,), nprocs=num_gpu, join=True)
        else:
            main('cuda', option, task_id, resume, save_folder,)

