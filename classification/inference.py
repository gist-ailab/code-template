import json
import numpy as np
import pickle
import argparse
import os
import neptune

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18

from module.load_module import load_model, load_loss, load_optimizer, load_scheduler
from utility.utils import config, train_module
from utility.earlystop import EarlyStopping

from data.dataset import load_data

from copy import deepcopy
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from utility.distributed import apply_gradient_allreduce, reduce_tensor


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main(rank, option, resume, save_folder):
    # Basic Options
    resume_path = os.path.join(save_folder, 'best_model.pt')

    num_gpu = len(option.result['train']['gpu'].split(','))

    multi_gpu = len(option.result['train']['gpu'].split(',')) > 1
    if multi_gpu:
        ddp = option.result['train']['ddp']
    else:
        ddp = False

    batch_size = option.result['train']['batch_size']

    # Logger
    if (rank == 0) or (rank == 'cuda'):
        neptune.init('sunghoshin/imp', api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiYzdlYWFkMjctOWExMS00YTRlLWI0MWMtY2FhNmIyNzZlYTIyIn0=')
        exp_name, exp_num = save_folder.split('/')[-2], save_folder.split('/')[-1]
        neptune.create_experiment(params={'exp_name':exp_name, 'exp_num':exp_num},
                                  tags=['inference:True'])

    # Load Model
    num_class = option.result['data']['num_class']
    model = load_model(option, num_class)
    criterion = load_loss(option)

    if resume:
        model.load_state_dict(torch.load(resume_path))

    # Multi-Processing GPUs
    if ddp:
        setup(rank, num_gpu)
        torch.manual_seed(0)
        torch.cuda.set_device(rank)

        model.to(rank)
        model = DDP(model, device_ids=[rank])
        model = apply_gradient_allreduce(model)

        criterion.to(rank)

    else:
        if multi_gpu:
            model = nn.DataParallel(model).to(rank)
        else:
            model = model.to(rank)

    # Dataset and DataLoader
    val_dataset = load_data(option, data_type='val')

    if ddp:
        val_sampler = torch.utils.data.distributed.DistributedSampler(dataset=val_dataset,
                                                                     num_replicas=num_gpu, rank=rank)

        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size,
                                                  shuffle=False, num_workers=4*num_gpu, pin_memory=True,
                                                  sampler=val_sampler)
    else:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4*num_gpu)


    # Training
    from module.trainer import naive_trainer
    model.eval()
    result = naive_trainer.imp_inference(option, rank, model, criterion, val_loader, neptune)

    if ddp:
        cleanup()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--save_dir', type=str, default='/home/sung/checkpoint/CL')
    parser.add_argument('--exp_name', type=str, default='cifar10-marker')
    parser.add_argument('--exp_num', type=int, default=0)

    parser.add_argument('--gpu', type=str, default='0,1,2')
    parser.add_argument('--data_type', type=str, default='cifar10')
    parser.add_argument('--batch', type=int, default=512)
    args = parser.parse_args()


    # Configure
    resume = True

    save_folder = os.path.join(args.save_dir, args.exp_name, str(args.exp_num))
    os.makedirs(save_folder, exist_ok=True)
    option = config(save_folder)


    # Resume Configuration
    config_path = os.path.join(save_folder, 'last_config.json')
    option.import_config(config_path)

    option.result['train']['gpu'] = args.gpu
    option.result['train']['batch_size'] = args.batch
    option.result['data_type'] = args.data_type


    # Data Directory
    option.result['data']['data_dir'] = os.path.join(option.result['data']['data_dir'], option.result['data']['data_type'])

    # GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = option.result['train']['gpu']
    num_gpu = len(option.result['train']['gpu'].split(','))
    multi_gpu = num_gpu > 1
    if multi_gpu:
        ddp = option.result['train']['ddp']
    else:
        ddp = False

    if ddp:
        mp.spawn(main, args=(option,resume,save_folder,), nprocs=num_gpu, join=True)
    else:
        main('cuda', option, resume, save_folder)

