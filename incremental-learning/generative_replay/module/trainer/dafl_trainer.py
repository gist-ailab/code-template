import numpy as np
import torch
from tqdm import tqdm
from utility.distributed import apply_gradient_allreduce, reduce_tensor
import os
from torch.autograd import Variable
from copy import deepcopy
from module.load_module import load_model, load_loss, load_optimizer, load_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import os

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)

        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def train_generator(option, rank, epoch, new_model_list, old_model, criterion, optimizer_list, save_module):
    num_gpu = len(option.result['train']['gpu'].split(','))
    multi_gpu = num_gpu > 1

    # For Log
    mean_loss = 0.
    iter_size = 120

    for _ in tqdm(range(iter_size)):
        z = Variable(torch.randn(option.result['train']['batch_size'], option.result['train']['latent_dim'])).to(rank)

        for optimizer in optimizer_list:
            optimizer.zero_grad()

        gen_imgs = new_model_list[1](z)

        # Loss about Retaining Old Network Knowledge
        with torch.no_grad():
            output_old, feature_old = old_model(gen_imgs)
            pred = output_old.data.max(1)[1]

        loss_activation = -feature_old.abs().mean()
        loss_one_hot = nn.CrossEntropyLoss()(output_old, pred)
        softmax_o_T = torch.nn.functional.softmax(output_old, dim=1).mean(dim=0)
        loss_information_entropy = -(softmax_o_T * torch.log10(softmax_o_T)).sum()

        loss = loss_one_hot * 0.5 + loss_information_entropy * 5. + loss_activation * 0.1

        # Loss about Student
        new_out_gen, _ = new_model_list[0](gen_imgs.detach())
        loss += criterion(new_out_gen, output_old.detach())

        loss.backward()
        for optimizer in optimizer_list:
            optimizer.step()

        if (num_gpu > 1) and (option.result['train']['ddp']):
            mean_loss += reduce_tensor(loss.data, num_gpu).item()

        else:
            mean_loss += loss.item()

    # Train Result
    mean_loss /= iter_size

    # Saving Network Params
    save_module.save_dict['model'] = []
    for new_model in new_model_list:
        if multi_gpu:
            save_module.save_dict['model'].append(new_model.module.state_dict())
        else:
            save_module.save_dict['model'].append(new_model.state_dict())

    # Save Optimizer
    save_module.save_dict['optimizer'] = []
    for optimizer in optimizer_list:
        save_module.save_dict['optimizer'] = [optimizer.state_dict()]

    # Save Epoch
    save_module.save_dict['save_epoch'] = epoch

    # Logging
    if (rank == 0) or (rank == 'cuda'):
        print('Epoch-(%d/%d) - tr_gen_loss:%.3f' %(epoch, option.result['train']['gen_epoch']-1, mean_loss))
    return new_model_list, optimizer_list, save_module


def train_new_task(option, rank, epoch, task_id, new_model_list, old_model, criterion, optimizer_list, tr_loader, save_module):
    num_gpu = len(option.result['train']['gpu'].split(','))
    multi_gpu = num_gpu > 1

    # For Log
    mean_loss = 0.
    mean_acc1 = 0.
    mean_acc5 = 0.

    # Fix Generative Model
    new_model_list[1].eval()

    for tr_data in tqdm(tr_loader):
        input, label = tr_data
        input, label = input.to(rank), label.to(rank)

        optimizer_list[0].zero_grad()

        if task_id == 0:
            new_out, _ = new_model_list[0](input)
            loss = nn.CrossEntropyLoss()(new_out, label)

        else:
            z = Variable(torch.randn(input.size(0), option.result['train']['latent_dim'])).to(rank)
            with torch.no_grad():
                gen_imgs = new_model_list[1](z)

            # Get Old Features
            with torch.no_grad():
                output_old, feature_old = old_model(gen_imgs)

            # Loss about New Task
            new_out_gen, _ = new_model_list[0](gen_imgs.detach())
            new_out, _ = new_model_list[0](input)

            loss = criterion(new_out_gen, output_old)
            loss += nn.CrossEntropyLoss()(new_out, label)

        loss.backward()
        optimizer_list[0].step()
        acc_result = accuracy(new_out, label, topk=(1, 5))

        if (num_gpu > 1) and (option.result['train']['ddp']):
            mean_loss += reduce_tensor(loss.data, num_gpu).item()
            mean_acc1 += reduce_tensor(acc_result[0], num_gpu)
            mean_acc5 += reduce_tensor(acc_result[1], num_gpu)

        else:
            mean_loss += loss.item()
            mean_acc1 += acc_result[0]
            mean_acc5 += acc_result[1]

    # Train Result
    mean_acc1 /= len(tr_loader)
    mean_acc5 /= len(tr_loader)
    mean_loss /= len(tr_loader)

    # Saving Network Params
    save_module.save_dict['model'] = []
    for new_model in new_model_list:
        if multi_gpu:
            save_module.save_dict['model'].append(new_model.module.state_dict())
        else:
            save_module.save_dict['model'].append(new_model.state_dict())

    # Save Optimizer
    save_module.save_dict['optimizer'] = []
    for optimizer in optimizer_list:
        save_module.save_dict['optimizer'] = [optimizer.state_dict()]

    # Save Epoch
    save_module.save_dict['save_epoch'] = epoch

    # Logging
    if (rank == 0) or (rank == 'cuda'):
        print('Epoch-(%d/%d) - tr_ACC@1: %.2f, tr_ACC@5-%.2f, tr_loss:%.3f' %(epoch, option.result['train']['total_epoch']-1, \
                                                                            mean_acc1, mean_acc5, mean_loss))
    return new_model_list, optimizer_list, save_module


def validation(option, rank, epoch, new_model_list, val_loader, num_class):
    num_gpu = len(option.result['train']['gpu'].split(','))
    multi_gpu = num_gpu > 1

    # For Log
    mean_loss = 0.
    mean_acc1 = 0.
    mean_acc5 = 0.

    with torch.no_grad():
        for val_data in tqdm(val_loader):
            input, label = val_data
            input, label = input.to(rank), label.to(rank)

            output, _ = new_model_list[0](input)
            acc_result = accuracy(output, label, topk=(1, 5))

            if (num_gpu > 1) and (option.result['train']['ddp']):
                mean_acc1 += reduce_tensor(acc_result[0], num_gpu)
                mean_acc5 += reduce_tensor(acc_result[1], num_gpu)

            else:
                mean_acc1 += acc_result[0]
                mean_acc5 += acc_result[1]

        # Train Result
        mean_acc1 /= len(val_loader)
        mean_acc5 /= len(val_loader)

        # Logging
        if (rank == 0) or (rank == 'cuda'):
            print('Epoch-(%d/%d) - val_ACC@1: %.2f, val_ACC@5-%.2f, val_loss:%.3f' % (epoch, option.result['train']['total_epoch']-1, \
                                                                                    mean_acc1, mean_acc5, mean_loss))
    result = {'acc1':mean_acc1, 'acc5':mean_acc5, 'val_loss':mean_loss}
    return result


def run(option, new_model_list, old_model, new_class, old_class, tr_loader, val_loader, val_old_loader, tr_dataset, val_dataset, tr_target_list,
        val_target_list, optimizer_list, criterion, scaler, scheduler_list, early, early_stop, save_folder, save_module, multi_gpu, rank, task_id, ddp):
    old_model.eval()

    # Training Old Task Generator
    save_module.save_dict['scheduler'] = [None, None]

    if task_id > 0:
        for epoch in range(option.result['train']['gen_epoch']):
            # Training
            for new_model in new_model_list:
                new_model.train()
            new_model_list, optimizer_list, save_module = train_generator(option, rank, epoch, new_model_list, old_model, criterion, optimizer_list, save_module)

            # Validation
            for new_model in new_model_list:
                new_model.eval()
            _ = validation(option, rank, epoch, new_model_list, val_old_loader, num_class=old_class)


            # Scheduler
            if scheduler_list is not None and scheduler_list[1] is not None:
                scheduler_list[1].step()
                save_module.save_dict['scheduler'][1] = scheduler_list[1]


    # Training New Task
    if task_id > 0:
        new_model_list[1].eval() # -> Fix Gradient of Generator

    for epoch in range(0, save_module.total_epoch):
        # Training
        new_model_list[0].train()
        new_model_list, optimizer_list, save_module = train_new_task(option, rank, epoch, task_id, new_model_list, old_model, \
                                                                     criterion, optimizer_list, tr_loader, save_module)

        # Validate
        new_model_list[0].eval()
        result = validation(option, rank, epoch, new_model_list, val_loader, num_class=new_class)

        # Scheduler
        if scheduler_list is not None and scheduler_list[0] is not None:
            scheduler_list[0].step()
            save_module.save_dict['scheduler'][0] = scheduler_list[0]


        # Early Stop
        param_list = []
        for new_model in new_model_list:
            if multi_gpu:
                param_list.append(deepcopy(new_model.module.state_dict()))
            else:
                param_list.append(deepcopy(new_model.state_dict()))

        if option.result['train']['early_criterion_loss']:
            early(result['val_loss'], param_list, result)
        else:
            early(-result['acc1'], param_list, result)

        if early.early_stop == True:
            break

    return early, save_module, option