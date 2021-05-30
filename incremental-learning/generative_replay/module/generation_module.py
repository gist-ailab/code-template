import torch
import torch.nn as nn
import collections
import random
import torch.nn.functional as F
import os

class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]

        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        # forcing mean and variance to match between two distributions
        # other ways might work better, e.g. KL divergence
        r_feature = torch.norm(module.running_var.data.type(var.type()) - var, 2) + torch.norm(
            module.running_mean.data.type(var.type()) - mean, 2)

        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()


class Generate_old(object):
    def __init__(self, option, save_folder, task_id, teacher, student, old_class, multi_gpu, random_label=False, device='cuda'):
        self.option = option
        self.save_folder = save_folder
        self.task_id = task_id

        self.teacher = teacher
        self.student = student

        self.old_class = old_class
        self.random_label = random_label

        self.generated_imgs = []

        self.device = device
        self.multi_gpu = multi_gpu


    def dream(self):
        num_images = self.option.result['train']['num_old_images'] * self.task_id
        batch_size = self.option.result['train']['gen_batch_size']

        num_iters, num_others = num_images // batch_size + 1, num_images % batch_size

        image_list, target_list = [], []

        os.makedirs(os.path.join(self.save_folder, 'exemplar'), exist_ok=True)

        for ix in range(num_iters):
            # Batch_size
            if ix == (num_iters - 1):
                bs = num_others
            else:
                bs = batch_size

            # Input Noise
            inputs = torch.randn((bs, 3, 32, 32), requires_grad=True, device=self.device, dtype=torch.float)
            optimizer = torch.optim.Adam([inputs], lr=self.option.result['train']['lr_gen'])

            kl_loss = nn.KLDivLoss(reduction='batchmean').to(self.device)

            best_cost = 1e6

            # initialize gaussian inputs
            inputs.data = torch.randn((bs, 3, 32, 32), requires_grad=True, device=self.device)

            # set up criteria for optimization
            criterion = nn.CrossEntropyLoss()

            optimizer.state = collections.defaultdict(dict)  # Reset state of optimizer

            # target outputs to generate
            if self.random_label:
                targets = torch.LongTensor([random.randint(0, int(self.old_class-1)) for _ in range(bs)]).to(self.device)
            else:
                targets = torch.LongTensor(list(range(self.old_class)) * int(bs // self.old_class) +\
                                           list(range(self.old_class))[:int(bs % self.old_class)]).to(self.device)

            ## Create hooks for feature statistics catching
            loss_r_feature_layers = []

            if self.multi_gpu:
                for module in self.teacher.module.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        loss_r_feature_layers.append(DeepInversionFeatureHook(module))
            else:
                for module in self.teacher.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        loss_r_feature_layers.append(DeepInversionFeatureHook(module))


            # setting up the range for jitter
            lim_0, lim_1 = 2, 2

            for epoch in range(self.option.result['train']['gen_epoch']):
                # apply random jitter offsets
                off1 = random.randint(-lim_0, lim_0)
                off2 = random.randint(-lim_1, lim_1)
                inputs_jit = torch.roll(inputs, shifts=(off1, off2), dims=(2, 3))

                # foward with jit images
                optimizer.zero_grad()
                self.teacher.zero_grad()
                outputs = self.teacher(inputs_jit)
                loss = criterion(outputs, targets)
                loss_target = loss.item()

                # competition loss, Adaptive DeepInvesrion
                competitive_scale = self.option.result['train']['competitive_scale']
                if competitive_scale != 0.0:
                    self.student.zero_grad()
                    outputs_student = self.student(inputs_jit)
                    T = 3.0

                    if 1:
                        # jensen shanon divergence:
                        # another way to force KL between negative probabilities
                        P = F.softmax(outputs_student / T, dim=1)
                        Q = F.softmax(outputs / T, dim=1)
                        M = 0.5 * (P + Q)

                        P = torch.clamp(P, 0.01, 0.99)
                        Q = torch.clamp(Q, 0.01, 0.99)
                        M = torch.clamp(M, 0.01, 0.99)
                        eps = 0.0
                        # loss_verifier_cig = 0.5 * kl_loss(F.log_softmax(outputs_verifier / T, dim=1), M) +  0.5 * kl_loss(F.log_softmax(outputs/T, dim=1), M)
                        loss_verifier_cig = 0.5 * kl_loss(torch.log(P + eps), M) + 0.5 * kl_loss(torch.log(Q + eps), M)
                        # JS criteria - 0 means full correlation, 1 - means completely different
                        loss_verifier_cig = 1.0 - torch.clamp(loss_verifier_cig, 0.0, 1.0)

                        loss = loss + competitive_scale * loss_verifier_cig

                # apply total variation regularization
                diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
                diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
                diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
                diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]
                loss_var = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
                loss = loss + self.option.result['train']['var_scale'] * loss_var

                # R_feature loss
                loss_distr = sum([mod.r_feature.to(self.device) for mod in loss_r_feature_layers])
                loss = loss + self.option.result['train']['bn_reg_scale'] * loss_distr  # best for noise before BN

                # l2 loss
                if 1:
                    loss = loss + self.option.result['train']['l2_coeff'] * torch.norm(inputs_jit, 2)

                if epoch % 200 == 0:
                    print(f"It {epoch}\t Losses: total: {loss.item():3.3f},\ttarget: {loss_target:3.3f} \tR_feature_loss unscaled:\t {loss_distr.item():3.3f}")

                if best_cost > loss.item():
                    best_cost = loss.item()
                    best_inputs = inputs.data

                loss.backward()
                optimizer.step()

            outputs = self.teacher(best_inputs)
            _, predicted_teach = outputs.max(1)

            outputs_student = self.student(best_inputs)
            _, predicted_std = outputs_student.max(1)

            print('Teacher correct out of {}: {}, loss at {}'.format(bs, predicted_teach.eq(targets).sum().item(), criterion(outputs, targets).item()))
            print('Student correct out of {}: {}, loss at {}'.format(bs, predicted_std.eq(targets).sum().item(), criterion(outputs_student, targets).item()))

            image_list.append(best_inputs.data.cpu().detach())
            target_list.append(targets.data.cpu().detach())

        image_list = torch.cat(image_list, dim=0)
        target_list = torch.cat(target_list, dim=0)

        self.generated_imgs = [(torch.unsqueeze(img,dim=0), target) for img, target in zip(image_list, target_list)]
        torch.save(self.generated_imgs, os.path.join(self.save_folder, 'exemplar', 'task_%d.pt' %(self.task_id - 1)))
