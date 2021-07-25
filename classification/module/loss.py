import torch.nn as nn
import torch.nn.functional as F
import torch

class distill_loss(nn.Module):
    def __init__(self, args):
        super(distill_loss, self).__init__()
        self.distill_type = args.distill_type
        self.distill_func = args.distill_func

        self.distill_loc = [int(loc.strip()) for loc in args.distill_loc.split(',')]
        self.distill_param = [float(param.strip()) for param in args.distill_param.split(',')]
        assert(len(self.distill_loc) == len(self.distill_param))

        self.temperature = args.temperature

        # Distillation Function
        if self.distill_func == 'l1':
            self.criterion = F.l1_loss
        elif self.distill_func == 'l2':
            self.criterion = F.mse_loss
        else:
            raise('Select proper distillation function')

    def forward(self, low_feature, high_feature):
        # Get the feature or attention value in the forward pass
        if self.distill_type == 'self-attention':
            low_imp = low_feature['feature']
            high_imp = low_feature['feature']

            low_imp[-1] = F.softmax(low_imp[-1] / self.temperature, dim=1)
            high_imp[-1] = F.softmax(high_imp[-1] / self.temperature, dim=1)

            low, high = [], []
            for ix, (l, h) in enumerate(zip(low_imp, high_imp)):
                if ix < 4:
                    low.append(torch.max(l, dim=1)[0])
                    high.append(torch.max(h,dim=1)[0])
                else:
                    low.append(l)
                    high.append(h)

        else:
            low = low_feature[self.distill_type]
            high = high_feature[self.distill_type]

            low[-1] = F.softmax(low[-1] / self.temperature, dim=1)
            high[-1] = F.softmax(high[-1] / self.temperature, dim=1)

        # Calculate the loss
        for ix, (loc, param) in enumerate(zip(self.distill_loc, self.distill_param)):
            l = low[loc]
            h = high[loc]

            d_loss = self.criterion(l, h)

            if ix == 0:
                loss_sum = d_loss * param
            else:
                loss_sum += d_loss * param

        return loss_sum


