import torch.nn as nn
import torch
import torch.nn.functional as F

class icarl_loss(nn.Module):
    def __init__(self):
        super(icarl_loss, self).__init__()
        self.loss_cls = nn.CrossEntropyLoss()
        self.loss_distill = nn.BCELoss()

    def forward(self, input, label):
        loss_cls = self.loss_cls(input, label)
        return loss_cls

    def forward_distill(self, old, new):
        loss_distill = sum(self.loss_distill(F.sigmoid(new[:,y]), F.sigmoid(old[:,y])) for y in range(old.size(1)))
        return loss_distill