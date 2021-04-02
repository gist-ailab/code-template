import torch
import torch.nn as nn
import numpy as np


def split_model(model):
    pass



class Identity_Layer(nn.Module):
    def __init__(self):
        super(Identity_Layer, self).__init__()

    def forward(self, x):
        return x


class Incremental_Wrapper(nn.Module):
    def __init__(self, option, model_enc, model_fc):
        super(Incremental_Wrapper, self).__init__()
        self.option = option
        self.model_enc = model_enc
        self.model_fc = model_fc

        self.exemplar_list = []

    def forward(self, image):
        out1 = self.model_enc(image)
        out2 = self.model_fc(out1)
        return out2


class Icarl_Wrapper(Incremental_Wrapper):
    def __init__(self, option, model_enc, model_fc):
        super(Icarl_Wrapper, self).__init__(option, model_enc, model_fc)
        self.option = option
        self.model_enc = model_enc

        self.bn = nn.BatchNorm1d(model_enc.fc.out_features, momentum=0.01)
        self.ReLU = nn.ReLU()

        self.model_fc = model_fc

        self.exemplar_list = []

    def forward(self, image):
        out1 = self.model_enc(image)
        imp = self.ReLU(self.bn(out1))
        out2 = self.model_fc(imp)
        return out2

    def feature_extractor(self, image):
        out1 = self.model_enc(image)
        return out1

    def get_new_exemplar(self, data, m, rank):
        # if self.option.result['train']['train_type'] == 'icarl':
        # Compute and cache features for each example
        features = []

        self.model_enc.eval()
        self.model_fc.eval()

        for img, label in data:
            x = img.to(rank)
            x.requires_grad = False
            with torch.no_grad():
                feature = self.feature_extractor(x.unsqueeze(0)).data.cpu().numpy()
            feature = feature / np.linalg.norm(feature)  # Normalize
            features.append(feature[0])

        features = np.array(features)
        class_mean = np.mean(features, axis=0)
        class_mean = class_mean / np.linalg.norm(class_mean)  # Normalize

        exemplar_set = []
        exemplar_features = []  # list of Variables of shape (feature_size,)
        for k in range(m):
            S = np.sum(exemplar_features, axis=0)
            phi = features
            mu = class_mean
            mu_p = 1.0 / (k + 1) * (phi + S)
            mu_p = mu_p / np.linalg.norm(mu_p)
            i = np.argmin(np.sqrt(np.sum((mu - mu_p) ** 2, axis=1)))

            exemplar_set.append(data[i])
            exemplar_features.append(features[i])

        self.exemplar_list.append(exemplar_set)


    def update_center(self, rank):
        exemplar_means = []

        self.model_enc.eval()
        self.model_fc.eval()

        for P_y in self.exemplar_list:
            features = []
            # Extract feature for each exemplar in P_y
            for img, label in P_y:
                ex = img.to(rank)
                ex.requires_grad = False
                with torch.no_grad():
                    feature = self.feature_extractor(ex.unsqueeze(0)).detach().cpu()
                feature = feature.squeeze()
                feature.data = feature.data / feature.data.norm()  # Normalize
                features.append(feature)
            features = torch.stack(features)
            mu_y = features.mean(0).squeeze()
            mu_y.data = mu_y.data / mu_y.data.norm()  # Normalize
            del features
            exemplar_means.append(mu_y)

        self.exemplar_means = exemplar_means


    def icarl_classify(self, x):
        batch_size = x.size(0)

        self.model_enc.eval()
        self.model_fc.eval()

        exemplar_means = self.exemplar_means
        means = torch.stack(exemplar_means)  # (n_classes, feature_size)
        means = torch.stack([means] * batch_size)  # (batch_size, n_classes, feature_size)
        means = means.transpose(1, 2).to(x.device)  # (batch_size, feature_size, n_classes)

        feature = self.feature_extractor(x)  # (batch_size, feature_size)
        for i in range(feature.size(0)):  # Normalize
            feature.data[i] = feature.data[i] / feature.data[i].norm()
        feature = feature.unsqueeze(2)  # (batch_size, feature_size, 1)
        feature = feature.expand_as(means)  # (batch_size, feature_size, n_classes)

        dists = (feature - means).pow(2).sum(1).squeeze()  # (batch_size, n_classes)
        neg_dists = -dists
        return neg_dists


    def reduce_old_exemplar(self, m):
        for ix in range(len(self.exemplar_list)):
            self.exemplar_list[ix] = self.exemplar_list[ix][:m]

    def my_hook(self, grad):
        grad_clone = grad.clone()
        grad_clone[:self.old_class] = 0
        return grad_clone

    def register_hook(self, n_old_class):
        self.old_class = n_old_class
        for param in self.model_fc.parameters():
            param.register_hook(self.my_hook)

