import sys
sys.path.append('../')
from MulticoreTSNE import MulticoreTSNE as TSNE
from matplotlib import pyplot as plt
import torch
import os
from torch.nn import DataParallel
import os
from tqdm import tqdm
import numpy as np

class gradcam():
    pass


def tsne(model, loader, device='cuda:0'):
    model = model.to(device)
    model.eval()

    target_list, output_list = [], []

    for batch_idx, data in enumerate(tqdm(loader)):
        inputs = data[0].to(device)
        target = data[1].detach().numpy()

        with torch.no_grad():
            outputs = model(inputs)
            outputs = outputs.mean(dim=[2, 3])
            outputs = outputs.cpu().detach().numpy()

        target_list.append(target)
        output_list.append(outputs)

    target_list = np.concatenate(target_list, axis=0)
    output_list = np.concatenate(output_list, axis=0)

    # t-SNE
    embeddings = TSNE(n_jobs=8, n_components=2).fit_transform(output_list)
    return embeddings, target_list
