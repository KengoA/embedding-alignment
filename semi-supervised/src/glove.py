import os
import json
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# Runtime -> Change runtime type -> GPU
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    device = 'cuda:0'
else:
    device = 'cpu'


def f(x, x_max=100, alpha=0.75):
    w = torch.div(x, x_max).pow(alpha)
    w = torch.min(w, torch.ones_like(x))
    return w


def wmse(inputs, targets, weights):
    return torch.mean(torch.mul(weights,(inputs-targets).pow(2)))


class Dataset(torch.utils.data.Dataset):
    def __init__(self, cooccur_path, random_id):
        self._cooccur_path = cooccur_path
        self._random_id = random_id
        self._assign_ids()

    def _assign_ids(self):
        cooccur = json.load(open(self._cooccur_path,'r'))
        self._vocab = sorted(list(set(cooccur['id_i']+cooccur['id_j'])))
        if self._random_id:
            np.random.shuffle(self._vocab)
        
        self._name_to_id = {name: i for i, name in enumerate(self._vocab)}
        self._id_to_name = {i: name for i, name in enumerate(self._vocab)}

        self._id_i = torch.LongTensor([self._name_to_id[w] for w in cooccur['id_i']]).to(device)
        self._id_j = torch.LongTensor([self._name_to_id[w] for w in cooccur['id_j']]).to(device)
        self._X_ij = torch.FloatTensor(cooccur['X_ij']).to(device)

    def __len__(self):
        return self._X_ij.size(0)

    def __getitem__(self, index):
        id_i = self._id_i[index]
        id_j = self._id_j[index]
        X_ij = self._X_ij[index]

        return id_i, id_j, X_ij


class GloVe(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(GloVe, self).__init__()
        self.wi = nn.Embedding(vocab_size, embedding_dim)
        self.wj = nn.Embedding(vocab_size, embedding_dim)
        self.bi = nn.Embedding(vocab_size, 1)
        self.bj = nn.Embedding(vocab_size, 1)

        self.wi.weight.data.uniform_(-1,1)
        self.wj.weight.data.uniform_(-1,1)
        self.bi.weight.data.zero_()
        self.bj.weight.data.zero_()

    def forward(self, id_i, id_j):
        wi = self.wi(id_i)
        wj = self.wj(id_j)
        bi = self.bi(id_i)
        bj = self.bj(id_j)
        
        return wi, wj, bi, bj


if __name__ == "__main__":
    ROOT_DIR =  '../assets/cooccur/'
    modality = 'text'

    cooccur = json.load(open(ROOT_DIR+f'/cooccur_{modality}_name.json', 'r'))
    data = {'id_i':[], 'id_j':[], 'X_ij':[]}

    for center in cooccur.keys():
        for context in cooccur[center].keys():
            if cooccur[center][context] > 0:
                data['id_i'].append(center)
                data['id_j'].append(context)
                data['X_ij'].append(cooccur[center][context])

    with open(ROOT_DIR+f'cooccur_{modality}.json','w') as w:
        json.dump(data, w)

    batch_size = 256

    dataset = Dataset(ROOT_DIR+f'cooccur_{modality}.json', random_id=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)