import torch
import numpy as np

def wmse(inputs, targets, weights):
    """
    Weighted Mean Squared Error
    """
    return torch.mean(torch.mul(weights,(inputs-targets).pow(2)))

def calc_cosine_sim(a,b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def calc_eucl_sim(a,b):
    return -np.linalg.norm(a-b)