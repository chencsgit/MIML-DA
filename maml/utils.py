import subprocess

import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

def accuracy(preds, y):
    _, preds = torch.max(preds.data, 1)
    total = y.size(0)
    correct = (preds == y).sum().float()
    return correct / total

def optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def get_git_revision_hash():
    return str(subprocess.check_output(['git', 'rev-parse', 'HEAD']))

class ShannonEntropyLoss(nn.Module):
    def __init__(self):
        super(ShannonEntropyLoss, self).__init__()

    def forward(self, inputs):
        # print('inputs.shape:', inputs.shape)
        entropy = F.softmax(inputs, dim=-1) * F.log_softmax(inputs, dim=-1)
        entropys = -1.0 * entropy.sum()
        return entropys

def shannon_entropy(inputs):
    criterion = ShannonEntropyLoss()
    return criterion(inputs)