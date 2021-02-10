import torch.nn.functional as F

def cross_entropy(output, target, ignore_index):
    return F.cross_entropy(output, target, ignore_index=ignore_index)
