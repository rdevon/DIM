"""Miscilaneous functions.

"""

import torch


def log_sum_exp(x, axis=None):
    """Log sum exp function

    Args:
        x: Input.
        axis: Axis over which to perform sum.

    Returns:
        torch.Tensor: log sum exp

    """
    x_max = torch.max(x, axis)[0]
    y = torch.log((torch.exp(x - x_max)).sum(axis)) + x_max
    return y


def random_permute(X):
    """Randomly permutes a tensor.

    Args:
        X: Input tensor.

    Returns:
        torch.Tensor

    """
    X = X.transpose(1, 2)
    b = torch.rand((X.size(0), X.size(1))).cuda()
    idx = b.sort(0)[1]
    adx = torch.range(0, X.size(1) - 1).long()
    X = X[idx, adx[None, :]].transpose(1, 2)
    return X
