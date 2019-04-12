"""Miscilaneous functions.

"""

import math

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


def ms_ssim(X_a, X_b, window_size=11, size_average=True, C1=0.01**2, C2=0.03**2):
    """
    Taken from Po-Hsun-Su/pytorch-ssim
    """

    channel = X_a.size(1)

    def gaussian(sigma=1.5):
        gauss = torch.Tensor(
            [math.exp(-(x - window_size // 2) **
                      2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window():
        _1D_window = gaussian(window_size).unsqueeze(1)
        _2D_window = _1D_window.mm(
            _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = torch.Tensor(
            _2D_window.expand(channel, 1, window_size,
                              window_size).contiguous())
        return window.cuda()

    window = create_window()

    mu1 = torch.nn.functional.conv2d(X_a, window,
                                     padding=window_size // 2, groups=channel)
    mu2 = torch.nn.functional.conv2d(X_b, window,
                                     padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = torch.nn.functional.conv2d(
        X_a * X_a, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(
        X_b * X_b, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = torch.nn.functional.conv2d(
        X_a * X_b, window, padding=window_size // 2, groups=channel) - mu1_mu2

    ssim_map = (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) /
                ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
