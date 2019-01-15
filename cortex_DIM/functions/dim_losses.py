'''cortex_DIM losses.

'''

import math

import torch
import torch.nn.functional as F

from cortex_DIM.functions.gan_losses import get_positive_expectation, get_negative_expectation


def fenchel_dual_loss(l, g, measure=None):
    '''Computes the f-divergence distance between positive and negative joint distributions.

    Divergences supported are Jensen-Shannon `JSD`, `GAN` (equivalent to JSD),
    Squared Hellinger `H2`, Chi-squeared `X2`, `KL`, and reverse KL `RKL`.

    Args:
        l: Local feature map.
        g: Global features.
        measure: f-divergence measure.

    Returns:
        torch.Tensor: Loss.

    '''
    N, local_units, n_locs = l.size()
    l = l.permute(0, 2, 1)
    l = l.reshape(-1, local_units)

    u = torch.mm(g, l.t())
    u = u.reshape(N, N, -1)
    mask = torch.eye(N).cuda()
    n_mask = 1 - mask

    E_pos = get_positive_expectation(u, measure, average=False).mean(2)
    E_neg = get_negative_expectation(u, measure, average=False).mean(2)
    E_pos = (E_pos * mask).sum() / mask.sum()
    E_neg = (E_neg * n_mask).sum() / n_mask.sum()
    loss = E_neg - E_pos
    return loss


def multi_fenchel_dual_loss(l, m, measure=None):
    '''Computes the f-divergence distance between positive and negative joint distributions.

    Used for multiple globals.

    Divergences supported are Jensen-Shannon `JSD`, `GAN` (equivalent to JSD),
    Squared Hellinger `H2`, Chi-squeared `X2`, `KL`, and reverse KL `RKL`.

    Args:
        l: Local feature map.
        m: Multiple globals feature map.
        measure: f-divergence measure.

    Returns:
        torch.Tensor: Loss.

    '''
    N, units, n_locals = l.size()
    n_multis = m.size(2)

    l = l.view(N, units, n_locals)
    l = l.permute(0, 2, 1)
    l = l.reshape(-1, units)

    m = m.view(N, units, n_multis)
    m = m.permute(0, 2, 1)
    m = m.reshape(-1, units)

    u = torch.mm(m, l.t())
    u = u.reshape(N, n_multis, N, n_locals).permute(0, 2, 3, 1)

    mask = torch.eye(N).cuda()
    n_mask = 1 - mask

    E_pos = get_positive_expectation(u, measure, average=False).mean(2).mean(2)
    E_neg = get_negative_expectation(u, measure, average=False).mean(2).mean(2)
    E_pos = (E_pos * mask).sum() / mask.sum()
    E_neg = (E_neg * n_mask).sum() / n_mask.sum()
    loss = E_neg - E_pos
    return loss


def nce_loss(l, g):
    '''Computes the noise contrastive estimation-based loss.

    Args:
        l: Local feature map.
        g: Global features.

    Returns:
        torch.Tensor: Loss.

    '''
    N, local_units, n_locs = l.size()
    l_p = l.permute(0, 2, 1)
    u_p = torch.matmul(l_p, g.unsqueeze(dim=2))

    l_n = l_p.reshape(-1, local_units)
    u_n = torch.mm(g, l_n.t())
    u_n = u_n.reshape(N, N, n_locs)

    mask = torch.eye(N).unsqueeze(dim=2).cuda()
    n_mask = 1 - mask

    u_n = (n_mask * u_n) - (10. * (1 - n_mask))  # mask out "self" examples
    u_n = u_n.reshape(N, -1).unsqueeze(dim=1).expand(-1, n_locs, -1)

    pred_lgt = torch.cat([u_p, u_n], dim=2)
    pred_log = F.log_softmax(pred_lgt, dim=2)
    loss = -pred_log[:, :, 0].mean()
    return loss


def multi_nce_loss(l, m):
    '''

    Used for multiple globals.

    Args:
        l: Local feature map.
        m: Multiple globals feature map.

    Returns:
        torch.Tensor: Loss.

    '''
    N, units, n_locals = l.size()
    _, _ , n_multis = m.size()

    l = l.view(N, units, n_locals)
    m = m.view(N, units, n_multis)
    l_p = l.permute(0, 2, 1)
    m_p = m.permute(0, 2, 1)
    u_p = torch.matmul(l_p, m).unsqueeze(2)

    l_n = l_p.reshape(-1, units)
    m_n = m_p.reshape(-1, units)
    u_n = torch.mm(m_n, l_n.t())
    u_n = u_n.reshape(N, n_multis, N, n_locals).permute(0, 2, 3, 1)

    mask = torch.eye(N)[:, :, None, None].cuda()
    n_mask = 1 - mask

    u_n = (n_mask * u_n) - (10. * (1 - n_mask))  # mask out "self" examples
    u_n = u_n.reshape(N, N * n_locals, n_multis).unsqueeze(dim=1).expand(-1, n_locals, -1, -1)

    pred_lgt = torch.cat([u_p, u_n], dim=2)
    pred_log = F.log_softmax(pred_lgt, dim=2)
    loss = -pred_log[:, :, 0].mean()

    return loss


def donsker_varadhan_loss(l, g):
    '''

    Args:
        l: Local feature map.
        g: Global features.

    Returns:
        torch.Tensor: Loss.

    '''
    N, local_units, n_locs = l.size()
    l = l.permute(0, 2, 1)
    l = l.reshape(-1, local_units)

    u = torch.mm(g, l.t())
    u = u.reshape(N, N, n_locs)

    mask = torch.eye(N).cuda()
    n_mask = (1 - mask)[:, :, None]

    E_pos = (u.mean(2) * mask).sum() / mask.sum()

    u -= 100 * (1 - n_mask)
    u_max = torch.max(u)
    E_neg = torch.log((n_mask * torch.exp(u - u_max)).sum() + 1e-6) + u_max - math.log(n_mask.sum())
    loss = E_neg - E_pos
    return loss


def multi_donsker_varadhan_loss(l, m):
    '''

    Used for multiple globals.

    Args:
        l: Local feature map.
        m: Multiple globals feature map.

    Returns:
        torch.Tensor: Loss.

    '''
    N, units, n_locals = l.size()
    n_multis = m.size(2)

    l = l.view(N, units, n_locals)
    l = l.permute(0, 2, 1)
    l = l.reshape(-1, units)

    m = m.view(N, units, n_multis)
    m = m.permute(0, 2, 1)
    m = m.reshape(-1, units)

    u = torch.mm(m, l.t())
    u = u.reshape(N, n_multis, N, n_locals).permute(0, 2, 3, 1)

    mask = torch.eye(N).cuda()
    n_mask = 1 - mask

    E_pos = (u.mean(2) * mask).sum() / mask.sum()

    u -= 100 * (1 - n_mask)
    u_max = torch.max(u)
    E_neg = torch.log((n_mask * torch.exp(u - u_max)).sum() + 1e-6) + u_max - math.log(n_mask.sum())
    loss = E_neg - E_pos
    return loss