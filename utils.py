from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from tensorboardX import SummaryWriter


def reconstruction_loss(self, x, x_, distribution):
    batch_size = x.size(0)
    if distribution == 'bernoulli':
        _reconstruction_loss = F.binary_cross_entropy_with_logits(x_, x, size_average=False)
    elif distribution == 'gaussian':
        x_ = F.sigmoid(x_)
        _reconstruction_loss = F.mse_loss(x_, x, size_average=False)
    return _reconstruction_loss.div(batch_size)


def kl_divergence(self, mu, logvar):
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    _kl_divergence = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kl_divergence = _kl_divergence.sum(1).mean(0, True)
    dimension_wise_kl_divergence = _kl_divergence.mean(0)
    mean_kl_divergence = _kl_divergence.mean(1).mean(0, True)

    return total_kl_divergence, dimension_wise_kl_divergence, mean_kl_divergence
