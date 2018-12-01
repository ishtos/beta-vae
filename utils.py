import os
import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder


def reconstruction_loss(self, x, x_, distribution):
    batch_size = x.size(0)
    x_ = F.sigmoid(x_)
    reconstruction_loss = F.mse_loss(x_, x, size_average=False)
    return reconstruction_loss.div(batch_size)


def kl_divergence(self, mu, logvar):
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    kl_divergence = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    sum_kl_divergence = kl_divergence.sum(1).mean(0, True)
    # dimension_wise_kl_divergence = _kl_divergence.mean(0)
    # mean_kl_divergence = _kl_divergence.mean(1).mean(0, True)

    return sum_kl_divergence, # dimension_wise_kl_divergence, mean_kl_divergence


class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)

    def __getitem__(self, index):
        path = self.imgs[index][0]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img


def dataloader(dataset_dir, image_size, batch_size, num_workers):
    root = os.path.join(dataset_dir)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()])
    dataset = CustomImageFolder(root, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True)

    return loader
