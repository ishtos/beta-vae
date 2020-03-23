import os
import sys
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from torchsummary import summary
from tensorboardX import SummaryWriter

from model import BetaVAE
from utils import reconstruction_loss, kl_divergence


def main(args):
    # settings
    epochs = args.epochs
    batch_size = args.batch_size
    global_step = 0

    dataset_dir = args.dataset_dir
    dataset = args.dataset
    image_size = args.image_size
    num_workers = args.num_workers

    n_latent = args.n_latent
    beta = args.beta
    beta1 = args.beta1
    beta2 = args.beta2

    ckpt_dir = args.ckpt_dir
    ckpt_name = args.ckpt_name

    # load data
    loader = dataloader(dataset_dir, image_size, batch_size, num_workers)

    # device
    if torch.cuda.is_available():
        print('Cuda is available. GPU MODE!')
        device = 'cuda'
    else:
        print('Cuda is not available. CPU MODE!')
        device = 'cpu'

    net = BetaVAE().to(device)
    optim = optim.Adam(net.parameters(), lr=lr, betas=(beta1, beta2))

    
    for _ in epochs:
        global_step += train(loader, net, optim, beta, device, global_step, writer)


def train(self, loader, net, optim, beta, device, global_step, writer):
    progress = tqdm(loader)
    for i, x in enumerate(progress):
        step = global_step + (i + 1)
        x = x.to(device)
        x_, mu, logvar = net(x)
        reconstruction_loss = reconstruction_loss(x, x_)
        sum_kl_divergence = kl_divergence(mu, logvar)

        beta_vae_loss = reconstruction_loss + beta*sum_kl_divergence

        optim.zero_grad()
        beta_vae_loss.backward()
        optim.step()

        writer.add_scalar('sum_kl_divergence', kl_divergence, step)
        writer.add_scalar('reconstruction_loss', _reconstruction_loss, step)
        writer.add_scalar('beta_vae_loss', beta_vae_loss, step)

        if i % 100 == 0:
            real_images = make_grid(x, nrow=2, normalize=True)
            reconstruction_images = make_grid(x_r, nrow=2, normalize=True)
            writer.add_image('real_images', real_images, step)
            writer.add_image('reconstruction_image', rcnst_images, step)

    return step


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    # related to train setting
    parser.add_argument(
        '--epochs', default=20, type=int, help='Number of epochs')
    parser.add_argument(
        '--batch_size', default=16, type=int, help='Batch size')
    
    # related to dataset setting
    parser.add_argument(
        '--dataset_dir', default='dataset', type=str, help='dataset directory')
    parser.add_argument(
        '--dataset', default='CelebA', type=str, help='dataset name')
    parser.add_argument(
        '--image_size', default=128, type=int, help='image size')
    parser.add_argument(
        '--num_workers', default=2, type=int, help='dataloader num_workers')

    # related to model setting
    parser.add_argument(
        '--n_latent', default=10, type=int, help='Dimension of the representation z')
    parser.add_argument(
        '--beta', default=4, type=float, help='beta parameter for KL-term in beta-VAE')
    parser.add_argument(
        '--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument(
        '--beta1', default=0.9, type=float, help='Adam optimizer beta1')
    parser.add_argument(
        '--beta2', default=0.999, type=float, help='Adam optimizer beta2')

    # related to ckpt setting
    parser.add_argument(
        '--ckpt_dir', default='checkpoints', type=str, help='checkpoint directory')
    parser.add_argument(
        '--ckpt_name', default=None, type=str, help='ckpt file name')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
