import os
import sys
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from torchsummary import summary
from tensorboardX import SummaryWriter

from model import BetaVAE
from utils import reconstruction_loss, kl_divergence


def main(args):
    net = BetaVAE()

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    # related to train
    parser.add_argument(
        '--n_epochs', type=int, help='Number of epochs', default=20)
    parser.add_argument(
        '--batch_size', type=int, help='Number of batch size', default=16)
    
    # related to model 
    parser.add_argument(
        '--ckpt_dir', type=str, help='Directory to save checkpoint', default='ckpt')
    parser.add_argument(
        '--ckpt_name', type=str, help='Path to previous checkpoint file', default=None)
    
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))