"""
the general training framework
"""

from __future__ import print_function

import os
import argparse
import time

# import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models import model_dict
from models.util import Embed, ConvReg, LinearEmbed
from models.util import Connector, Translator, Paraphraser

from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample

from helper.util import adjust_learning_rate, Logger, count_parameters, get_teacher_name, WarmUpLR, plot_tensor

from distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss
from distiller_zoo import PKT, ABLoss, FactorTransfer, KDSVD, FSP, NSTLoss
from crd.criterion import CRDLoss

from helper.loops import train_distill as train, validate
from helper.pretrain import init
import numpy as np


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=5, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--device', type=str, default='cuda:1', help='batch_size')
    parser.add_argument('--num_workers', type=int, default=2, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=600, help='number of training epochs')
    parser.add_argument('--init_epochs', type=int, default=30, help='init training for two-stage methods')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate')
    parser.add_argument('--epochs_warmup', type=int, default=5, help='number of epochs for learning rate warm up')
    parser.add_argument('--lr_decay_epochs', type=str, default='200, 300, 400, 500',  # '150, 250, 350, 450',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset')

    # model
    parser.add_argument('--model_s', type=str, default='resnet20',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2'])
    parser.add_argument('--path_t', type=str, default='./save/models/resnet110_vanilla/ckpt_epoch_240.pth',
                        help='teacher model snapshot')

    # distillation
    parser.add_argument('--distill', type=str, default='kd', choices=['kd', 'hint', 'attention', 'similarity',
                                                                      'correlation', 'vid', 'crd', 'kdsvd', 'fsp',
                                                                      'rkd', 'pkt', 'abound', 'factor', 'nst'])

    # parser.add_argument('--aug', type=str, default=None,
    #                     help='address of the augmented dataset')

    # augmentation parameters
    parser.add_argument('--aug_type', type=str, default='supermix', choices=[None, 'mixup', 'cutmix', 'supermix'],
                        help='type of augmentation')
    parser.add_argument('--aug_dir', type=str, default='/home/aldb/outputs/new2/wrn_40_2_k:3_alpha:3',
                        help='address of the augmented dataset')
    parser.add_argument('--aug_size', type=str, default=-1,
                        help='size of the augmented dataset, -1 means the maximum possible size')
    parser.add_argument('--aug_lambda', type=float, default=0.5, help='lambda for mixup, must be between 0 and 1')
    parser.add_argument('--aug_alpha', type=float, default=10000,
                        help='alpha for the beta distribution to sample the lambda, this is active when --aug_lambda is -1')

    parser.add_argument('--trial', type=str, default='augmented', help='trial id')

    parser.add_argument('-r', '--gamma', type=float, default=2, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=0, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=0, help='weight balance for other losses')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')

    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')

    # hint layer
    parser.add_argument('--hint_layer', default=2, type=int, choices=[0, 1, 2, 3, 4])

    parser.add_argument('--test_interval', type=int, default=None, help='test interval')
    parser.add_argument('--seed', default=1001, type=int, help='random seed')

    opt = parser.parse_args()

    return opt


def load_teacher(model_path, n_cls):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path)['model'])
    print('==> done')
    return model


def build_grid(source_size, target_size):
    k = float(target_size) / float(source_size)
    direct = torch.linspace(0, k, target_size).unsqueeze(0).repeat(target_size, 1).unsqueeze(-1)
    full = torch.cat([direct, direct.transpose(1, 0)], dim=2).unsqueeze(0)
    return full.cuda()


def random_crop_grid(x, grid):
    delta = x.size(2) - grid.size(1)
    grid = grid.repeat(x.size(0), 1, 1, 1).cuda()
    # Add random shifts by x
    grid[:, :, :, 0] = grid[:, :, :, 0] + torch.FloatTensor(x.size(0)).cuda().random_(0, delta).unsqueeze(-1).unsqueeze(
        -1).expand(-1, grid.size(1), grid.size(2)) / x.size(2)
    # Add random shifts by y
    grid[:, :, :, 1] = grid[:, :, :, 1] + torch.FloatTensor(x.size(0)).cuda().random_(0, delta).unsqueeze(-1).unsqueeze(
        -1).expand(-1, grid.size(1), grid.size(2)) / x.size(2)
    return grid


############

############


if __name__ == '__main__':
    opt = parse_option()
    distill(opt)
