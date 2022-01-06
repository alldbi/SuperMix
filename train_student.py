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


def distill(opt):
    # refine the opt arguments

    opt.model_path = './save/student_model'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_t = get_teacher_name(opt.path_t)

    opt.print_freq = int(50000 / opt.batch_size / opt.print_freq)

    opt.model_name = 'S:{}_T:{}_{}_{}/r:{}_a:{}_b:{}_{}_{}_{}_{}_lam:{}_alp:{}_augsize:{}_T:{}'.format(
        opt.model_s, opt.model_t,
        opt.dataset,
        opt.distill,
        opt.gamma, opt.alpha, opt.beta,
        opt.trial,
        opt.device, opt.seed,
        opt.aug_type,
        opt.aug_lambda,
        opt.aug_alpha,
        opt.aug_size, opt.kd_T)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    opt.learning_rate = 0.1 * opt.batch_size / 128

    # set different learning rate from these 4 models
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = opt.learning_rate / 5

    print("learning rate is set to:", opt.learning_rate)

    best_acc = 0
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    # dataloader
    if opt.dataset == 'cifar100':
        if opt.distill in ['crd']:
            train_loader, val_loader, n_data = get_cifar100_dataloaders_sample(batch_size=opt.batch_size,
                                                                               num_workers=opt.num_workers,
                                                                               k=opt.nce_k,
                                                                               mode=opt.mode)
        else:
            train_loader, val_loader, n_data = get_cifar100_dataloaders(opt, is_instance=True)
        n_cls = 100
    else:
        raise NotImplementedError(opt.dataset)

    # set the interval for testing
    opt.test_freq = int(50000 / opt.batch_size)

    # compute number of epochs using the original cifar100 dataset size
    opt.lr_decay_epochs = list(int(i * 50000 / opt.aug_size) for i in opt.lr_decay_epochs)
    opt.epochs = int(opt.epochs * 50000 / opt.aug_size)

    print('Decay epochs: ', opt.lr_decay_epochs)
    print('Max epochs: ', opt.epochs)

    # set the device
    if torch.cuda.is_available():
        device = torch.device(opt.device)
    else:
        device = torch.device('cpu')

    # model
    model_t = load_teacher(opt.path_t, n_cls)
    model_s = model_dict[opt.model_s](num_classes=n_cls)

    # print(model_s)

    print("Size of the teacher:", count_parameters(model_t))
    print("Size of the student:", count_parameters(model_s))

    data = torch.randn(2, 3, 32, 32)
    model_t.eval()
    model_s.eval()
    feat_t, _ = model_t(data, is_feat=True)
    feat_s, _ = model_s(data, is_feat=True)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)
    if opt.distill == 'kd':
        criterion_kd = DistillKL(opt.kd_T)
    elif opt.distill == 'hint':
        criterion_kd = HintLoss()
        regress_s = ConvReg(feat_s[opt.hint_layer].shape, feat_t[opt.hint_layer].shape)
        module_list.append(regress_s)
        trainable_list.append(regress_s)
    elif opt.distill == 'crd':
        opt.s_dim = feat_s[-1].shape[1]
        opt.t_dim = feat_t[-1].shape[1]
        opt.n_data = n_data
        criterion_kd = CRDLoss(opt)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
    elif opt.distill == 'attention':
        criterion_kd = Attention()
    elif opt.distill == 'nst':
        criterion_kd = NSTLoss()
    elif opt.distill == 'similarity':
        criterion_kd = Similarity()
    elif opt.distill == 'rkd':
        criterion_kd = RKDLoss()
    elif opt.distill == 'pkt':
        criterion_kd = PKT()
    elif opt.distill == 'kdsvd':
        criterion_kd = KDSVD()
    elif opt.distill == 'correlation':
        criterion_kd = Correlation()
        embed_s = LinearEmbed(feat_s[-1].shape[1], opt.feat_dim)
        embed_t = LinearEmbed(feat_t[-1].shape[1], opt.feat_dim)
        module_list.append(embed_s)
        module_list.append(embed_t)
        trainable_list.append(embed_s)
        trainable_list.append(embed_t)
    elif opt.distill == 'vid':
        s_n = [f.shape[1] for f in feat_s[1:-1]]
        t_n = [f.shape[1] for f in feat_t[1:-1]]
        criterion_kd = nn.ModuleList(
            [VIDLoss(s, t, t) for s, t in zip(s_n, t_n)]
        )
        # add this as some parameters in VIDLoss need to be updated
        trainable_list.append(criterion_kd)
    elif opt.distill == 'abound':
        s_shapes = [f.shape for f in feat_s[1:-1]]
        t_shapes = [f.shape for f in feat_t[1:-1]]
        connector = Connector(s_shapes, t_shapes)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(connector)
        init_trainable_list.append(model_s.get_feat_modules())
        criterion_kd = ABLoss(len(feat_s[1:-1]))
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, opt)
        # classification
        module_list.append(connector)
    elif opt.distill == 'factor':
        s_shape = feat_s[-2].shape
        t_shape = feat_t[-2].shape
        paraphraser = Paraphraser(t_shape)
        translator = Translator(s_shape, t_shape)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(paraphraser)
        criterion_init = nn.MSELoss()
        init(model_s, model_t, init_trainable_list, criterion_init, train_loader, opt)
        # classification
        criterion_kd = FactorTransfer()
        module_list.append(translator)
        module_list.append(paraphraser)
        trainable_list.append(translator)
    elif opt.distill == 'fsp':
        s_shapes = [s.shape for s in feat_s[:-1]]
        t_shapes = [t.shape for t in feat_t[:-1]]
        criterion_kd = FSP(s_shapes, t_shapes)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(model_s.get_feat_modules())
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, opt)
        # classification training
        pass
    else:
        raise NotImplementedError(opt.distill)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # classification loss
    criterion_list.append(criterion_div)  # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)  # other knowledge distillation loss

    # optimizer
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.to(device)
        criterion_list.to(device)
        cudnn.benchmark = True

    # setup warmup
    warmup_scheduler = WarmUpLR(optimizer, len(train_loader) * opt.epochs_warmup)

    # validate teacher accuracy
    teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt)
    print('teacher accuracy: %.2f \n' % (teacher_acc))

    # creat logger
    logger = Logger(dir=opt.save_folder,
                    var_names=['Epoch', 'l_xent', 'l_kd', 'l_other', 'acc_train', 'acc_test', 'acc_test_best', 'lr'],
                    format=['%02d', '%.4f', '%.4f', '%.4f', '%.2f', '%.2f', '%.2f', '%.6f'], args=opt)

    total_t = 0
    # routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(epoch, opt, optimizer)
        time1 = time.time()
        best_acc, total_t = train(epoch, train_loader, val_loader, module_list, criterion_list, optimizer, opt,
                                  best_acc, logger,
                                  device, warmup_scheduler, total_t)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        print('Best accuracy: %.2f \n' % (best_acc))

def distill(opt):
    # refine the opt arguments

    opt.model_path = './save/student_model'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_t = get_teacher_name(opt.path_t)

    opt.print_freq = int(50000 / opt.batch_size / opt.print_freq)

    opt.model_name = 'S:{}_T:{}_{}_{}/r:{}_a:{}_b:{}_{}_{}_{}_{}_lam:{}_alp:{}_augsize:{}_T:{}'.format(
        opt.model_s, opt.model_t,
        opt.dataset,
        opt.distill,
        opt.gamma, opt.alpha, opt.beta,
        opt.trial,
        opt.device, opt.seed,
        opt.aug_type,
        opt.aug_lambda,
        opt.aug_alpha,
        opt.aug_size, opt.kd_T)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    opt.learning_rate = 0.1 * opt.batch_size / 128

    # set different learning rate from these 4 models
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = opt.learning_rate / 5

    print("learning rate is set to:", opt.learning_rate)

    best_acc = 0
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    # dataloader
    if opt.dataset == 'cifar100':
        if opt.distill in ['crd']:
            train_loader, val_loader, n_data = get_cifar100_dataloaders_sample(batch_size=opt.batch_size,
                                                                               num_workers=opt.num_workers,
                                                                               k=opt.nce_k,
                                                                               mode=opt.mode)
        else:
            train_loader, val_loader, n_data = get_cifar100_dataloaders(opt, is_instance=True)
        n_cls = 100
    else:
        raise NotImplementedError(opt.dataset)

    # set the interval for testing
    opt.test_freq = int(50000 / opt.batch_size)

    # compute number of epochs using the original cifar100 dataset size
    opt.lr_decay_epochs = list(int(i * 50000 / opt.aug_size) for i in opt.lr_decay_epochs)
    opt.epochs = int(opt.epochs * 50000 / opt.aug_size)

    print('Decay epochs: ', opt.lr_decay_epochs)
    print('Max epochs: ', opt.epochs)

    # set the device
    if torch.cuda.is_available():
        device = torch.device(opt.device)
    else:
        device = torch.device('cpu')

    # model
    model_t = load_teacher(opt.path_t, n_cls)
    model_s = model_dict[opt.model_s](num_classes=n_cls)

    # print(model_s)

    print("Size of the teacher:", count_parameters(model_t))
    print("Size of the student:", count_parameters(model_s))

    data = torch.randn(2, 3, 32, 32)
    model_t.eval()
    model_s.eval()
    feat_t, _ = model_t(data, is_feat=True)
    feat_s, _ = model_s(data, is_feat=True)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)
    if opt.distill == 'kd':
        criterion_kd = DistillKL(opt.kd_T)
    elif opt.distill == 'hint':
        criterion_kd = HintLoss()
        regress_s = ConvReg(feat_s[opt.hint_layer].shape, feat_t[opt.hint_layer].shape)
        module_list.append(regress_s)
        trainable_list.append(regress_s)
    elif opt.distill == 'crd':
        opt.s_dim = feat_s[-1].shape[1]
        opt.t_dim = feat_t[-1].shape[1]
        opt.n_data = n_data
        criterion_kd = CRDLoss(opt)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
    elif opt.distill == 'attention':
        criterion_kd = Attention()
    elif opt.distill == 'nst':
        criterion_kd = NSTLoss()
    elif opt.distill == 'similarity':
        criterion_kd = Similarity()
    elif opt.distill == 'rkd':
        criterion_kd = RKDLoss()
    elif opt.distill == 'pkt':
        criterion_kd = PKT()
    elif opt.distill == 'kdsvd':
        criterion_kd = KDSVD()
    elif opt.distill == 'correlation':
        criterion_kd = Correlation()
        embed_s = LinearEmbed(feat_s[-1].shape[1], opt.feat_dim)
        embed_t = LinearEmbed(feat_t[-1].shape[1], opt.feat_dim)
        module_list.append(embed_s)
        module_list.append(embed_t)
        trainable_list.append(embed_s)
        trainable_list.append(embed_t)
    elif opt.distill == 'vid':
        s_n = [f.shape[1] for f in feat_s[1:-1]]
        t_n = [f.shape[1] for f in feat_t[1:-1]]
        criterion_kd = nn.ModuleList(
            [VIDLoss(s, t, t) for s, t in zip(s_n, t_n)]
        )
        # add this as some parameters in VIDLoss need to be updated
        trainable_list.append(criterion_kd)
    elif opt.distill == 'abound':
        s_shapes = [f.shape for f in feat_s[1:-1]]
        t_shapes = [f.shape for f in feat_t[1:-1]]
        connector = Connector(s_shapes, t_shapes)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(connector)
        init_trainable_list.append(model_s.get_feat_modules())
        criterion_kd = ABLoss(len(feat_s[1:-1]))
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, opt)
        # classification
        module_list.append(connector)
    elif opt.distill == 'factor':
        s_shape = feat_s[-2].shape
        t_shape = feat_t[-2].shape
        paraphraser = Paraphraser(t_shape)
        translator = Translator(s_shape, t_shape)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(paraphraser)
        criterion_init = nn.MSELoss()
        init(model_s, model_t, init_trainable_list, criterion_init, train_loader, opt)
        # classification
        criterion_kd = FactorTransfer()
        module_list.append(translator)
        module_list.append(paraphraser)
        trainable_list.append(translator)
    elif opt.distill == 'fsp':
        s_shapes = [s.shape for s in feat_s[:-1]]
        t_shapes = [t.shape for t in feat_t[:-1]]
        criterion_kd = FSP(s_shapes, t_shapes)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(model_s.get_feat_modules())
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, opt)
        # classification training
        pass
    else:
        raise NotImplementedError(opt.distill)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # classification loss
    criterion_list.append(criterion_div)  # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)  # other knowledge distillation loss

    # optimizer
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.to(device)
        criterion_list.to(device)
        cudnn.benchmark = True

    # setup warmup
    warmup_scheduler = WarmUpLR(optimizer, len(train_loader) * opt.epochs_warmup)

    # validate teacher accuracy
    teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt)
    print('teacher accuracy: %.2f \n' % (teacher_acc))

    # creat logger
    logger = Logger(dir=opt.save_folder,
                    var_names=['Epoch', 'l_xent', 'l_kd', 'l_other', 'acc_train', 'acc_test', 'acc_test_best', 'lr'],
                    format=['%02d', '%.4f', '%.4f', '%.4f', '%.2f', '%.2f', '%.2f', '%.6f'], args=opt)

    total_t = 0
    # routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(epoch, opt, optimizer)
        time1 = time.time()
        best_acc, total_t = train(epoch, train_loader, val_loader, module_list, criterion_list, optimizer, opt,
                                  best_acc, logger,
                                  device, warmup_scheduler, total_t)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        print('Best accuracy: %.2f \n' % (best_acc))


if __name__ == '__main__':
    opt = parse_option()
    distill(opt)
