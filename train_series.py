from train_student import distill

import os
import argparse
from helper.util import get_teacher_name


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=5, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--device', type=str, default='cuda:2', help='batch_size')
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
    parser.add_argument('--model_s', type=str, default='wrn_16_2',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2'])
    parser.add_argument('--path_t', type=str, default='./save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth',
                        help='teacher model snapshot')

    # distillation
    parser.add_argument('--distill', type=str, default='kd', choices=['kd', 'hint', 'attention', 'similarity',
                                                                      'correlation', 'vid', 'crd', 'kdsvd', 'fsp',
                                                                      'rkd', 'pkt', 'abound', 'factor', 'nst'])

    # parser.add_argument('--aug', type=str, default=None,
    #                     help='address of the augmented dataset')

    # augmentation parameters
    parser.add_argument('--aug_type', type=str, default='cutmix', choices=[None, 'mixup', 'cropmix', 'supermix'],
                        help='type of augmentation')
    parser.add_argument('--aug_dir', type=str, default='/home/aldb2/aug_dataset/',
                        help='address of the augmented dataset')
    parser.add_argument('--aug_size', type=str, default=-1,
                        help='size of the augmented dataset, -1 means the maximum possible size')
    parser.add_argument('--aug_lambda', type=float, default=-1, help='lambda for mixup, must be between 0 and 1')
    parser.add_argument('--aug_alpha', type=float, default=10,
                        help='alpha for the beta distribution to sample the lambda, this is active when --aug_lambda is -1')
    parser.add_argument('--aug_k', type=float, default=2,
                        help='number of samples to mix')



    parser.add_argument('--trial', type=str, default='06Feb2020', help='trial id')

    parser.add_argument('-r', '--gamma', type=float, default=0.2, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=1.8, help='weight balance for KD')
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
    parser.add_argument('--seed', default=102, type=int, help='random seed')

    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    aug_size_list = [50000, 100000, 200000, 300000, 400000]
    # aug_lambda = [0.4, 0.3, 0.2, 0.1]
    aug_alpha = [0.1, 0.5, 1, 3, 5, 15, 10000]
    # aug_alpha.reverse()

    # gamma = [0.1, 0.3, 0.5, 0.7, 0.9]

    student_list = [8, 9, 10, 11, 12]

    k_list = [3]
    k_list.reverse()
    for k in k_list:
        opt = parse_option()
        # opt.aug_size = a
        opt.aug_alpha = 3
        opt.aug_lambda = -1
        opt.gamma = 2
        opt.alpha = 0
        opt.aug_type = 'supermix'
        opt.trial = "07Feb20"
        s = 0
        opt.aug_k = k


        if s==0:
            opt.model_s = 'wrn_16_2'
            opt.path_t = './save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth'
        elif s==1:
            opt.model_s = 'wrn_40_1'
            opt.path_t = './save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth'
        elif s==2:
            opt.model_s = 'resnet20'
            opt.path_t = './save/models/resnet56_vanilla/ckpt_epoch_240.pth'
        elif s==3:
            opt.model_s = 'resnet20'
            opt.path_t = './save/models/resnet110_vanilla/ckpt_epoch_240.pth'
        elif s==4:
            opt.model_s = 'resnet32'
            opt.path_t = './save/models/resnet110_vanilla/ckpt_epoch_240.pth'
        elif s==5:
            opt.model_s = 'resnet8x4'
            opt.path_t = './save/models/resnet32x4_vanilla/ckpt_epoch_240.pth'
        elif s==6:
            opt.model_s = 'vgg8'
            opt.path_t = './save/models/vgg13_vanilla/ckpt_epoch_240.pth'

        #######################################################
        elif s==7:
            opt.model_s = 'MobileNetV2'
            opt.path_t = './save/models/vgg13_vanilla/ckpt_epoch_240.pth'
        elif s==8:
            opt.model_s = 'MobileNetV2'
            opt.path_t = './save/models/ResNet50_vanilla/ckpt_epoch_240.pth'
        elif s==9:
            opt.model_s = 'vgg8'
            opt.path_t = './save/models/ResNet50_vanilla/ckpt_epoch_240.pth'
        elif s==10:
            opt.model_s = 'ShuffleV1'
            opt.path_t = './save/models/resnet32x4_vanilla/ckpt_epoch_240.pth'
        elif s==11:
            opt.model_s = 'ShuffleV2'
            opt.path_t = './save/models/resnet32x4_vanilla/ckpt_epoch_240.pth'
        elif s==12:
            opt.model_s = 'ShuffleV1'
            opt.path_t = './save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth'

        # train the model
        distill(opt)
