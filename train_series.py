from train_student import main

import os
import argparse
from helper.util import get_teacher_name


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=50, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--device', type=str, default='cuda:2', help='batch_size')
    parser.add_argument('--num_workers', type=int, default=2, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=600, help='number of training epochs')
    parser.add_argument('--init_epochs', type=int, default=30, help='init training for two-stage methods')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='200, 300, 400, 500',  # '150, 250, 350, 450',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset')

    # model
    parser.add_argument('--model_s', type=str, default='wrn_16_2',
                        choices=['resnet8', 'resnet14', '0', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
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
    parser.add_argument('--aug', type=str, default='/home/lab320/dataset/cifar_augmented_kl^3/out_imgnet',
                        help='address of the augmented dataset')
    parser.add_argument('--aug_size', type=str, default=500000,
                        help='size of the augmented dataset, -1 means the maximum possible size')

    parser.add_argument('--trial', type=str, default='withwarmpup', help='trial id')

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
    parser.add_argument('--seed', default=505, type=int, help='random seed')

    opt = parser.parse_args()

    # set different learning rate from these 4 models
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.02

    opt.model_path = './save/student_model'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_t = get_teacher_name(opt.path_t)



    return opt


if __name__ == '__main__':
    aug_size_list = [50000, 100000, 200000, 300000, 400000]

    for a in aug_size_list:
        opt = parse_option()
        opt.aug_size = a
        opt.model_name = 'S:{}_T:{}_{}_{}_{}/r:{}_a:{}_b:{}_{}_{}_{}_{}'.format(opt.model_s, opt.model_t, opt.dataset,
                                                                                opt.distill, opt.aug[-7:],
                                                                                opt.gamma, opt.alpha, opt.beta,
                                                                                opt.trial,
                                                                                opt.device, opt.seed, opt.aug_size)
        opt.save_folder = os.path.join(opt.model_path, opt.model_name)
        if not os.path.isdir(opt.save_folder):
            os.makedirs(opt.save_folder)




        main(opt)
