import os
import sys
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
import copy
import time
import matplotlib.pyplot as plt
import scipy.misc as misc
from helper.util import get_teacher_name
from models import model_dict
import math
import torchvision.datasets as datasets
import torchvision.models as models


class Datasubset(torch.utils.data.Dataset):
    def __init__(self, dataset, len):
        self.dataset = dataset
        self.len = len

    def __getitem__(self, i):
        return self.dataset[i % self.len]

    def __len__(self):
        return self.len


def load_teacher(model_path, n_cls):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path)['model'])
    print('==> done')
    return model


def onehot(y, n_classes=100):
    bs = y.size(0)
    y = y.type(torch.LongTensor).view(-1, 1)
    y_onehot = torch.FloatTensor(bs, n_classes)

    # In your for loop
    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)
    return y_onehot.cuda()


class Smoothing(nn.Module):
    def __init__(self):
        super(Smoothing, self).__init__()

    def compute_kernels(self, sigma=1, chennels=1):
        size_denom = 5.
        sigma = int(sigma * size_denom)
        kernel_size = sigma
        mgrid = torch.arange(kernel_size, dtype=torch.float32)
        mean = (kernel_size - 1.) / 2.
        mgrid = mgrid - mean
        mgrid = mgrid * size_denom
        kernel = 1. / (sigma * math.sqrt(2. * math.pi)) * \
                 torch.exp(-(((mgrid - 0.) / (sigma)) ** 2) * 0.5)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernelx = kernel.view(1, 1, int(kernel_size), 1).repeat(chennels, 1, 1, 1)
        kernely = kernel.view(1, 1, 1, int(kernel_size)).repeat(chennels, 1, 1, 1)

        return kernelx.cuda(), kernely.cuda(), kernel_size

    def forward(self, input, sigma):
        if sigma > 0:
            channels = input.size(1)
            kx, ky, kernel_size = self.compute_kernels(sigma=sigma, chennels=channels)

            # padd the input
            padd0 = int(kernel_size // 2)
            evenorodd = int(1 - kernel_size % 2)
            # self.pad = torch.nn.ConstantPad2d((padd0 - evenorodd, padd0, padd0 - evenorodd, padd0), 0.)

            input = F.pad(input, (padd0 - evenorodd, padd0, padd0 - evenorodd, padd0), 'constant', 0.)
            input = F.conv2d(input, weight=kx, groups=channels)
            input = F.conv2d(input, weight=ky, groups=channels)
        return input


smoother = Smoothing().cuda()


def normalize01(x):
    return (x - x.min()) / (x.max() - x.min())


def tensor2img(t, ismask=False):
    x = t.cpu().detach().numpy().squeeze()
    if len(x.shape) == 3:
        x = x.transpose(1, 2, 0)
    if ismask:
        return x
    return normalize01(x)


def plott(t_list):
    for ti in range(len(t_list)):
        x = tensor2img(t_list[ti])
        plt.subplot(1, len(t_list), ti + 1)
        plt.imshow(x)
    plt.show()


def kldiv(x, y):
    x = F.log_softmax(x, 1)
    y = F.softmax(y, 1)
    return nn.KLDivLoss(reduction='none')(x, y).sum(1)


def kldiv2(x, y):
    x = F.log_softmax(x, 1)
    return nn.KLDivLoss(reduction='none')(x, y).sum(1)


def mask_process(x, upsample_size):
    bs = x.size(0)
    K = x.size(1)
    mask_w = x.size(3)
    m1 = x.view(bs * K, 1, mask_w, mask_w)
    m1 = F.interpolate(m1, upsample_size, mode='bilinear')
    m1 = m1.view(bs, K, 1, upsample_size, upsample_size)
    m1 = torch.sigmoid(m1)
    sum_masks = m1.sum(1, keepdim=True)
    m1 = m1 / sum_masks
    return m1


def mix_batch(net, data, K, alpha=1, mask_w=16, sigma_grad=2, max_iter=200, toler=0):



    # size of the current batch
    bs = data.size(0)
    # spatial size of the input images
    inw = data.size(2)

    # predict the label of the input images
    f_data = net(data)
    pred_lbl = f_data.argmax(1)

    # generate the shuffle indexes to construct the sets X
    idx = list(range(bs))
    idx_arr = [idx]
    for i in range(K - 1):
        idx = idx_arr[-1].copy()
        idx[:-1] = idx_arr[-1][1:]
        idx[-1] = idx_arr[-1][0]
        idx_arr.append(idx)
    idx_arr = np.array(idx_arr)

    # construct K set and store them in data_X
    data_X = torch.zeros([bs, K, 3, inw, inw])
    lbl_X = torch.zeros([bs, K])
    for i in range(K):
        data_X[:, i, ...] = data[idx_arr[i], ...]
        lbl_X[:, i] = pred_lbl[idx_arr[i], ...]
    data_X = data_X.cuda()

    # construct the target soft labels, Equation 2 in the paper
    soft_targets = torch.zeros([bs, opt.n_classes])
    for i in range(bs):
        lbl_set = lbl_X[i:i + 1, :]
        lbl_set = lbl_set.view(K, 1)
        lambda_aug = np.random.dirichlet(np.ones(K) * alpha, 1).reshape(K, 1)
        lambda_aug = torch.from_numpy(lambda_aug).type(torch.FloatTensor).cuda()
        lbl_set_onehot = onehot(lbl_set, opt.n_classes) * lambda_aug
        lbl_soft = lbl_set_onehot.sum(0)
        soft_targets[i, :] = lbl_soft
    soft_targets = soft_targets.cuda()

    # construct the mask variables
    mask_init = 0.
    mask = torch.ones([bs, K, 1, mask_w, mask_w]).cuda() * mask_init

    loop_i = 0

    _, top2lbl = torch.topk(soft_targets, K, 1)
    top2lbl, _ = top2lbl.sort()

    batch_mask = torch.ones([bs]).cuda()

    while batch_mask.sum().item() > toler and loop_i < max_iter:
        # define the variable of the mask
        m = Variable(mask, requires_grad=True)

        # process the mask variable which will: 1) upsample the mask, 2) normalize it
        m_pr = mask_process(m, upsample_size=inw)

        # construct mixed images
        mixed_data = m_pr * data_X
        mixed_data = mixed_data.sum(1)

        # compute the prediction on mixed images
        f_mix = net.forward(mixed_data)

        stdloss = torch.abs(m_pr * (m_pr - 1))
        stdloss = stdloss.mean(1).mean(1).mean(1).mean(1)

        # compute the kldiv between the predictions and the target soft labels
        kl = kldiv2(f_mix, soft_targets)

        # zero out the loss for successfully mixed samples
        kl = (kl + stdloss * opt.lambda_s) * batch_mask

        loss = kl.sum()

        # compute the gradients of the loss w.r.t. to the mask variable
        grad = torch.autograd.grad(loss, m)[0]

        w_k = copy.deepcopy(grad.data)  # bs x K x 1 x mask_w x mask_w

        w_k = w_k.view(bs * K, 1, mask_w, mask_w)
        w = smoother(w_k, sigma=sigma_grad)
        w = w.view(bs, K, 1, mask_w, mask_w)

        f_k = -1 * kl

        dot = w_k.view(bs, -1) @ w.view(bs, -1).t()
        dot = torch.diag(dot)

        pert = torch.abs(f_k) / (dot + 1e-10)

        pert = torch.clamp(pert, 0.0001, 2000)

        r_i = -1 * pert.view(bs, 1, 1, 1, 1).repeat(1, K, 1, 1, 1) * w

        mask = mask + r_i.detach() * batch_mask.view(bs, 1, 1, 1, 1)
        mask_pr = mask_process(mask, upsample_size=inw)
        check_mix = mask_pr * data_X
        check_mix = check_mix.sum(1)

        pred_mix = net.forward(check_mix)

        _, pred_lbl_top2 = torch.topk(pred_mix, K, 1)
        pred_lbl_top2, _ = pred_lbl_top2.sort()

        batch_mask = pred_lbl_top2 != top2lbl

        batch_mask = batch_mask.sum(1).type(torch.FloatTensor).cuda()
        batch_mask = (batch_mask > 0).type(torch.FloatTensor).cuda()
        loop_i += 1

    idx = np.where(batch_mask.detach().cpu().numpy() == 0)[0].reshape(-1)

    check_mix = check_mix[idx, ...]
    mask_pr = mask_pr[idx, ...]
    pred_mix = pred_mix[idx, ...]
    data_X = data_X[idx, ...]

    return check_mix, mask_pr, pred_mix, data_X, loop_i


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def plott(t_list):
    for ti in range(len(t_list)):
        x = tensor2img(t_list[ti])
        plt.subplot(1, len(t_list), ti + 1)
        plt.imshow(x)
    plt.show()


def convert_time(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return [hour, minutes, seconds]


def augment(opt, data_loader):
    model.eval()
    counter = 0
    total_iter = 0
    batch_counter = 0
    total_time = 0

    while counter < opt.aug_size:
        for batch_index, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            bs = images.size(0)

            model.zero_grad()

            t0 = time.time()

            if bs != opt.bs:
                break

            # use the data in the batch to generated new data
            images_mixed, mask, pred_mix, data_X, iter = mix_batch(model, images, alpha=opt.alpha, K=opt.k,
                                                                   mask_w=opt.w,
                                                                   sigma_grad=opt.sigma,
                                                                   toler=opt.tol, max_iter=opt.max_iter)

            delta_t = time.time() - t0
            total_time += delta_t

            # number of generated images
            n_suc = images_mixed.size(0)

            # plot the results
            if opt.plot and n_suc>0:
                n_samples = min(n_suc, 4)

                for p in range(n_samples):
                    n_cols = opt.k * 2 + 1

                    # plot mixed images
                    plt.subplot(n_samples, n_cols, p * n_cols + 1)
                    plt.imshow(tensor2img(images_mixed[p, ...]))
                    plt.axis('off')
                    plt.title('Mixed')

                    # plot input images
                    for ps in range(opt.k):
                        plt.subplot(n_samples, n_cols, p * n_cols + 1 + ps + 1)
                        plt.imshow(tensor2img(data_X[p, ps, ...]))
                        plt.axis('off')
                        plt.title('input ' + str(ps))

                    # plot input images
                    for ps in range(opt.k):
                        plt.subplot(n_samples, n_cols, p * n_cols + 1 + ps + opt.k + 1)
                        plt.imshow(tensor2img(mask[p, ps, ...], ismask=True), cmap='jet')
                        plt.axis('off')
                        plt.title('mask ' + str(ps))

                plt.show()

            for i in range(n_suc):
                img = images_mixed[i].detach().cpu().numpy().transpose(1, 2, 0)
                img = img * std + mean
                img = img * 255

                img = img.astype(np.uint8)

                misc.imsave(opt.save_dir + '/' + str(counter + i) + '.png', img)

            counter += n_suc

            total_iter += iter
            batch_counter += 1

            remaining_time = (opt.aug_size - counter) * total_time / (counter+1)
            ert = convert_time(remaining_time)

            print(
                "iter: %d, n_generated: %d, iters: %02d, ert: %d:%d:%02d" % (
                    batch_index, counter, iter, ert[0], ert[1],
                    ert[2]))
            if counter > opt.aug_size:
                return 0


def eval(device, net):
    net.eval()
    test_loss = 0.0  # cost function error
    correct = 0.0
    criterion = nn.CrossEntropyLoss()
    for (images, labels) in cifar100_test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = net(images)

        loss = criterion(outputs, labels)
        test_loss += loss.item() * images.size()[0]
        preds = outputs.argmax(1)
        correct += preds.eq(labels).sum()

    acc = correct.float() / len(cifar100_test_loader.dataset)
    loss = test_loss / len(cifar100_test_loader.dataset)

    return acc, loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar100', help='dataset to augment', choices=['imagenet', 'cifar100'])
    parser.add_argument('--model', type=str, default='resnet32',
                        help='name of the supervisor model to load')
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda or cpu')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='output directory to save results')
    parser.add_argument('--input_dir', type=str, default='/home/aldb/outputs/imgenet/imgnet_train1',
                        help='directory of the training set of ImageNet')
    parser.add_argument('--bs', type=int, default=16, help='batch size')
    parser.add_argument('--aug_size', type=int, default=500000, help='number of images to generate')
    parser.add_argument('--k', type=int, default=2, help='number of samples to mix')
    parser.add_argument('--max_iter', type=int, default=50, help='maximum number of iteration for each batch')
    parser.add_argument('--alpha', type=float, default=3, help='alpha of the Dirichlet distribution')
    parser.add_argument('--sigma', type=float, default=2, help='standard deviation for the Gaussian blurring')
    parser.add_argument('--w', type=int, default=16, help='width of the mixing masks')
    parser.add_argument('--lambda_s', type=float, default=25, help='multiplier of the sparsity loss')
    parser.add_argument('--tol', type=int, default=30,
                        help='tolerance (percent) for the number of unsuccessful samples in the batch')
    parser.add_argument('--plot', type=bool, default=True, help='plot the results')
    opt = parser.parse_args()

    # set the device
    device = torch.device(opt.device)

    opt.tol = int(opt.bs * opt.tol / 100)

    if opt.dataset == 'cifar100':

        # mean and std of the training set of cifar100
        CIFAR100_MEAN = (0.5070, 0.4865, 0.4409)
        CIFAR100_STD = (0.2673, 0.2564, 0.2761)
        std = np.array(CIFAR100_STD)
        mean = np.array(CIFAR100_MEAN)
        std = std.reshape(1, 1, 3)
        mean = mean.reshape(1, 1, 3)

        # load the data
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
        ])

        cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
                                                          transform=transform)

        data_loader = DataLoader(cifar100_training, shuffle=True, num_workers=2, batch_size=opt.bs)

        cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True,
                                                      transform=transform_test)
        cifar100_test_loader = DataLoader(
            cifar100_test, shuffle=False, num_workers=2, batch_size=100)

        # load the teacher model
        path_t = './save/models/' + opt.model + '_vanilla/ckpt_epoch_240.pth'
        model = load_teacher(path_t, 100)
        model.eval()
        model.to(device)
        opt.n_classes = 100


    elif opt.dataset == 'imagenet':
        # mean and std of the training set of ImageNet
        mean_imgnet = (0.485, 0.456, 0.406)
        std_imgnet = (0.229, 0.224, 0.225)
        std = np.array(std_imgnet)
        mean = np.array(mean_imgnet)
        std = std.reshape(1, 1, 3)
        mean = mean.reshape(1, 1, 3)

        train_dataset = datasets.ImageFolder(
            opt.input_dir,
            transforms.Compose([
                transforms.Scale(260),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean_imgnet,
                                     std=std_imgnet),
            ]))

        data_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=opt.bs, num_workers=4, pin_memory=True, shuffle=True)

        loader = getattr(models, opt.model)

        model = loader(pretrained=True)
        model.eval()
        model.to(device)
        opt.n_classes = 1000

    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)

    augment(opt, data_loader)
