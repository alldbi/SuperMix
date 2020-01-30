from __future__ import print_function

import torch
import numpy as np
import datetime, os
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt


class AugDataset(Dataset):
    def __init__(self, root_dir, size=500000, transforms=None):
        for i in range(4):

            dict = np.load(root_dir + str(i + 1), allow_pickle=True)
            data = dict['data']
            if i == 0:
                x = data
            else:
                x = np.concatenate((x, data), 0)

        x = x.reshape(-1, 3, 32, 32)

        x = x[0:size, ...]

        self.data = x
        self.transforms = transforms

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # img = self.transforms(self.data[idx, ...])
        # print(img)
        # exit()
        img = self.data[idx, ...]
        img = img.transpose(1, 2, 0)
        img = Image.fromarray(img)
        img = self.transforms(img)
        sample = {'image': img, 'target': 0}

        return sample


# augset = AugDataset('/media/aldb/DATA1/DATABASE/imagenet32x32/Imagenet32_train/train_data_batch_')
#
# print(len(augset))
# exit()


def adjust_learning_rate_new(epoch, optimizer, LUT):
    """
    new learning rate schedule according to RotNet
    """
    lr = next((lr for (max_epoch, lr) in LUT if max_epoch > epoch), LUT[-1][1])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class Logger:
    def __init__(self, dir, var_names=None, format=None, args=None):
        self.dir = dir
        self.var_names = var_names
        self.format = format
        self.vars = []

        # create the log folder
        if not os.path.exists(dir):
            os.makedirs(dir)

        file = open(dir + '/log.txt', 'w')
        file.write('Log file created on ' + str(datetime.datetime.now()) + '\n\n')

        dict = {}
        for arg in vars(args):
            dict[arg] = str(getattr(args, arg))

        for d in sorted(dict.keys()):
            file.write(d + ' : ' + dict[d] + '\n')
        file.write('\n')
        file.close()

    def store(self, vars, log=False):
        self.vars = self.vars + vars
        if log:
            self.log()

    def log(self):

        vars = self.vars
        file = open(self.dir + '/log.txt', 'a')
        st = ''
        for i in range(len(vars)):
            st += self.var_names[i] + ': ' + self.format[i] % (vars[i]) + ', '
        st += 'time: ' + str(datetime.datetime.now()) + '\n'
        file.write(st)
        file.close()
        self.vars = []


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def normalize01(x):
    x = (x - x.min()) / (x.max() - x.min())
    return x


def plot_tensor(tensor_list):
    for i, t in enumerate(tensor_list):
        t_np = t.detach().cpu().numpy().squeeze()
        if len(t_np.shape) == 3:
            t_np = t_np.transpose(1, 2, 0)
        t_np = normalize01(t_np)
        plt.subplot(1, len(tensor_list), i + 1)
        plt.imshow(t_np)
    plt.show()


if __name__ == '__main__':
    pass
