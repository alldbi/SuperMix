from __future__ import print_function, division

import sys
import time
import torch
from helper.util import plot_tensor
from .util import AverageMeter, accuracy
import os
import numpy as np


def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt, warmup_scheduler):
    device = opt.device
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):

        if epoch < 5 + 1:
            warmup_scheduler.step()

        data_time.update(time.time() - end)

        input = input.float()
        input = input.to(device)
        target = target.to(device)

        # ===================forward=====================
        output = model(input)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


def convert_time(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return [hour, minutes, seconds]


def train_distill(epoch, train_loader, val_loader, module_list, criterion_list, optimizer, opt, best_acc, logger,
                  device, warmup_scheduler, total_t):
    t_0 = time.time()

    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    if opt.distill == 'abound':
        module_list[1].eval()
    elif opt.distill == 'factor':
        module_list[2].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    xentm = AverageMeter()
    kdm = AverageMeter()
    otherm = AverageMeter()

    end = time.time()

    t_data = time.time()

    ag_time = 0
    for idx, data_combined in enumerate(train_loader):

        ag_time += time.time() - t_data

        if epoch < opt.epochs_warmup + 1:
            warmup_scheduler.step()

        model_s.train()
        model_t.eval()

        if opt.aug_type is None:
            data = data_combined
        else:
            data = data_combined[0]
            data_aug = data_combined[1]

        if opt.distill in ['crd']:
            input, target, index, contrast_idx = data
        else:
            input, target, index = data
            if opt.aug_type is not None and opt.aug_type != 'cutmix':
                input_aug = data_aug[0]

        input = input.float()
        input = input.to(device)
        target = target.to(device)
        index = index.to(device)
        bs = input.size(0)

        if opt.distill in ['crd']:
            contrast_idx = contrast_idx.to(device)

        if opt.aug_type is not None:
            # construct augmentation samples using mixup or cropmix
            if opt.aug_type == 'mixup':
                input_aug = input_aug.to(device)
                # shift samples in the batch to make pairs
                idx_aug = torch.arange(bs)
                idx_aug[0:bs - 1] = idx_aug[1:bs].clone()
                idx_aug[-1] = 0
                input_aug_b = input_aug[idx_aug]
                if opt.aug_lambda > 0:
                    # compute mixup samples using fixed lambda
                    input_aug = opt.aug_lambda * input_aug + (1 - opt.aug_lambda) * input_aug_b
                elif opt.aug_lambda == -1:
                    # compute mixup samples using the beta distribution
                    lambda_aug = np.random.beta(opt.aug_alpha, opt.aug_alpha, size=[bs, 1, 1, 1])
                    lambda_aug = torch.from_numpy(lambda_aug).type(torch.FloatTensor).to(opt.device)
                    input_aug = lambda_aug * input_aug + (1 - lambda_aug) * input_aug_b
            elif opt.aug_type == 'cutmix':
                input_aug = data_aug[0]
                mask = data_aug[2].view(bs, 1, 32, 32)
                input_aug, mask = input_aug.to(device), mask.to(device)
                # shift samples in the batch to make pairs
                idx_aug = torch.arange(bs)
                idx_aug[0:bs - 1] = idx_aug[1:bs].clone()
                idx_aug[-1] = 0
                input_aug_b = input_aug[idx_aug]

                input_aug = mask * input_aug + (1 - mask) * input_aug_b
                # for i in range(10):
                #     plot_tensor([input_aug[i], mask[i]])
            input_aug = input_aug.to(device)

        # ===================forward=====================
        preact = False
        if opt.distill in ['abound']:
            preact = True
        feat_s, logit_s = model_s(input, is_feat=True, preact=preact)

        # make training faster when there is no need to the prediction of the teacher for nat samples
        if not (opt.distill in ['kd'] and opt.alpha == 0):
            feat_t, logit_t = model_t(input, is_feat=True, preact=preact)
            feat_t = [f.detach() for f in feat_t]

        # compute the predicted label of the teacher for the augmented samples
        if opt.aug_type is not None:
            logit_aug_t = model_t(input_aug)
            logit_aug_s = model_s(input_aug)
            pred_lbl_t = logit_aug_t.argmax(1)

        # cls + kl div
        loss_cls_nat = criterion_cls(logit_s, target)

        loss_cls_aug = 0
        if opt.aug_type is not None:
            loss_cls_aug = criterion_cls(logit_aug_s, pred_lbl_t)

        loss_cls = loss_cls_nat + loss_cls_aug

        if opt.alpha > 0:
            # if opt.aug_type is not None:
            #     loss_div = criterion_div(logit_aug_s, logit_aug_t)
            # else:
            loss_div = criterion_div(logit_s, logit_t)
        else:
            loss_div = torch.zeros([1])
            loss_div = loss_div.to(device)

        # other kd beyond KL divergence
        if opt.distill == 'kd':
            loss_kd = 0
        elif opt.distill == 'hint':
            f_s = module_list[1](feat_s[opt.hint_layer])
            f_t = feat_t[opt.hint_layer]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'crd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
        elif opt.distill == 'attention':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'nst':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'similarity':
            g_s = [feat_s[-2]]
            g_t = [feat_t[-2]]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'rkd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'pkt':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'kdsvd':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'correlation':
            f_s = module_list[1](feat_s[-1])
            f_t = module_list[2](feat_t[-1])
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'vid':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
            loss_kd = sum(loss_group)
        elif opt.distill == 'abound':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'fsp':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'factor':
            factor_s = module_list[1](feat_s[-2])
            factor_t = module_list[2](feat_t[-2], is_factor=True)
            loss_kd = criterion_kd(factor_s, factor_t)
        else:
            raise NotImplementedError(opt.distill)

        loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd

        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        losses.update(loss.item(), bs)
        top1.update(acc1.item(), bs)
        top5.update(acc5.item(), bs)
        xentm.update(loss_cls.item(), bs)
        kdm.update(loss_div.item())
        otherm.update(loss_kd)

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        total_t += time.time() - end
        batch_time.update(time.time() - end, 1)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0 and idx > 0:
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
            # compute the remaining time
            epoch_remaining = opt.epochs - epoch
            # total_iters_remaining = len(train_loader) * (opt.epochs - epoch + 1) - idx
            iters_passed = len(train_loader) * (epoch - 1) + idx
            iters_remaining = len(train_loader) * (opt.epochs - epoch + 1) - idx

            ert = total_t * iters_remaining / iters_passed
            ert = convert_time(ert)

            print(
                'Epoch: %d [%03d, %03d], l_xent: %.4f, l_kd: %.4f, l_other: %.4f, acc: %.2f, lr: %.4f, time: %.1f, ert: %d:%02d:%02d' % (
                    epoch, idx, len(train_loader), xentm.avg, kdm.avg, otherm.avg, top1.avg, lr,
                    batch_time.avg * opt.print_freq, ert[0], ert[1], ert[2]))

        if idx % opt.test_freq == 0 and idx > 0:
            test_acc, tect_acc_top5, test_loss = validate(val_loader, model_s, criterion_cls, opt)
            model_s.train()
            if test_acc > best_acc:
                best_acc = test_acc
                state = {
                    'epoch': epoch,
                    'model': model_s.state_dict(),
                    'best_acc': best_acc,
                }
                save_file = os.path.join(opt.save_folder, '{}.pth'.format(opt.model_s))
                torch.save(state, save_file)
            print("\nTest acc: %.2f, best: %.2f\n" % (test_acc, best_acc))

            logger.store([epoch, xentm.avg, kdm.avg, otherm.avg, top1.avg, test_acc, best_acc, lr], log=True)

            xentm.reset()
            kdm.reset()
            top1.reset()
            otherm.reset()
            batch_time.reset()
            end = time.time()

        t_data = time.time()

    return best_acc, total_t


def validate(val_loader, model, criterion, opt):
    device = opt.device
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):
            input = input.float()
            input = input.to(device)
            target = target.to(device)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    model.train()
    return top1.avg, top5.avg, losses.avg
