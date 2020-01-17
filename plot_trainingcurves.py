import numpy as np
import matplotlib.pyplot as plt

from numpy import *
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def set_size(w, h, ax=None):
    """ w, h: width, height in inches """

    if not ax: ax = plt.gca()

    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w) / (r - l)
    figh = float(h) / (t - b)

    print(figw, figh)
    ax.figure.set_size_inches(int(figw), int(figh))


# below are all percentage
# left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
# ax1 = fig.add_axes([left, bottom, width, height])  # main axes
# ax1.plot(x, y, 'r')
# ax1.set_xlabel('x')
# ax1.set_ylabel('y')
# ax1.set_title('title')
#
# ax2 = fig.add_axes([0.2, 0.6, 0.25, 0.25])  # inside axes
# ax2.plot(y, x, 'b')
# ax2.set_xlabel('x')
# ax2.set_ylabel('y')
# ax2.set_title('title inside 1')
#
# # different method to add axes
# ####################################
# plt.axes([0.6, 0.2, 0.25, 0.25])
# plt.plot(y[::-1], x, 'g')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('title inside 2')
#
# plt.show()


def get_val(filename_list, val='acc_test:'):
    for fi in range(len(filename_list)):

        with open(filename_list[fi]) as f:
            content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        content = [x.strip() for x in content]

        # get first line that has l_xent
        for i in range(len(content)):
            if content[i].find('l_xent') > 0:
                break

        content = content[i:]

        out = [0]

        for c in content:
            idx_0 = c.find(val)
            s = c[idx_0 + len(val):idx_0 + len(val) + 6]
            if s.find(',') > 0:
                s = s[0:-1]



            out.append(float(s))


        if fi == 0:
            out_arr = np.array(out)
        else:
            out_arr += np.array(out)

    return out_arr / len(filename_list)


f1 = '/home/lab320/github/KDA/save/student_model/S:resnet20_T:resnet110_cifar100_kd_snet110/r:2_a:0_b:0_extended_cuda:0_1/log.txt'
f2 = '/home/lab320/github/KDA/save/student_model/S:resnet20_T:resnet110_cifar100_kd_snet110/r:2_a:0_b:0_extended_cuda:1_2/log.txt'
f3 = '/home/lab320/github/KDA/save/student_model/S:resnet20_T:resnet110_cifar100_kd_snet110/r:2_a:0_b:0_extended_cuda:2_200/log.txt'
acc_test_supermix = get_val([f1, f2, f3])
acc_train_supermix = get_val([f1, f2, f3], 'acc_train:')

f1 = '/home/lab320/github/KDA/save/student_model/S:resnet20_T:resnet110_cifar100_kd_out_avg/r:2_a:0_b:0_extended_cuda:0_22/log.txt'
f2 = '/home/lab320/github/KDA/save/student_model/S:resnet20_T:resnet110_cifar100_kd_out_avg/r:2_a:0_b:0_extended_cuda:1_222/log.txt'
f3 = '/home/lab320/github/KDA/save/student_model/S:resnet20_T:resnet110_cifar100_kd_out_avg/r:2_a:0_b:0_extended_cuda:2_2222/log.txt'
acc_test_mixup = get_val([f1, f2, f3])
acc_train_mixup = get_val([f1, f2, f3], 'acc_train:')

f1 = '/home/lab320/github/KDA/txt_files/resnet110-resnet20-imgnet_0.txt'
f2 = '/home/lab320/github/KDA/txt_files/resnet110-resnet20-imgnet_1.txt'
f3 = '/home/lab320/github/KDA/txt_files/resnet110-resnet20-imgnet_2.txt'
acc_test_imgnet = get_val([f1, f2, f3])
acc_train_imgnet = get_val([f1, f2, f3], 'acc_train:')

f1 = '/home/lab320/github/KDA/txt_files/resnet20.txt'
acc_test_orig = get_val([f1])
acc_train_orig = get_val([f1], 'acc_train:')


fig, ax = plt.subplots(figsize=(8, 6), dpi=180)

# set_size(10, 10, ax)
ax.plot(100 - acc_test_orig, '-k', label='orig')
ax.plot(100 - acc_train_orig, '--k')
ax.plot(100 - acc_test_imgnet, '-r', label='ImageNet32x32')
ax.plot(100 - acc_train_imgnet, '--r')
ax.plot(100 - acc_test_mixup, '-b', label='MixUp')
ax.plot(100 - acc_train_mixup, '--b')
ax.plot(100 - acc_test_supermix, '-g', label='SuperMix')
ax.plot(100 - acc_train_supermix, '--g')


ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
          ncol=4, mode="expand", borderaxespad=0., prop={'size': 14})

# fig.subplots_adjust(top=0.2)
plt.xlim((-10, 610))
# plt.ylim((35, 100))
# zoom in

# small rec
rec1_x = 401
rec1_y = 27
rec1_w = 199
rec1_h = 3

# big rec
rec2_x = 220
rec2_y = 45
rec2_w = 380
rec2_h = 50

ax1 = plt.axes([0.4, 0.42, 0.5, 0.4])

# plot lines
ax.plot([rec1_x, rec2_x + 1], [rec1_y + rec1_h, rec2_y], '-k', linewidth=1)
ax.plot([rec1_x + rec1_w, rec2_x + rec2_w], [rec1_y + rec1_h, rec2_y], '-k', linewidth=1)


for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(14)

for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(14)

ax.set_xlabel('Epoch', fontsize=14)
ax.set_ylabel('Accuracy', fontsize=14)

epoch = 401
ax1.plot(100 - acc_test_imgnet[epoch:], '-r', label='ImageNet32x32')
ax1.plot(100 - acc_test_mixup[epoch:], '-b', label='MixUp')
ax1.plot(100 - acc_test_supermix[epoch:], '-g', label='SuperMix')
ax1.axis('off')


# plot rectangle

def get_ax_size(ax):
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    width *= fig.dpi
    height *= fig.dpi
    return width, height


w, h = get_ax_size(ax)
print(w, h)

rect1 = patches.Rectangle((rec1_x, rec1_y), rec1_w, rec1_h, linewidth=1, edgecolor=[0, 0, 0], facecolor='none')
rect2 = patches.Rectangle((rec2_x, rec2_y), rec2_w, rec2_h, linewidth=1, edgecolor=[0, 0, 0], facecolor='none')
# Add the patch to the Axes
ax.add_patch(rect1)
ax.add_patch(rect2)

plt.show()
