import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import numpy as np

fig, ax = plt.subplots(1, 4)

# set size and resolution of the figure
fig.set_figheight(2.6)
fig.set_figwidth(12)
fig.set_dpi(120)
plt.tight_layout()
fig.subplots_adjust(top=0.9, bottom=0.14, left=0.044)

#############################################################################
############ WRN ############################################################
#############################################################################

# set ax properties
ax[0].set_xticks([0, 2, 4, 6, 8, 10])
ax[0].set_xticklabels(['0', '2', '4', '6', '8', '10'])
ax[0].grid(axis='y')
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)
ax[0].xaxis.set_label_coords(0.5, -0.1)

# wrn with supermix
n = np.array([0, 1, 2, 4, 6, 8, 10])
wrn403_wrn162_sueprmix = np.array([73.25, 74.61, 75.81, 75.91, 76.21, 76.30, 76.30])
err_supermix = np.array([0, 0.15, 0.14, 0.21, 0.14, 0.15, 0.14])

# wrn with mixup
wrn403_wrn162_mixup = np.array([73.26, 73.85, 75.18, 75.78, 75.84, 75.60, 75.70])
err_mixup = np.array([0, 0.13, 0.25, 0.14, 0.26, 0.22, 0.11])

# wrn with imgnet
wrn403_wrn162_imgnet = np.array([[73.26, 73.78, 74.40, 74.49, 74.79, 75.05, 75.37],
                                 [73.26, 73.78, 74.07, 74.88, 75.20, 74.95, 74.89],
                                 [73.26, 73.61, 74.93, 74.73, 74.80, 74.58, 74.63]])
err_imgnet = wrn403_wrn162_imgnet.std(0)
wrn403_wrn162_imgnet = wrn403_wrn162_imgnet.mean(0)

# plot values
ax[0].plot(n, wrn403_wrn162_sueprmix, '-o', color='g', label='SuperMix', markersize=3)
ax[0].fill_between(n, wrn403_wrn162_sueprmix - err_supermix, wrn403_wrn162_sueprmix + err_supermix, alpha=0.3,
                   facecolor='g')

ax[0].plot(n, wrn403_wrn162_mixup, '-o', color='r', label='MixUp', markersize=3)
ax[0].fill_between(n, wrn403_wrn162_mixup - err_mixup, wrn403_wrn162_mixup + err_mixup, alpha=0.3,
                   facecolor='r')

ax[0].plot(n, wrn403_wrn162_imgnet, '-o', color='b', label='ImageNet', markersize=3)
ax[0].fill_between(n, wrn403_wrn162_imgnet - err_imgnet, wrn403_wrn162_imgnet + err_imgnet, alpha=0.3,
                   facecolor='b')

# set ax labels
ax[0].set_ylabel('Test acc.', fontsize=14)
ax[0].set_xlabel(r'$\kappa$', fontsize=14)
ax[0].set_title(r'WRN-40-2$\rightarrow$WRN-16-2')
ax[0].legend()

#############################################################################
############ Resnet ############################################################
#############################################################################

# set ax properties
ax[1].set_xticks([0, 2, 4, 6, 8, 10])
ax[1].set_xticklabels(['0', '2', '4', '6', '8', '10'])
ax[1].grid(axis='y')
ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)
ax[1].xaxis.set_label_coords(0.5, -0.1)

# wrn with supermix
n = np.array([0, 1, 2, 4, 6, 8, 10])
res110_res20_sueprmix = np.array([[69.06, 71.05, 72.29, 72.26, 72.60, 71.96, 72.39],
                                  [69.06, 71.19, 72.58, 72.66, 72.34, 72.28, 72.39],
                                  [69.06, 70.96, 71.80, 72.11, 72.13, 72.40, 72.39]])
err_supermix = res110_res20_sueprmix.std(0)
res110_res20_sueprmix = res110_res20_sueprmix.mean(0)

res110_res20_mixup = np.array([[69.06, 70.54, 71.46, 71.72, 72.20, 71.69, 72.26],
                               [69.06, 70.46, 71.87, 72.52, 71.89, 71.47, 72.26],
                               [69.06, 70.82, 71.58, 71.79, 72.40, 72.37, 72.26]])
err_mixup = res110_res20_mixup.std(0)
res110_res20_mixup = res110_res20_mixup.mean(0)

res110_res20_imgnet = np.array([[69.06, 70.18, 70.98, 70.79, 71.22, 71.77, 71.57],
                                [69.06, 70.18, 70.86, 71.36, 71.52, 71.20, 71.69],
                                [69.06, 69.81, 71.02, 71.25, 71.40, 71.24, 71.82]])
err_imgnet = res110_res20_imgnet.std(0)
res110_res20_imgnet = res110_res20_imgnet.mean(0)

# plot values
ax[1].plot(n, res110_res20_sueprmix, '-o', color='g', label='SuperMix', markersize=3)
ax[1].fill_between(n, res110_res20_sueprmix - err_supermix, res110_res20_sueprmix + err_supermix, alpha=0.3,
                   facecolor='g')

ax[1].plot(n, res110_res20_mixup, '-o', color='r', label='MixUp', markersize=3)
ax[1].fill_between(n, res110_res20_mixup - err_mixup, res110_res20_mixup + err_mixup, alpha=0.3,
                   facecolor='r')

ax[1].plot(n, res110_res20_imgnet, '-o', color='b', label='ImageNet', markersize=3)
ax[1].fill_between(n, res110_res20_imgnet - err_imgnet, res110_res20_imgnet + err_imgnet, alpha=0.3,
                   facecolor='b')

# set ax labels
# ax[1].set_ylabel('Test acc.', fontsize=14)
ax[1].set_xlabel(r'$\kappa$', fontsize=14)
ax[1].set_title(r'ResNet110$\rightarrow$ResNet20')

#############################################################################
############ Resnetxx4 ######################################################
#############################################################################

# set ax properties
ax[2].set_xticks([0, 2, 4, 6, 8, 10])
ax[2].set_xticklabels(['0', '2', '4', '6', '8', '10'])
ax[2].grid(axis='y')
ax[2].spines['right'].set_visible(False)
ax[2].spines['top'].set_visible(False)
ax[2].xaxis.set_label_coords(0.5, -0.1)

n = np.array([0, 1, 2, 4, 6, 8, 10])
res32x4_res8x4_sueprmix = np.array([72.50, 74.76, 75.82, 76.41, 76.65, 76.85, 76.92])
err_supermix = np.array([0, 0.23, 0.25, 0.15, 0.12, 0.33, 0.08])

res32x4_res8x4_mixup = np.array([[72.50, 74.33, 74.82, 76.39, 76.46, 76.34, 76.17],
                                 [72.50, 74.54, 75.74, 76.12, 76.60, 76.43, 76.17],
                                 [72.50, 74.52, 75.13, 75.82, 76.85, 76.55, 76.17]])

err_mixup = res32x4_res8x4_mixup.std(0)
res32x4_res8x4_mixup = res32x4_res8x4_mixup.mean(0)

res32x4_res8x4_imgnet = np.array([[72.50, 74.36, 75.32, 75.31, 75.13, 75.63, 75.50],
                                  [72.50, 74.55, 74.66, 74.82, 76.19, 75.70, 75.50],
                                  [72.50, 74.29, 74.91, 75.70, 75.10, 76.11, 75.50]])
err_imgnet = res32x4_res8x4_imgnet.std(0)
res32x4_res8x4_imgnet = res32x4_res8x4_imgnet.mean(0)

# plot values
ax[2].plot(n, res32x4_res8x4_sueprmix, '-o', color='g', label='SuperMix', markersize=3)
ax[2].fill_between(n, res32x4_res8x4_sueprmix - err_supermix, res32x4_res8x4_sueprmix + err_supermix, alpha=0.3,
                   facecolor='g')

ax[2].plot(n, res32x4_res8x4_mixup, '-o', color='r', label='MixUp', markersize=3)
ax[2].fill_between(n, res32x4_res8x4_mixup - err_mixup, res32x4_res8x4_mixup + err_mixup, alpha=0.3,
                   facecolor='r')

ax[2].plot(n, res32x4_res8x4_imgnet, '-o', color='b', label='ImageNet', markersize=3)
ax[2].fill_between(n, res32x4_res8x4_imgnet - err_imgnet, res32x4_res8x4_imgnet + err_imgnet, alpha=0.3,
                   facecolor='b')

# set ax labels
# ax[2].set_ylabel('Test acc.', fontsize=14)
ax[2].set_xlabel(r'$\kappa$', fontsize=14)
ax[2].set_title(r'ResNet32x4$\rightarrow$ResNet8x4')

#############################################################################
############ VGG13/VGG8  ####################################################
#############################################################################

# set ax properties
ax[3].set_xticks([0, 2, 4, 6, 8, 10])
ax[3].set_xticklabels(['0', '2', '4', '6', '8', '10'])
ax[3].grid(axis='y')
ax[3].spines['right'].set_visible(False)
ax[3].spines['top'].set_visible(False)
ax[3].xaxis.set_label_coords(0.5, -0.1)

n = np.array([0, 1, 2, 4, 6, 8, 10])
vgg13_vgg8_sueprmix = np.array([70.36, 71.74, 73.18, 74.47, 74.57, 74.68, 74.59])
err_supermix = np.array([0, 0.2, 0.1, 0.43, 0.18, 0.24, 0.12])

vgg13_vgg8_mixup = np.array([[70.36, 71.72, 72.73, 73.39, 73.89, 73.95, 74.07],
                             [70.36, 71.87, 72.33, 73.60, 73.55, 73.91, 74.07]])

err_mixup = vgg13_vgg8_mixup.std(0)
vgg13_vgg8_mixup = vgg13_vgg8_mixup.mean(0)

vgg13_vgg8_imgnet = np.array([[70.36, 71.40, 72.49, 73.49, 74.08, 74.17, 73.97],
                              [70.36, 71.18, 72.86, 73.20, 73.74, 73.67, 73.89]])
err_imgnet = vgg13_vgg8_imgnet.std(0)
vgg13_vgg8_imgnet = vgg13_vgg8_imgnet.mean(0)

# plot values
ax[3].plot(n, vgg13_vgg8_sueprmix, '-o', color='g', label='SuperMix', markersize=3)
ax[3].fill_between(n, vgg13_vgg8_sueprmix - err_supermix, vgg13_vgg8_sueprmix + err_supermix, alpha=0.3,
                   facecolor='g')

ax[3].plot(n, vgg13_vgg8_mixup, '-o', color='r', label='MixUp', markersize=3)
ax[3].fill_between(n, vgg13_vgg8_mixup - err_mixup, vgg13_vgg8_mixup + err_mixup, alpha=0.3,
                   facecolor='r')

ax[3].plot(n, vgg13_vgg8_imgnet, '-o', color='b', label='ImageNet', markersize=3)
ax[3].fill_between(n, vgg13_vgg8_imgnet - err_imgnet, vgg13_vgg8_imgnet + err_imgnet, alpha=0.3,
                   facecolor='b')

# set ax labels
# ax[3].set_ylabel('Test acc.', fontsize=14)
ax[3].set_xlabel(r'$\kappa$', fontsize=14)
ax[3].set_title(r'ResNet32x4$\rightarrow$ResNet8x4')

plt.show()
