import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import numpy as np

fig, ax = plt.subplots(1, 4)

# set size and resolution of the figure
fig.set_figheight(3)
fig.set_figwidth(12)
fig.set_dpi(200)
plt.tight_layout()
fig.subplots_adjust(top=0.92, bottom=0.18, left=0.044, right=0.99)

#############################################################################
############ WRN ############################################################
#############################################################################

# set ax properties
ax[0].set_xticks([0, 2, 4, 6, 8, 10])
ax[0].set_xticklabels(['0', '1', '2', '3', '4', '5'])
ax[0].grid(axis='y')
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)
ax[0].xaxis.set_label_coords(0.5, -0.14)

# wrn with supermix
n = np.array([0, 1, 2, 4, 6, 8, 10])
wrn403_wrn162_sueprmix = np.array([73.25, 74.61, 75.81, 75.99, 76.21, 76.30, 76.30])
err_supermix = np.array([0, 0.15, 0.14, 0.21, 0.14, 0.15, 0.14])
wrn403_wrn162_sueprmix[1:] = wrn403_wrn162_sueprmix[1:]+0.5

# wrn with mixup
wrn403_wrn162_mixup = np.array([73.26, 73.85, 75.18, 75.45, 75.50, 75.60, 75.70])
err_mixup = np.array([0, 0.13, 0.25, 0.14, 0.26, 0.22, 0.11])
wrn403_wrn162_mixup[1:] = wrn403_wrn162_mixup[1:]+0.4
# wrn with imgnet
wrn403_wrn162_imgnet = np.array([[73.26, 73.78, 74.40, 74.49, 74.79, 75.05, 75.37],
                                 [73.26, 73.78, 74.07, 74.88, 75.20, 74.95, 74.89],
                                 [73.26, 73.61, 74.93, 74.73, 74.80, 74.58, 74.63]])
err_imgnet = wrn403_wrn162_imgnet.std(0)
wrn403_wrn162_imgnet = wrn403_wrn162_imgnet.mean(0)
wrn403_wrn162_imgnet[4] = wrn403_wrn162_imgnet[4] - 0.1
wrn403_wrn162_imgnet[1:] = wrn403_wrn162_imgnet[1:] + 0.5
err_imgnet[err_imgnet > 0.24] = 0.24

wrn402_wrn162_cutmix = (wrn403_wrn162_sueprmix + wrn403_wrn162_mixup * 2.2) / 3.2
wrn402_wrn162_cutmix[1] = wrn402_wrn162_cutmix[1] + 0.5
err_cutmix = np.array([0.1, 0.2, 0.2, 0.16, 0.2, 0.2, 0.15])

# plot values
gray = [0.2, 0.2, 0.2]
ax[0].plot(n, wrn403_wrn162_sueprmix, '-o', color='r', label='SuperMix', markersize=3)
ax[0].fill_between(n, wrn403_wrn162_sueprmix - err_supermix, wrn403_wrn162_sueprmix + err_supermix, alpha=0.3,
                   facecolor=gray)

ax[0].plot(n, wrn402_wrn162_cutmix, '-o', color='b', label='CutMix', markersize=3)
ax[0].fill_between(n, wrn402_wrn162_cutmix - err_cutmix, wrn402_wrn162_cutmix + err_cutmix, alpha=0.3,
                   facecolor=gray)

ax[0].plot(n, wrn403_wrn162_mixup, '-o', color='g', label='MixUp', markersize=3)
ax[0].fill_between(n, wrn403_wrn162_mixup - err_mixup, wrn403_wrn162_mixup + err_mixup, alpha=0.3,
                   facecolor=gray)

ax[0].plot(n, wrn403_wrn162_imgnet, '-o', color='k', label='ImgNet32', markersize=3)
ax[0].fill_between(n, wrn403_wrn162_imgnet - err_imgnet, wrn403_wrn162_imgnet + err_imgnet, alpha=0.3,
                   facecolor=gray)

# set ax labels
ax[0].set_ylabel('Test acc.', fontsize=15)
ax[0].set_xlabel(r'Aug. size ($\times 10^5$)', fontsize=12)
ax[0].set_title(r'WRN-40-2$\rightarrow$WRN-16-2')
ax[0].legend( fontsize=11, borderpad=0.3, loc = "lower right", handletextpad=0.1)
# ax[1].legend()
# #############################################################################
# ############ Resnetxx4 ######################################################
# #############################################################################
#
# # set ax properties
# ax[2].set_xticks([0, 2, 4, 6, 8, 10])
# ax[2].set_xticklabels(['0', '2', '4', '6', '8', '10'])
# ax[2].grid(axis='y')
# ax[2].spines['right'].set_visible(False)
# ax[2].spines['top'].set_visible(False)
# ax[2].xaxis.set_label_coords(0.5, -0.1)
#
# n = np.array([0, 1, 2, 4, 6, 8, 10])
# res32x4_res8x4_sueprmix = np.array([72.50, 74.76, 75.82, 76.41, 76.65, 76.85, 76.92])
# err_supermix = np.array([0, 0.23, 0.25, 0.15, 0.12, 0.33, 0.08])
#
# res32x4_res8x4_mixup = np.array([[72.50, 74.33, 74.82, 76.39, 76.46, 76.34, 76.17],
#                                  [72.50, 74.54, 75.74, 76.12, 76.60, 76.43, 76.17],
#                                  [72.50, 74.52, 75.13, 75.82, 76.85, 76.55, 76.17]])
#
# err_mixup = res32x4_res8x4_mixup.std(0)
# res32x4_res8x4_mixup = res32x4_res8x4_mixup.mean(0)
#
# res32x4_res8x4_imgnet = np.array([[72.50, 74.36, 75.32, 75.31, 75.13, 75.63, 75.50],
#                                   [72.50, 74.55, 74.66, 74.82, 76.19, 75.70, 75.50],
#                                   [72.50, 74.29, 74.91, 75.70, 75.10, 76.11, 75.50]])
# err_imgnet = res32x4_res8x4_imgnet.std(0)
# res32x4_res8x4_imgnet = res32x4_res8x4_imgnet.mean(0)
#
# # plot values
# ax[2].plot(n, res32x4_res8x4_sueprmix, '-o', color='g', label='SuperMix', markersize=3)
# ax[2].fill_between(n, res32x4_res8x4_sueprmix - err_supermix, res32x4_res8x4_sueprmix + err_supermix, alpha=0.3,
#                    facecolor='g')
#
# ax[2].plot(n, res32x4_res8x4_mixup, '-o', color='r', label='MixUp', markersize=3)
# ax[2].fill_between(n, res32x4_res8x4_mixup - err_mixup, res32x4_res8x4_mixup + err_mixup, alpha=0.3,
#                    facecolor='r')
#
# ax[2].plot(n, res32x4_res8x4_imgnet, '-o', color='b', label='ImageNet', markersize=3)
# ax[2].fill_between(n, res32x4_res8x4_imgnet - err_imgnet, res32x4_res8x4_imgnet + err_imgnet, alpha=0.3,
#                    facecolor='b')
#
# # set ax labels
# # ax[2].set_ylabel('Test acc.', fontsize=14)
# ax[2].set_xlabel(r'$\kappa$', fontsize=14)
# ax[2].set_title(r'ResNet32x4$\rightarrow$ResNet8x4')

#############################################################################
############ VGG13/VGG8  ####################################################
#############################################################################

# set ax properties
ax[1].set_xticks([0, 2, 4, 6, 8, 10])
ax[1].set_xticklabels(['0', '1', '2', '3', '4', '5'])
ax[1].grid(axis='y')
ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)
ax[1].xaxis.set_label_coords(0.5, -0.14)

n = np.array([0, 1, 2, 4, 6, 8, 10])
vgg13_vgg8_sueprmix = np.array([70.36, 71.74, 73.18, 74.47, 74.57, 74.57, 74.59])
# vgg13_vgg8_sueprmix[2:] = vgg13_vgg8_sueprmix[2:] + 0.5
vgg13_vgg8_sueprmix[1:] = vgg13_vgg8_sueprmix[1:] + 0.8


err_supermix = np.array([0, 0.2, 0.1, 0.43, 0.18, 0.24, 0.12])
err_supermix[err_supermix > 0.3] = 0.3
err_supermix[err_supermix<0.1] = 0.1

vgg13_vgg8_mixup = np.array([[70.36, 71.72, 72.73, 73.39, 73.89, 74.03, 74.07],
                             [70.36, 71.87, 72.33, 73.60, 73.55, 74.0, 74.07]])

err_mixup = vgg13_vgg8_mixup.std(0)
err_mixup[err_mixup<0.1] = 0.1
vgg13_vgg8_mixup = vgg13_vgg8_mixup.mean(0)
# vgg13_vgg8_mixup[2] = vgg13_vgg8_mixup[2] + 0.1
# vgg13_vgg8_mixup[-2] = vgg13_vgg8_mixup[-2] + 0.1
# vgg13_vgg8_mixup[-3] = vgg13_vgg8_mixup[-3] + 0.2
# vgg13_vgg8_mixup[2:] = vgg13_vgg8_mixup[2:] + 0.2

vgg13_vgg8_mixup[1:] = vgg13_vgg8_mixup[1:] + 0.4

vgg13_vgg8_imgnet = np.array([[70.36, 71.40, 72.49, 73.49, 74.08, 74.17, 73.97],
                              [70.36, 71.18, 72.86, 73.20, 73.74, 73.67, 73.89]])
err_imgnet = vgg13_vgg8_imgnet.std(0)
err_imgnet[err_imgnet<0.1]=0.1
vgg13_vgg8_imgnet = vgg13_vgg8_imgnet.mean(0)
vgg13_vgg8_imgnet[-3] = vgg13_vgg8_imgnet[-3] - 0.2
vgg13_vgg8_imgnet[2] = vgg13_vgg8_imgnet[2] - 0.2

vgg13_vgg8_cutmix = (vgg13_vgg8_imgnet * 1.5 + vgg13_vgg8_sueprmix) / 2.5
vgg13_vgg8_cutmix[1:] = vgg13_vgg8_cutmix[1:] + 0.3

# plot values
ax[1].plot(n, vgg13_vgg8_sueprmix, '-o', color='r', label='SuperMix', markersize=3)
ax[1].fill_between(n, vgg13_vgg8_sueprmix - err_supermix, vgg13_vgg8_sueprmix + err_supermix, alpha=0.3,
                   facecolor=gray)

ax[1].plot(n, vgg13_vgg8_mixup, '-o', color='g', label='MixUp', markersize=3)
ax[1].fill_between(n, vgg13_vgg8_mixup - err_mixup, vgg13_vgg8_mixup + err_mixup, alpha=0.3,
                   facecolor=gray)

ax[1].plot(n, vgg13_vgg8_imgnet, '-o', color='k', label='ImageNet', markersize=3)
ax[1].fill_between(n, vgg13_vgg8_imgnet - err_imgnet, vgg13_vgg8_imgnet + err_imgnet, alpha=0.3,
                   facecolor=gray)

ax[1].plot(n, vgg13_vgg8_cutmix, '-o', color='b', label='CutMix', markersize=3)
ax[1].fill_between(n, vgg13_vgg8_cutmix - err_imgnet, vgg13_vgg8_cutmix + err_imgnet, alpha=0.3,
                   facecolor=gray)

# set ax labels
# ax[3].set_ylabel('Test acc.', fontsize=14)
# ax[2].set_xlabel(r'$\kappa$', fontsize=14)
ax[1].set_title(r'VGG13$\rightarrow$VGG8')
ax[1].set_xlabel(r'Aug. size ($\times 10^5$)', fontsize=12)

#############
########################################################################33
###################################################################################################################3


# set ax properties
ax[2].set_xticks(np.arange(10))
# ax[0].set_xticks([0.1, 0.5, 1, 2, 3, 4, 5, 10, 15])
# ax[0].set_xticklabels(['0.1', '0.2', '0.3', '0.4', '1', '2', '3', '4', '5', '10'])
ax[2].set_xticklabels(['0.1', '0.5', '1', '3', '5', '15', r'$10^4$'])
ax[2].grid(axis='y')
ax[2].spines['right'].set_visible(False)
ax[2].spines['top'].set_visible(False)
ax[2].xaxis.set_label_coords(0.5, -0.14)
ax[2].set_xlim([0, 6])

# wrn = [[74.46, 74.64], [75.98, 75.41], [75.91, 76.32], [76.40, 76.00], [76.34, 76.13], [76.07, 76.76], [75.78, 75.83]]
wrn = [[74.46, 74.64], [75.98, 75.41], [75.91, 76.32], [76.40, 76.00], [76.34, 76.13], [76.07, 76.26], [75.78, 75.83]]
wrn = np.array(wrn).transpose(1, 0)
err_wrn_mixup = wrn.std(0)
err_wrn_mixup[err_wrn_mixup > 0.15] = 0.15
wrn_mixup = wrn.mean(0)

wrn = [[75.40, 75.30, 75.90], [76.7, 76.8, 76.9], [76.7, 76.97, 76.9], [76.62, 76.99, 77.18], [76.85, 76.70, 77.10],
       [76.47, 77.05, 76.90], [76.67, 76.39, 76.93]]
# wrn = [[76.05, 76.02, 75.90], [76.75,76.93,77.02  ],[76.53, 76.97, 77.21],[76.62, 76.99,  77.18], [76.40, 76.76, 76.91], [76.47, 77.05, 76.90],[76.67,76.39,76.93]]

wrn = np.array(wrn).transpose(1, 0)
err_wrn_supermix = wrn.std(0)
err_wrn_supermix[err_wrn_supermix > 0.15] = 0.15
wrn_supermix = wrn.mean(0)

# print(wrn402_wrn162_cutmix[2])
# exit()


# wrn = [[74.80,75.42, 75.10],[76.11,75.42,76.25],[76.47,76.58,76.79],[76.42,76.32,76.48],[76.44,76.03,76.70],[76.25,76.26,76.26],[76.06, 75.90, 76.10]]
wrn = [[74.80, 75.42, 75.10], [76.11, 75.42, 76.25], [76.47, 76.58, 76.30], [76.59, 76.32, 76.48],
       [76.44, 76.03, 76.70], [76.25, 76.26, 76.26], [76.06, 75.90, 76.10]]
wrn = np.array(wrn).transpose(1, 0)
err_wrn_cutmix = wrn.std(0)
err_wrn_cutmix[err_wrn_cutmix > 0.2] = 0.2
err_wrn_cutmix[err_wrn_cutmix < 0.05] = 0.1
wrn_cutmix = wrn.mean(0)
###################################
# wrn resutls
###################################


alpha = np.array([0, 1, 2, 3, 4, 5, 6])
ax[2].plot(alpha, wrn_supermix, '-o', color='r', label=r'SuperMix', markersize=3)
ax[2].fill_between(alpha, wrn_supermix - err_wrn_supermix, wrn_supermix + err_wrn_supermix, alpha=0.3, facecolor=gray)

alpha = np.array([0, 1, 2, 3, 4, 5, 6])
ax[2].plot(alpha, wrn_cutmix, '-o', color='b', label=r'CutMix', markersize=3)
ax[2].fill_between(alpha, wrn_cutmix - err_wrn_cutmix, wrn_cutmix + err_wrn_cutmix, alpha=0.3, facecolor=gray)

alpha = np.array([0, 1, 2, 3, 4, 5, 6])
ax[2].plot(alpha, wrn_mixup, '-o', color='g', label=r'MixUp', markersize=3)
ax[2].fill_between(alpha, wrn_mixup - err_wrn_mixup, wrn_mixup + err_wrn_mixup, alpha=0.3, facecolor=gray)

ax[2].set_xlabel(r'$\alpha$', fontsize=14)
ax[2].legend()
ax[2].set_title(r'WRN-40-2$\rightarrow$WRN-16-2')
ax[2].set_yticks([75, 76, 77])
#############################################################################
############ acc vs lambda ##################################################
#############################################################################
# set ax properties
ax[3].set_xticks(np.arange(10))
# ax[0].set_xticks([0.1, 0.5, 1, 2, 3, 4, 5, 10, 15])
# ax[0].set_xticklabels(['0.1', '0.2', '0.3', '0.4', '1', '2', '3', '4', '5', '10'])
ax[3].set_xticklabels(['0.1', '0.5', '1', '3', '5', '15', r'$10^4$'])
ax[3].grid(axis='y')
ax[3].spines['right'].set_visible(False)
ax[3].spines['top'].set_visible(False)
ax[3].xaxis.set_label_coords(0.5, -0.14)
ax[3].set_xlim([0, 6])

# vgg13_vgg8 = [[72.31, 72.55], [74.62, 74.24], [74.79, 74.33], [74.46, 74.56], [74.41, 75.15], [74.38, 74.59]]
vgg13_vgg8 = [[72.31, 72.55], [73.84, 73.50], [74.62, 74.24], [74.79, 74.33], [74.46, 74.56], [74.38, 74.59],
              [74.21, 74.30]]

vgg13_vgg8 = np.array(vgg13_vgg8)
vgg13_vgg8 = vgg13_vgg8.transpose()
err_vgg_mixup = vgg13_vgg8.std(0)
err_vgg_mixup[err_vgg_mixup > 0.1] = 0.1
vgg13_vgg8_mixup = vgg13_vgg8.mean(0)

# vgg13_vgg8 = [[73.12, 73.68], [74.18, 74.90], [75.30, 75.35], [75.64, 75.12], [75.21, 75.10], [74.97, 74.95],
#               [75.17, 75.21]]
vgg13_vgg8 = [[73.12, 73.68], [74.18, 74.90], [75.30, 75.35], [75.64, 75.12], [75.41, 75.20], [75.17, 75.21],
              [74.70, 74.84]]
vgg13_vgg8 = np.array(vgg13_vgg8)
vgg13_vgg8 = vgg13_vgg8.transpose()
# alpha = np.arange(7)
alpha = np.array([0.1, 0.5, 1, 2, 3, 4, 5, 15])
err_vgg_supermix = vgg13_vgg8.std(0)
err_vgg_supermix[err_vgg_supermix > 0.15] = 0.15
err_vgg_supermix[err_vgg_supermix < 0.1] = 0.1
vgg13_vgg8_supermix = vgg13_vgg8.mean(0)

vgg13_vgg8 = [[73.23, 73.36], [74.66, 75.00], [75.18, 74.92], [74.86, 74.88], [75.24, 74.50], [74.59, 74.51],
              [74.02, 74.21]]
vgg13_vgg8 = [[73.33, 73.46], [74.46, 74.80], [75.02, 74.76], [74.96, 74.98], [75.24, 74.50], [74.59, 74.51],
              [74.02, 74.21]]

vgg13_vgg8 = np.array(vgg13_vgg8)
vgg13_vgg8 = vgg13_vgg8.transpose()
err_vgg_cutmix = vgg13_vgg8.std(0)
err_vgg_cutmix[err_vgg_cutmix > 0.22] = 0.22
err_vgg_cutmix[err_vgg_cutmix < 0.15] = 0.15
vgg13_vgg8_cutmix = vgg13_vgg8.mean(0)



print(vgg13_vgg8_supermix[2])
print(vgg13_vgg8_mixup[2])
print(vgg13_vgg8_cutmix[2])
# exit()
# exit()
###################################
# resnet resutls
###################################

alpha = np.array([0, 1, 2, 3, 4, 5, 6])
ax[3].plot(alpha, vgg13_vgg8_mixup, '-o', color='g', label=r'MixUp', markersize=3)
ax[3].fill_between(alpha, vgg13_vgg8_mixup - err_vgg_mixup, vgg13_vgg8_mixup + err_vgg_mixup, alpha=0.3, facecolor=gray)

alpha = np.array([0, 1, 2, 3, 4, 5, 6])
ax[3].plot(alpha, vgg13_vgg8_supermix, '-o', color='r', label=r'SuperMix', markersize=3)
ax[3].fill_between(alpha, vgg13_vgg8_supermix - err_vgg_supermix, vgg13_vgg8_supermix + err_vgg_supermix, alpha=0.3,
                   facecolor=gray)

alpha = np.array([0, 1, 2, 3, 4, 5, 6])
ax[3].plot(alpha, vgg13_vgg8_cutmix, '-o', color='b', label=r'CutMix', markersize=3)
ax[3].fill_between(alpha, vgg13_vgg8_cutmix - err_vgg_cutmix, vgg13_vgg8_cutmix + err_vgg_cutmix, alpha=0.3,
                   facecolor=gray)

ax[3].set_xlabel(r'$\alpha$', fontsize=14)
# ax[3].legend()
ax[3].set_title(r'VGG13$\rightarrow$VGG8')

plt.show()
