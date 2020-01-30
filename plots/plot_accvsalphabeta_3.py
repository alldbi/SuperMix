import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import numpy as np

fig, ax = plt.subplots(1, 3)

# set size and resolution of the figure
fig.set_figheight(4)
fig.set_figwidth(12)
fig.set_dpi(120)
plt.tight_layout()
fig.subplots_adjust(top=0.9, bottom=0.18, left=0.044)

#############################################################################
############ acc vs alpha ###################################################
#############################################################################


# set ax properties
ax[0].set_xticks(np.arange(10))
# ax[0].set_xticks([0.1, 0.5, 1, 2, 3, 4, 5, 10, 15])
# ax[0].set_xticklabels(['0.1', '0.2', '0.3', '0.4', '1', '2', '3', '4', '5', '10'])
ax[0].set_xticklabels(['0.1', '0.5', '1', '3', '5', '15', r'$10^4$'])
ax[0].grid(axis='y')
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)
ax[0].xaxis.set_label_coords(0.5, -0.14)
ax[0].set_xlim([0, 6])

# wrn = [[74.46, 74.64], [75.98, 75.41], [75.91, 76.32], [76.40, 76.00], [76.34, 76.13], [76.07, 76.76], [75.78, 75.83]]
wrn = [[74.46, 74.64], [75.98, 75.41], [75.91, 76.32], [76.40, 76.00], [76.34, 76.13], [76.07, 76.26], [75.78, 75.83]]
wrn = np.array(wrn).transpose(1, 0)
err_wrn_mixup = wrn.std(0)
err_wrn_mixup[err_wrn_mixup>0.15]=0.15
wrn_mixup = wrn.mean(0)

wrn = [[76.05, 76.02, 75.90],[77.02, 76.50, 76.30], [76.53, 76.97, 76.80], [76.62, 76.99, 77.18], [76.85, 76.70, 77.10],[76.47, 77.05, 76.90], [76.67, 76.39, 76.93]]
wrn = np.array(wrn).transpose(1, 0)
err_wrn_supermix = wrn.std(0)
err_wrn_supermix[err_wrn_supermix>0.2]=0.2
wrn_supermix = wrn.mean(0)

###################################
# wrn resutls
###################################
alpha = np.array([0, 1, 2, 3, 4, 5, 6])
ax[0].plot(alpha, wrn_mixup, '-o', color='r', label=r'MixUp', markersize=3)
ax[0].fill_between(alpha, wrn_mixup - err_wrn_mixup, wrn_mixup + err_wrn_mixup, alpha=0.3, facecolor='r')


alpha = np.array([0, 1, 2, 3, 4, 5, 6])
ax[0].plot(alpha, wrn_supermix, '-o', color='g', label=r'SuperMix', markersize=3)
ax[0].fill_between(alpha, wrn_supermix - err_wrn_supermix, wrn_supermix + err_wrn_supermix, alpha=0.3, facecolor='g')

ax[0].set_xlabel(r'$\alpha$', fontsize=14)
ax[0].legend()
ax[0].set_title(r'WRN-40-2$\rightarrow$WRN-16-2')

#############################################################################
############ acc vs lambda ##################################################
#############################################################################
# set ax properties
ax[1].set_xticks(np.arange(10))
# ax[0].set_xticks([0.1, 0.5, 1, 2, 3, 4, 5, 10, 15])
# ax[0].set_xticklabels(['0.1', '0.2', '0.3', '0.4', '1', '2', '3', '4', '5', '10'])
ax[1].set_xticklabels(['0.1', '0.5', '1', '3', '5', '15', r'$10^4$'])
ax[1].grid(axis='y')
ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)
ax[1].xaxis.set_label_coords(0.5, -0.14)
ax[1].set_xlim([0, 6])

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
err_vgg_supermix[err_vgg_supermix <0.1] = 0.1
vgg13_vgg8_supermix = vgg13_vgg8.mean(0)




vgg13_vgg8 = [[75.18,74.92],[74.86,74.88],[75.24,74.50],[74.59,74.51],[74.02,74.21]]

vgg13_vgg8 = np.array(vgg13_vgg8)
vgg13_vgg8 = vgg13_vgg8.transpose()
err_vgg_cutmix = vgg13_vgg8.std(0)
# err_vgg_mixup[err_vgg_mixup > 0.1] = 0.1
vgg13_vgg8_cutmix = vgg13_vgg8.mean(0)

###################################
# resnet resutls
###################################

alpha = np.array([0, 1, 2, 3, 4, 5, 6])
ax[1].plot(alpha, vgg13_vgg8_mixup, '-o', color='r', label=r'MixUp', markersize=3)
ax[1].fill_between(alpha, vgg13_vgg8_mixup - err_vgg_mixup, vgg13_vgg8_mixup + err_vgg_mixup, alpha=0.3, facecolor='r')

alpha = np.array([0, 1, 2, 3, 4, 5, 6])
ax[1].plot(alpha, vgg13_vgg8_supermix, '-o', color='g', label=r'SuperMix', markersize=3)
ax[1].fill_between(alpha, vgg13_vgg8_supermix - err_vgg_supermix, vgg13_vgg8_supermix + err_vgg_supermix, alpha=0.3,
                   facecolor='g')

alpha = np.array([ 2, 3, 4, 5, 6])
ax[1].plot(alpha, vgg13_vgg8_cutmix, '-o', color='b', label=r'CutMix', markersize=3)
ax[1].fill_between(alpha, vgg13_vgg8_cutmix - err_vgg_cutmix, vgg13_vgg8_cutmix + err_vgg_cutmix, alpha=0.3,
                   facecolor='b')


ax[1].set_xlabel(r'$\alpha$', fontsize=14)
ax[1].legend()
ax[1].set_title(r'VGG13$\rightarrow$VGG8')
plt.show()
