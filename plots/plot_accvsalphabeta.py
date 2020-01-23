import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import numpy as np

fig, ax = plt.subplots(1, 4)

# set size and resolution of the figure
fig.set_figheight(2.6)
fig.set_figwidth(12)
fig.set_dpi(120)
plt.tight_layout()
fig.subplots_adjust(top=0.9, bottom=0.18, left=0.044)

#############################################################################
############ acc vs alpha ###################################################
#############################################################################


# set ax properties
# ax[0].set_xticks(np.arange(7))
ax[0].set_xticks([0.1, 0.3, 1, 3, 5, 10])
# ax[0].set_xticklabels(['0.1', '0.2', '0.3', '0.4', '1', '2', '3', '4', '5', '10'])
ax[0].set_xticklabels(['0.1', '0.3', '1', '3', '5', '10', '15'])
ax[0].grid(axis='y')
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)
ax[0].xaxis.set_label_coords(0.5, -0.14)

# vgg13_vgg8 = [[72.61, 73.27, 73.64, 73.98, 74.42, 74.76, 74.75, 74.65, 74.54, 74.30],
#               [72.86, 73.14, 73.21, 73.75, 74.31, 74.72, 74.62, 74.67, 74.66, 74.35],
#               [72.76, 72.97, 73.50, 73.99, 74.31, 75.05, 74.69, 74.53, 74.80, 74.30]]

vgg13_vgg8 = [[72.61, 73.64, 74.42, 74.75, 74.54, 74.30, 75.05],
              [72.86, 73.21, 74.31, 74.62, 74.66, 74.35, 74.68],
              [72.76, 73.50, 74.31, 74.69, 74.80, 74.30, 74.45]]

vgg13_vgg8 = np.array(vgg13_vgg8)
# alpha = np.arange(7)
alpha = np.array([0.1, 0.3, 1, 3, 5, 10])
err_vgg = vgg13_vgg8.std(0)
vgg13_vgg8 = vgg13_vgg8.mean(0)

res110_res20 = [[70.43, 71.46, 72.62, 72.59, 72.23],
                [70.04, 71.90, 72.41, 72.43, 72.23],
                [70.45, 71.82, 71.83, 72.12, 72.23]]

res110_res20 = np.array(res110_res20)

err_res = res110_res20.std(0)
res110_res20 = res110_res20.mean(0)

# alpha = np.arange(7)
alpha = np.array([0.1, 0.3, 1, 3, 5, 10, 15])
ax[0].plot(alpha, vgg13_vgg8, '-o', color='b', label='VGG', markersize=3)
ax[0].fill_between(alpha, vgg13_vgg8 - err_vgg, vgg13_vgg8 + err_vgg, alpha=0.3, facecolor='b')

# alpha = np.array([0, 1, 2, 4, 5])
# ax[0].plot(alpha, res110_res20, '-o', color='r', label='ResNet', markersize=3)
# ax[0].fill_between(alpha, res110_res20 - err_res, res110_res20 + err_res, alpha=0.3, facecolor='r')

ax[0].set_xlabel(r'$\alpha$', fontsize=14)
ax[0].legend()
# ax[0].set_title(r'ResNet110$\rightarrow$ResNet20')

#############################################################################
############ acc vs lambda ##################################################
#############################################################################


# set ax properties
ax[1].set_xticks(np.arange(5))
ax[1].set_xticklabels(['0.1', '0.2', '0.3', '0.4', '0.5'])
ax[1].grid(axis='y')
ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)
ax[1].xaxis.set_label_coords(0.5, -0.14)

vgg13_vgg8 = [[70.41, 73.03, 74.16, 74.18, 73.84],
              [70.80, 72.55, 74.84, 74.26, 73.97]]
vgg13_vgg8 = np.array(vgg13_vgg8)

alpha = np.arange(5)
err = vgg13_vgg8.std(0)
vgg13_vgg8 = vgg13_vgg8.mean(0)

ax[1].plot(alpha, vgg13_vgg8, '-o', color='b', label='VGG-cls', markersize=3)
ax[1].fill_between(alpha, vgg13_vgg8 - err, vgg13_vgg8 + err, alpha=0.3, facecolor='b')

vgg13_vgg8 = [[71.67, 73.90, 74.76, 74.96, 74.50],
              [71.59, 73.60, 74.58, 74.81, 74.51],
              [70.71, 73.88, 74.80, 74.72, 74.39]]

vgg13_vgg8 = np.array(vgg13_vgg8)

alpha = np.arange(5)
err = vgg13_vgg8.std(0)
vgg13_vgg8 = vgg13_vgg8.mean(0)

ax[1].plot(alpha, vgg13_vgg8, '-o', color='r', label='VGG-newkd', markersize=3)
ax[1].fill_between(alpha, vgg13_vgg8 - err, vgg13_vgg8 + err, alpha=0.3, facecolor='r')

ax[1].set_xlabel(r'$\lambda$', fontsize=14)
ax[1].legend()
plt.show()
