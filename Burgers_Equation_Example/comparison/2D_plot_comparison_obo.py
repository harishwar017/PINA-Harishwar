# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 15:43:02 2023

@author: rahul
"""

import scipy.io
import matplotlib.pyplot as plt

method = 'AD'

Data_used = 0


# Ground Truth
u_actual = scipy.io.loadmat('../Results.mat')['u_Mat']
time_actual = scipy.io.loadmat('../Results.mat')['total_time'].T
x_actual = scipy.io.loadmat('../Results.mat')['total_L']

# Prediction

u_pred = scipy.io.loadmat('../Burgers_Dis_{}_{}.mat'.format(method,Data_used))['predicted_output_{}_{}'.format(method,Data_used)].reshape(100,20).T
x_pred = scipy.io.loadmat('../Burgers_Dis_{}_{}.mat'.format(method,Data_used))['pts_array_{}_{}'.format(method,Data_used)][:,0].reshape(100,20).T
time_pred = scipy.io.loadmat('../Burgers_Dis_{}_{}.mat'.format(method,Data_used))['pts_array_{}_{}'.format(method,Data_used)][:,1].reshape(100,20).T




fig, ax = plt.subplots(figsize=(10, 10))

#plot Original 

custom_xticks = [0.2, 0.4, 0.6, 0.8]  # Replace with your desired custom tick values

cb = getattr(ax, 'contourf')(x_actual,time_actual,u_actual)
colorbar =fig.colorbar(cb,ax = ax)
colorbar.ax.tick_params(labelsize=32)
ax.set_title("Ground Truth",fontsize=30)
ax.set_xlabel('x',fontsize=32)
ax.set_ylabel('t',fontsize=32)
ax.set_xticks(custom_xticks)
ax.tick_params(axis='x', labelsize=32)
ax.tick_params(axis='y', labelsize=32)


#plot prediciton

# cb = getattr(ax, 'contourf')(x_pred,time_pred,u_pred)
# colorbar =fig.colorbar(cb,ax = ax)
# colorbar.ax.tick_params(labelsize=32)
# ax.set_title("Prediction",fontsize=32)
# ax.set_xlabel('x',fontsize=30)
# ax.set_ylabel('t',fontsize=30)
# ax.set_xticks(custom_xticks)
# ax.tick_params(axis='x', labelsize=32)
# ax.tick_params(axis='y', labelsize=32)


#plot absolute error

# cb = getattr(ax, 'contourf')(x_pred,time_pred,abs(u_pred-u_actual))
# colorbar =fig.colorbar(cb,ax = ax)
# colorbar.ax.tick_params(labelsize=32)
# ax.set_title("error (MAE)",fontsize=32)
# ax.set_xlabel('x',fontsize=32)
# ax.set_ylabel('t',fontsize=32)
# ax.set_xticks(custom_xticks)
# ax.tick_params(axis='x', labelsize=32)
# ax.tick_params(axis='y', labelsize=32)



# Adjust x-axis and y-axis tick label font sizes
# for i in range(3):
#     ax[i].tick_params(axis='x', labelsize=20)
#     ax[i].tick_params(axis='y', labelsize=20)

plt.savefig('Plots_comparison_2D/{}_{}_Ground_Truth.pdf'.format(method, Data_used), dpi=1200)


# a = abs(u_pred-u_actual)

# a_plain = a.reshape(-1,1)

# a_max = max(a_plain)




            
# cb = getattr(ax[1], method)(*grids, truth_output.cpu().detach(), **kwargs)
#             fig.colorbar(cb, ax=ax[1])
# cb = getattr(ax[2], method)(*grids,
#                                         (truth_output-pred_output).cpu().detach(),
#                                         **kwargs)
#             fig.colorbar(cb, ax=ax[2])

