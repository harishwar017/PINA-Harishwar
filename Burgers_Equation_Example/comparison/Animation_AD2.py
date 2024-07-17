# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 22:41:19 2023

@author: rahul
"""

import numpy as np
from matplotlib.animation import FuncAnimation,PillowWriter
import matplotlib.pyplot as plt
import scipy.io


method = 'AD'

Data_used = 1000


# Ground Truth
u_actual = scipy.io.loadmat('../Results.mat')['u_Mat']
time_actual = scipy.io.loadmat('../Results.mat')['total_time'].T
x_actual = scipy.io.loadmat('../Results.mat')['total_L']

# Prediction

u_pred = scipy.io.loadmat('../Burgers_Dis_{}_{}.mat'.format(method,Data_used))['predicted_output_{}_{}'.format(method,Data_used)].reshape(100,20).T
x_pred = scipy.io.loadmat('../Burgers_Dis_{}_{}.mat'.format(method,Data_used))['pts_array_{}_{}'.format(method,Data_used)][:,0].reshape(100,20).T
time_pred = scipy.io.loadmat('../Burgers_Dis_{}_{}.mat'.format(method,Data_used))['pts_array_{}_{}'.format(method,Data_used)][:,1].reshape(100,20).T


point_mid = np.linspace(0,99,5,dtype=('int'))

point_list = list(point_mid)



u_pred_selected = u_pred[:,point_list]
x_pred_selected = x_pred[:,point_list]


u_vector = u_pred_selected.T.reshape(-1,1)
x_vector = x_pred_selected.T.reshape(-1,1)
 
# Create the traveling wave
def wave(x, t, wavelength, speed):
    return np.sin((2*np.pi)*(x-speed*t)/wavelength)
 
x = x_vector
yt = u_vector
 
# Create the figure and axes to animate
fig, ax = plt.subplots(1)
# init_func() is called at the beginning of the animation

def init_func():
    ax.clear()
 
# update_plot() is called between frames
def update_plot(i):
    ax.clear()
    ax.plot(x[0,:], yt[i,:], color='k')
 
# Create animation
anim = FuncAnimation(fig,
                     update_plot,
                     frames=30,
                     init_func=init_func)
 
# Save animation
anim.save('traveling_wave.gif',
          dpi=150,
          fps=30,
          writer='ffmpeg')