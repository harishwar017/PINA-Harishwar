import scipy.io
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter 
import numpy as np


method = 'AD'

Data_used = 2


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
u_actual_selected = u_actual[:,point_list]
x_pred_selected = x_pred[:,point_list]


u_pred_vector = u_pred_selected.T.reshape(-1,1)
u_act_vector = u_actual_selected.T.reshape(-1,1)
x_vector = x_pred_selected.T.reshape(-1,1)


# This is the final example I showed in the code - notice I have 2 "cursor marks" not shown in the video
fig = plt.figure()
l, = plt.plot([], [], 'k-',label='AD')
p1, = plt.plot([], [], 'ko')
l2, = plt.plot([], [], 'm--',label='Benchmark')
p2, = plt.plot([], [], 'mo')

plt.xlabel('x/L')
plt.ylabel('u')
plt.title('Conventional PINN vs Benchmark')
plt.legend(ncol=2, loc=8, fontsize=10) # 9 means top center


plt.xlim(0, 1)
plt.ylim(0.01, 1.1)



metadata = dict(title='Movie', artist='codinglikemad')
writer = PillowWriter(fps=100, metadata=metadata)


xlist = []
ylist1 = []
ylist2 = []


xval = x_vector
upredval = u_pred_vector
uactval = u_act_vector



with writer.saving(fig, "sinWave2.gif", 500):

    # Plot the first line and cursor
    for i in range(len(xval)):
        xlist.append(xval[i])
        ylist1.append(upredval[i])
        ylist2.append(uactval[i])

        l.set_data(xlist,ylist1)
        l2.set_data(xlist,ylist2)

        p1.set_data(xval[i],upredval[i])
        p2.set_data(xval[i],uactval[i])
        writer.grab_frame()


    
plt.savefig('Plots_comparison_2D/burgers_AD.png', dpi=300)



    