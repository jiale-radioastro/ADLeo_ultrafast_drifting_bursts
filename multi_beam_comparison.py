import numpy as np
import matplotlib.pyplot as plt
from FAST_reduction_funcs import *


# Produce Fig. S3

file_dir="/Volumes/jiale_disk1/projects/PT2021_0019/20220319/"
img_dir='publication_figures/'

ttbin=32
ttbin2=1
timelist=give_time(tbin=0.000196608,ttbin=ttbin,ttbin2=ttbin2,length=512*1024,file=0/16384)-22.99

dx=0.2
dy=0.2
wx=0.18
wy=0.16
x0=0.5
y0=0.5

fig=plt.figure(figsize=(7.25,6),dpi=400)
plt.rcParams.update({'font.size': 8})
axlist=[]
for i in range(19):
    if i==0:
        ax = fig.add_axes([x0, y0, wx, wy])
        ax.plot()
        
    if i==1:
        ax = fig.add_axes([x0-dx, y0, wx, wy])
        ax.plot()
    if i==2:
        ax = fig.add_axes([x0-0.5*dx, y0-dy, wx, wy])
        ax.plot()
    if i==3:
        ax = fig.add_axes([x0+0.5*dx, y0-dy, wx, wy])
        ax.plot()
    if i==4:
        ax = fig.add_axes([x0+dx, y0, wx, wy])
        ax.plot()
    if i==5:
        ax = fig.add_axes([x0+0.5*dx, y0+dy, wx, wy])
        ax.plot()
    if i==6:
        ax = fig.add_axes([x0-0.5*dx, y0+dy, wx, wy])
        ax.plot()
        
    if i==7:
        ax = fig.add_axes([x0-2*dx, y0, wx, wy])
        ax.plot()
    if i==8:
        ax = fig.add_axes([x0-1.5*dx, y0-dy, wx, wy])
        ax.plot()
    if i==9:
        ax = fig.add_axes([x0-1*dx, y0-2*dy, wx, wy])
        ax.plot()
    if i==10:
        ax = fig.add_axes([x0, y0-2*dy, wx, wy])
        ax.plot()
    if i==11:
        ax = fig.add_axes([x0+dx, y0-2*dy, wx, wy])
        ax.plot()
    if i==12:
        ax = fig.add_axes([x0+1.5*dx, y0-dy, wx, wy])
        ax.plot()
    if i==13:
        ax = fig.add_axes([x0+2*dx, y0, wx, wy])
        ax.plot()
    if i==14:
        ax = fig.add_axes([x0+1.5*dx, y0+dy, wx, wy])
        ax.plot()
    if i==15:
        ax = fig.add_axes([x0+dx, y0+2*dy, wx, wy])
        ax.plot()
    if i==16:
        ax = fig.add_axes([x0, y0+2*dy, wx, wy])
        ax.plot()
    if i==17:
        ax = fig.add_axes([x0-dx, y0+2*dy, wx, wy])
        ax.plot()
    if i==18:
        ax = fig.add_axes([x0-1.5*dx, y0+dy, wx, wy])
        ax.plot()
    
    ax.set_title('Beam M'+str(i+1).zfill(2))
    if i not in [9,10,11]:
        ax.set_xticks([])
    if i not in [7,8,9,17,18]:
        ax.set_yticks([])
    if i==7:
        ax.set_ylabel('Frequency [MHz]')
    if i==10:
        ax.set_xlabel('Time [s] since 2022-03-19 15:05:40 UT')
    if i==15:
        ax1 = fig.add_axes([x0+2.4*dx, y0+2*dy, 0.015, wy])
        plt.colorbar(a,cax=ax1,label='Stokes V flux density [mJy]')
        
    beam=i+1
    V_dspec=np.load(file_dir+'multibeam_dspec/'+'V_dspec_'+str(beam).zfill(2)+'.npy')
    plot_mask=give_mask(V_dspec,method='channel',thresh=2)

    index0=find_neartime(timelist,0)
    index1=find_neartime(timelist,33)
    V_mask1=give_mask(V_dspec,method='time',time=np.arange(index0,index1,1))
    if i==0:
        plot_mask=give_mask(V_dspec*V_mask1,method='channel',thresh=10)
    a=ax.imshow((V_dspec*plot_mask).T,origin='lower',aspect='auto',cmap='viridis_r',vmin=-30,vmax=30,\
              extent=[timelist[0],timelist[-1],1000,1500])
    ax.set_xlim([0,50])
fig.savefig(img_dir+'sfigure3.pdf',format='pdf',bbox_inches='tight')