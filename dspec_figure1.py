import numpy as np
import matplotlib.pyplot as plt
from FAST_reduction_funcs import *

# The file directories
sav_dir='data/FAST/'
img_dir='publication_figures/'

npz_name='entire_dspec.npz'
npz_data=np.load(sav_dir+npz_name)
V_dspec=npz_data['V_dspec']
timelist=npz_data['timelist']
freqlist=npz_data['freqlist']
ttbin=32

# Produce Fig. 1
fig=plt.figure(figsize=(5.6,5.5),dpi=400)
plt.rcParams.update({'font.size': 8})
cmap=plt.get_cmap('viridis_r').copy()
cmap.set_bad('lightgrey',alpha=0.5)

ax0=fig.add_axes([0.1,0.7,0.85,0.2])
a0=ax0.imshow(V_dspec.T,aspect='auto',vmin=-120,vmax=0,origin='lower',\
                 extent=[timelist[0],timelist[-1],1000,1500],cmap=cmap)
ax0.set_xlim([0,33])
ax0.set_ylabel('Frequency [MHz]')
color="tomato"
ax0.plot([timelist[189*1024//ttbin],timelist[189*1024//ttbin],\
          timelist[194*1024//ttbin],timelist[194*1024//ttbin],timelist[189*1024//ttbin]],\
        [1500,1000,1000,1500,1500],'--',color=color,clip_on=False,linewidth=1)
ax0.text(0.5*(timelist[189*1024//ttbin]+timelist[194*1024//ttbin])-0.3,1020,'B',weight='bold',color=color)
ax0.plot([timelist[204*1024//ttbin],timelist[204*1024//ttbin],\
          timelist[209*1024//ttbin],timelist[209*1024//ttbin],timelist[204*1024//ttbin]],\
        [1500,1000,1000,1500,1500],'--',color=color,clip_on=False,linewidth=1)
ax0.text(0.5*(timelist[204*1024//ttbin]+timelist[209*1024//ttbin])-0.3,1020,'C',weight='bold',color=color)
ax0.plot([timelist[244*1024//ttbin],timelist[244*1024//ttbin],\
          timelist[249*1024//ttbin],timelist[249*1024//ttbin],timelist[244*1024//ttbin]],\
        [1500,1000,1000,1500,1500],'--',color=color,clip_on=False,linewidth=1)
ax0.text(0.5*(timelist[244*1024//ttbin]+timelist[249*1024//ttbin])-0.3,1020,'D',weight='bold',color=color)
ax0.text(16,1530,'Stokes V flux density [mJy]',clip_on=False)
ax0.text(-2,1480,'A',clip_on=False,fontsize=9,weight='bold')
ax00=fig.add_axes([0.8,0.91,0.15,0.015])
a00=plt.colorbar(a0,cax=ax00,orientation='horizontal')
a00.ax.invert_xaxis()
ax00.xaxis.set_ticks_position('top')
npz_name='fs_dspec1.npz'
npz_data=np.load(sav_dir+npz_name)
V_dspec=npz_data['V_dspec']
ax1=fig.add_axes([0.1,0.45,0.85,0.18])
a1=ax1.imshow(V_dspec.T,vmin=-120,vmax=0,origin='lower',aspect='auto',\
             extent=[timelist[189*1024//ttbin],timelist[194*1024//ttbin],1000,1500],cmap=cmap)
ax1.set_ylim([freqlist[0],freqlist[-1]])
ax1.set_ylabel('Frequency [MHz]')
ax1.text(timelist[189*1024//ttbin]-2/33,1485,'B',clip_on=False,fontsize=9,weight='bold')
ax1.plot([timelist[189*1024//ttbin]+0.05,timelist[189*1024//ttbin]+0.05+300/(11.5*1e3)],[1400,1100],\
         '--',color=color,linewidth=1)
ax1.text(timelist[189*1024//ttbin]+0.05,1020,'11.5 GHz$\cdot$s$^{-1}$',color=color)
npz_name='fs_dspec2.npz'
npz_data=np.load(sav_dir+npz_name)
V_dspec=npz_data['V_dspec']
ax2=fig.add_axes([0.1,0.23,0.85,0.18])
a2=ax2.imshow(V_dspec.T,vmin=-120,vmax=0,origin='lower',aspect='auto',\
             extent=[timelist[204*1024//ttbin],timelist[209*1024//ttbin],1000,1500],cmap=cmap)
ax2.set_ylim([freqlist[0],freqlist[-1]])
ax2.set_ylabel('Frequency [MHz]')
ax2.text(timelist[204*1024//ttbin]-2/33,1485,'C',clip_on=False,fontsize=9,weight='bold')
ax2.plot([timelist[204*1024//ttbin]+0.05,timelist[204*1024//ttbin]+0.05+300/(8.7*1e3)],[1400,1100],\
         '--',color=color,linewidth=1)
ax2.text(timelist[204*1024//ttbin]+0.05,1020,'8.7 GHz$\cdot$s$^{-1}$',color=color)
npz_name='fs_dspec3.npz'
npz_data=np.load(sav_dir+npz_name)
V_dspec=npz_data['V_dspec']
ax3=fig.add_axes([0.1,0.01,0.85,0.18])
a3=ax3.imshow(V_dspec.T,vmin=-120,vmax=0,origin='lower',aspect='auto',\
             extent=[timelist[244*1024//ttbin],timelist[249*1024//ttbin],1000,1500],cmap=cmap)
ax3.set_ylim([freqlist[0],freqlist[-1]])
ax3.set_ylabel('Frequency [MHz]')
ax3.set_xlabel('Time [s] since 2022-03-19 15:05:40 UT')
ax3.text(timelist[244*1024//ttbin]-2/33,1485,'D',clip_on=False,fontsize=9,weight='bold')
ax3.plot([timelist[244*1024//ttbin]+0.05,timelist[244*1024//ttbin]+0.05+300/(7.5*1e3)],[1400,1100],\
         '--',color=color,linewidth=1)
ax3.text(timelist[244*1024//ttbin]+0.05,1020,'7.5 GHz$\cdot$s$^{-1}$',color=color)
fig.savefig(img_dir+'figure1.pdf',format='pdf',bbox_inches='tight')






