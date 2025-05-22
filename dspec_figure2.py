import numpy as np
import matplotlib.pyplot as plt
from FAST_reduction_funcs import *

# The file directories
sav_dir='data/FAST/'
img_dir='publication_figures/'

npz_name='entire_dspec.npz'
npz_data=np.load(sav_dir+npz_name)
I_dspec=npz_data['I_dspec']
V_dspec=npz_data['V_dspec']
timelist=npz_data['timelist']
freqlist=npz_data['freqlist']
ttbin=32

# Produce Fig. S1
fig1=plt.figure(num=1,figsize=(5.6,5),dpi=400)
plt.rcParams.update({'font.size': 8})
plt.subplot(3,1,1)
cmap=plt.get_cmap('viridis').copy()
cmap.set_bad('lightgrey',alpha=0.5)
a=plt.imshow(I_dspec.T,aspect='auto',vmin=0,vmax=120,origin='lower',\
                 extent=[timelist[0],timelist[-1],1000,1500],cmap=cmap)
plt.xlim([0,33])
plt.colorbar(a,label='Stokes I [mJy]',aspect=40,pad=0.02)
plt.ylabel('Frequency [MHz]')
plt.text(-5,1480,'A',fontsize=9,color='black',weight='bold')
plt.subplot(3,1,2)
cmap=plt.get_cmap('viridis_r').copy()
cmap.set_bad('lightgrey',alpha=0.5)
a=plt.imshow(V_dspec.T,aspect='auto',vmin=-120,vmax=0,origin='lower',\
                 extent=[timelist[0],timelist[-1],1000,1500],cmap=cmap)
plt.xlim([0,33])
plt.colorbar(a,label='Stokes V [mJy]',aspect=40,pad=0.02)
plt.ylabel('Frequency [MHz]')
plt.text(-5,1480,'B',fontsize=9,color='black',weight='bold')
plt.subplot(3,1,3)
cmap=plt.get_cmap('bwr_r').copy()
cmap.set_bad('lightgrey',alpha=0.5)
a=plt.imshow((I_dspec/V_dspec*100).T,aspect='auto',vmin=-140,vmax=-60,origin='lower',\
                 extent=[timelist[0],timelist[-1],1000,1500],cmap=cmap)
plt.xlim([0,33])
plt.colorbar(a,label='Polarization degree [%]',aspect=40,pad=0.02)
plt.xlabel('Time [s] since 2022-03-19 15:05:40 UT')
plt.ylabel('Frequency [MHz]')
plt.text(-5,1480,'C',fontsize=9,color='black',weight='bold')
plt.subplots_adjust(hspace=0.25)
fig1.savefig(img_dir+'sfigure1.pdf',format='pdf',bbox_inches='tight')

# Produce Fig. S2

fig2=plt.figure(figsize=(3.55,7),dpi=400)

plt.subplot(3,1,1)
i0=find_neartime(0,timelist)
i1=find_neartime(33,timelist)
hist_data=(I_dspec[i0:i1,:]).flatten()
a=plt.hist(hist_data,range=[-1000,1000],bins=100,alpha=0.4,label='All')
bkgd=np.ones(len(a[0][:]))
bkgd[0:50]=a[0][0:50]
bkgd[50:100]=a[0][0:50][::-1]
plt.stairs(bkgd,a[1],color='green',label='Background')
plt.stairs(a[0]-bkgd,a[1],color='red',label='Radio emission')
plt.yscale('log')
plt.ylabel('Counts')
plt.ylim([20,3e6])
plt.xlim([-800,800])
plt.xlabel('Stokes I flux density [mJy]')
plt.legend(loc='upper left')
plt.text(-1050,1.5e6,'A',fontsize=9,color='black',weight='bold')

plt.subplot(3,1,2)
hist_data=(V_dspec[i0:i1,:]).flatten()
a=plt.hist(hist_data,range=[-1000,1000],bins=100,alpha=0.4,label='All')
bkgd=np.ones(len(a[0][:]))
bkgd[50:100]=a[0][50:100]
bkgd[0:50]=a[0][50:100][::-1]
plt.stairs(bkgd,a[1],color='green')
plt.stairs(a[0]-bkgd,a[1],color='red')
plt.yscale('log')
plt.ylabel('Counts')
plt.ylim([20,3e6])
plt.xlim([800,-800])
plt.xlabel('Stokes V flux density [mJy]')
plt.text(1050,1.5e6,'B',fontsize=9,color='black',weight='bold')

plt.subplot(3,1,3)
hist_data=(V_dspec[i0:i1,:]/I_dspec[i0:i1,:]).flatten()*100
a=plt.hist(hist_data,range=[-200,200],bins=100,alpha=0.4,label='All')
bkgd=np.ones(len(a[0][:]))
bkgd[50:100]=a[0][50:100]
bkgd[0:50]=a[0][50:100][::-1]
plt.stairs(bkgd,a[1],color='green')
plt.stairs(a[0]-bkgd,a[1],color='red')
plt.yscale('log')
plt.ylabel('Counts')
plt.xlabel('Polarization degree [%]')
plt.ylim([5e3,1.2e5])
plt.xlim([200,-200])
plt.text(260,1e5,'C',fontsize=9,color='black',weight='bold')

plt.subplots_adjust(hspace=0.42)
fig2.savefig(img_dir+'sfigure2.pdf',format='pdf',bbox_inches='tight')
