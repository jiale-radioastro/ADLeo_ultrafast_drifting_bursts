import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from scipy.optimize import curve_fit
from scipy.fft import fft,fft2,fftshift,fftfreq,ifft,ifft2
from FAST_reduction_funcs import *

def radon_trans(img,k,radius=0):
    # Radon transformation
    sz=np.shape(img)
    if radius==0:
        radius=np.min(np.shape(img))
    f=interp2d(np.arange(sz[0]),np.arange(sz[1]),img.T)
    theta=np.arctan(k)
    r=np.arange(-radius+1,radius,1)
    x=(sz[0]-1)/2+r*np.cos(theta)
    y=(sz[1]-1)/2+r*np.sin(theta)
    return np.sum([f(x[i],y[i]) for i in range(len(r))])

def normalize(lc):
    return (lc-np.min(lc))/(np.max(lc)-np.min(lc))

def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + 0.5



# The file directories
sav_dir='data/FAST/'
img_dir='publication_figures/'

# Produce Fig. S1
fig, ax=plt.subplots(2,3,figsize=(7.25,5),dpi=400,gridspec_kw={'height_ratios':[3,2],'hspace':0.45,'wspace':0.08})
plt.rcParams.update({'font.size': 8})
plt.set_cmap('Greys')

npz_name='fs_dspec1.npz'
npz_data=np.load(sav_dir+npz_name)
V_dspec=npz_data['V_dspec']
freqlist=npz_data['freqlist']
ttbin2=2
fft_img=fftshift(fft2(fix_nan(V_dspec,mode='freq')))
sz=np.shape(fft_img)
f_time=fftshift(fftfreq(sz[0],d=0.000196608*ttbin2))
f_freq=fftshift(fftfreq(sz[1],d=freqlist[1]-freqlist[0]))

ax00=ax[0,0]
ax00.imshow(np.log10(np.abs(fft_img)**2).T,aspect='auto',vmin=10.5,vmax=11.5,origin='lower',\
           extent=[f_time[0],f_time[-1],f_freq[0],f_freq[-1]])
ax00.set_xlim([-400,400])
ax00.set_ylim([-0.07,0.07])
ax00.set_ylabel('Frequency$^{-1}$ [MHz$^{-1}$]')
ax00.set_xlabel('Time$^{-1}$ [s$^{-1}$]')
ax00.set_xticks([-400,-200,0,200],[-400,-200,0,200])
ax00.set_yticks([-0.06,-0.03,0,0.03,0.06],[-0.06,-0.03,0,0.03,0.06])
ax00.text(300,0.055,'A',fontsize=9,color='black',weight='bold')

klist=np.arange(0.01,0.15,0.002)
tlist=[]
for k in klist:
    tlist.append(radon_trans(np.abs(fft_img)**2,k,400))
tlist=np.array(normalize(tlist))
slope=klist*(f_freq[1]-f_freq[0])/(f_time[1]-f_time[0])*1e3
ax10=ax[1,0]
ax10.set_ylabel('Normalized intensity')
ax10.set_xlabel('Slope [GHz$^{-1}\cdot$s]')
ax10.set_yticks([0,0.5,1],[0,0.5,1])
index=np.where(tlist>0.5)
popt, pcov = curve_fit(gaussian, slope[index], tlist[index], p0=[0.5,0.1,0.03])
xdata= np.arange(slope[index][0],slope[index][-1],0.001)
ax10.set_ylim([-0.05,1.05])
mu_fit = popt[1]
mu_err = popt[2]
ax10.plot(slope,tlist,':',color='Black',linewidth=1.5,label='Integrated value')
ax10.plot(xdata,gaussian(xdata,popt[0],popt[1],popt[2]),color='blue',linewidth=1.5,\
          alpha=0.6,label='Gaussian fit')
ax10.plot([mu_fit,mu_fit],[-0.05,1.05],'--',color='red',label='Slope value',alpha=0.8)
ax10.fill_between([mu_fit-mu_err,mu_fit+mu_err],[-0.05,-0.05],[1.05,1.05],color='red',edgecolor=None,alpha=0.2)
ax10.legend()
ax10.text(0.21,0.9,"$\mu=$"+"{:.3f}".format(mu_fit))
ax10.text(0.21,0.75,"$\sigma=$"+"{:.3f}".format(mu_err))
ax10.text(0.02,0.9,'D',fontsize=9,color='black',weight='bold')
ax00.plot([-400,400],[-400*mu_fit/1e3,400*mu_fit/1e3],'--',color='red',alpha=0.8,\
          label='Fitted slope')
ax00.text(-60,-0.06,'Drift rate:\n'+'{:.1f}'.format(1/mu_fit)+'$\pm$'+'{:.1f}'.format(1/mu_fit**2*mu_err)+\
          ' GHz$\cdot$s$^{-1}$')
ax00.legend(loc='upper left')

npz_name='fs_dspec2.npz'
npz_data=np.load(sav_dir+npz_name)
V_dspec=npz_data['V_dspec']
fft_img=fftshift(fft2(fix_nan(V_dspec,mode='freq')))
ax01=ax[0,1]
ax01.imshow(np.log10(np.abs(fft_img)**2).T,aspect='auto',vmin=10,vmax=12.2,origin='lower',\
           extent=[f_time[0],f_time[-1],f_freq[0],f_freq[-1]])
ax01.set_xlim([-400,400])
ax01.set_ylim([-0.07,0.07])
ax01.set_xlabel('Time$^{-1}$ [s$^{-1}$]')
ax01.set_yticks([])
ax01.set_xticks([-400,-200,0,200],[-400,-200,0,200])
ax01.text(300,0.055,'B',fontsize=9,color='black',weight='bold')

klist=np.arange(0.01,0.15,0.002)
tlist=[]
for k in klist:
    tlist.append(radon_trans(np.abs(fft_img)**2,k,400))
tlist=np.array(normalize(tlist))
slope=klist*(f_freq[1]-f_freq[0])/(f_time[1]-f_time[0])*1e3
ax11=ax[1,1]
ax11.set_xlabel('Slope [GHz$^{-1}\cdot$s]')
index=np.where(tlist>0.5)
popt, pcov = curve_fit(gaussian, slope[index], tlist[index], p0=[0.5,0.1,0.03])
xdata= np.arange(slope[index][0],slope[index][-1],0.001)
ax11.set_ylim([-0.05,1.05])
ax11.set_yticks([])
mu_fit = popt[1]
mu_err = popt[2]
ax11.plot(slope,tlist,':',color='Black',linewidth=1.5)
ax11.plot(xdata,gaussian(xdata,popt[0],popt[1],popt[2]),color='blue',linewidth=1.5,alpha=0.6)
ax11.plot([mu_fit,mu_fit],[-0.05,1.05],'--',color='red',alpha=0.8)
ax11.fill_between([mu_fit-mu_err,mu_fit+mu_err],[-0.05,-0.05],[1.05,1.05],color='red',edgecolor=None,alpha=0.2)
ax11.text(0.21,0.9,"$\mu=$"+"{:.3f}".format(mu_fit))
ax11.text(0.21,0.75,"$\sigma=$"+"{:.3f}".format(mu_err))
ax11.text(0.02,0.9,'E',fontsize=9,color='black',weight='bold')
ax01.plot([-400,400],[-400*mu_fit/1e3,400*mu_fit/1e3],'--',color='red',alpha=0.8)
ax01.text(-60,-0.06,'Drift rate:\n'+'{:.1f}'.format(1/mu_fit)+'$\pm$'+'{:.1f}'.format(1/mu_fit**2*mu_err)+\
          ' GHz$\cdot$s$^{-1}$')

npz_name='fs_dspec3.npz'
npz_data=np.load(sav_dir+npz_name)
V_dspec=npz_data['V_dspec']
fft_img=fftshift(fft2(fix_nan(V_dspec,mode='freq')))
ax02=ax[0,2]
ax02.imshow(np.log10(np.abs(fft_img)**2).T,aspect='auto',vmin=10,vmax=12,origin='lower',\
           extent=[f_time[0],f_time[-1],f_freq[0],f_freq[-1]])
ax02.set_xlim([-400,400])
ax02.set_ylim([-0.07,0.07])
ax02.set_xlabel('Time$^{-1}$ [s$^{-1}$]')
ax02.set_yticks([])
ax02.set_xticks([-400,-200,0,200,400],[-400,-200,0,200,400])
ax02.text(300,0.055,'C',fontsize=9,color='black',weight='bold')

klist=np.arange(0.01,0.15,0.002)
tlist=[]
for k in klist:
    tlist.append(radon_trans(np.abs(fft_img)**2,k,400))
tlist=np.array(normalize(tlist))
slope=klist*(f_freq[1]-f_freq[0])/(f_time[1]-f_time[0])*1e3
ax12=ax[1,2]
ax12.set_xlabel('Slope [GHz$^{-1}\cdot$s]')
index=np.where(tlist>0.5)
popt, pcov = curve_fit(gaussian, slope[index], tlist[index], p0=[0.5,0.1,0.03])
xdata= np.arange(slope[index][0],slope[index][-1],0.001)
ax12.set_ylim([-0.05,1.05])
ax12.set_yticks([])
mu_fit = popt[1]
mu_err = popt[2]
ax12.plot(slope,tlist,':',color='Black',linewidth=1.5)
ax12.plot(xdata,gaussian(xdata,popt[0],popt[1],popt[2]),color='blue',linewidth=1.5,\
          alpha=0.6)
ax12.plot([mu_fit,mu_fit],[-0.05,1.05],'--',color='red',alpha=0.8)
ax12.fill_between([mu_fit-mu_err,mu_fit+mu_err],[-0.05,-0.05],[1.05,1.05],color='red',edgecolor=None,alpha=0.2)
ax12.text(0.21,0.9,"$\mu=$"+"{:.3f}".format(mu_fit))
ax12.text(0.21,0.75,"$\sigma=$"+"{:.3f}".format(mu_err))
ax12.text(0.02,0.9,'F',fontsize=9,color='black',weight='bold')
ax02.plot([-400,400],[-400*mu_fit/1e3,400*mu_fit/1e3],'--',color='red',alpha=0.8)
ax02.text(-60,-0.06,'Drift rate:\n'+'{:.1f}'.format(1/mu_fit)+'$\pm$'+'{:.1f}'.format(1/mu_fit**2*mu_err)+\
          ' GHz$\cdot$s$^{-1}$')
fig.savefig(img_dir+'sfigure1.pdf',format='pdf',bbox_inches='tight')