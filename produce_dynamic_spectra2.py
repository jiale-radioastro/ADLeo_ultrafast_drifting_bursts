import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy.io import fits
from FAST_reduction_funcs import *

# The file directories
file_dir="/Users/jiale/Desktop/projects/PT2021_0019/0319/"
cal_dir=file_dir+"high/"
img_dir=file_dir+'publication_figures/'

# Read the file data
filei=39
filename=file_dir+"AD_leo_arctracking-M01_"+str(filei+1).zfill(4)+".fits"
file=fits.open(filename)
header = file[0].header
filedata = file[1].data
ttbin=32
[I_data2,Q_data2,U_data2,V_data2]=give_sp(filedata,ttbin)
filei=38
filename=file_dir+"AD_leo_arctracking-M01_"+str(filei+1).zfill(4)+".fits"
file=fits.open(filename)
header = file[0].header
filedata = file[1].data
[I_data1,Q_data1,U_data1,V_data1]=give_sp(filedata,ttbin)
I_data=np.concatenate((I_data1,I_data2),axis=0)
Q_data=np.concatenate((Q_data1,Q_data2),axis=0)
U_data=np.concatenate((U_data1,U_data2),axis=0)
V_data=np.concatenate((V_data1,V_data2),axis=0)

# Noise subtraction and polarization calibration
noise_T=give_noise_T(cal_dir+'T_noise_W_high_01a.dat',cal_dir+'T_noise_W_high_01b.dat',cal_dir+'freq.dat',filedata[0]['DAT_FREQ'])
szdata=np.shape(I_data)
noise_num=noise_count(file=filei,currlen=szdata[0]*ttbin)
I_spec=np.zeros(szdata[1])
Q_spec=np.zeros(szdata[1])
U_spec=np.zeros(szdata[1])
V_spec=np.zeros(szdata[1])
for noise_i in range(noise_num):
    [noise_on,noise_off]=noise_onoff(noise_i,ttbin=ttbin,file=filei)
    I_spec_tmp=np.mean(I_data[noise_on[0]:noise_on[1],:],axis=0)-np.mean(I_data[noise_off[0]:noise_off[1],:],axis=0)
    Q_spec_tmp=np.mean(Q_data[noise_on[0]:noise_on[1],:],axis=0)-np.mean(Q_data[noise_off[0]:noise_off[1],:],axis=0)
    U_spec_tmp=np.mean(U_data[noise_on[0]:noise_on[1],:],axis=0)-np.mean(U_data[noise_off[0]:noise_off[1],:],axis=0)
    V_spec_tmp=np.mean(V_data[noise_on[0]:noise_on[1],:],axis=0)-np.mean(V_data[noise_off[0]:noise_off[1],:],axis=0)
    I_spec+=I_spec_tmp
    Q_spec+=Q_spec_tmp
    U_spec+=U_spec_tmp
    V_spec+=V_spec_tmp
    I_data[noise_on[0]:noise_on[1],:]-=I_spec_tmp
    Q_data[noise_on[0]:noise_on[1],:]-=Q_spec_tmp
    U_data[noise_on[0]:noise_on[1],:]-=U_spec_tmp
    V_data[noise_on[0]:noise_on[1],:]-=V_spec_tmp
[I_spec,Q_spec,U_spec,V_spec]=[I_spec/noise_num,Q_spec/noise_num,U_spec/noise_num,V_spec/noise_num]
[gamma,phi,au2t]=diff_cal(I_spec,Q_spec,U_spec,V_spec,noise_T=noise_T)
I_mask=give_mask_lc(I_spec-medfilt(I_spec,4))
I_spec1=fix_nan_lc(I_spec*I_mask)
Q_mask=give_mask_lc(Q_spec-medfilt(Q_spec,4))
Q_spec1=fix_nan_lc(Q_spec*Q_mask)
U_mask=give_mask_lc(U_spec-medfilt(U_spec,4))
U_spec1=fix_nan_lc(U_spec*U_mask)
V_mask=give_mask_lc(V_spec-medfilt(V_spec,4))
V_spec1=fix_nan_lc(V_spec*V_mask)
[gamma,phi,au2t]=diff_cal(I_spec1,Q_spec1,U_spec1,V_spec1,noise_T=noise_T)
[I_data,Q_data,U_data,V_data]=obs2true(I_data,Q_data,U_data,V_data,gamma,phi,au2t)

# Bandpass calibration
G=np.load(file_dir+'G_cal.npy')
I_data=np.multiply(1/G,I_data)*1e3
Q_data=np.multiply(1/G,Q_data)*1e3
U_data=np.multiply(1/G,U_data)*1e3
V_data=np.multiply(1/G,V_data)*1e3

# RFI masking and background subtraction
ffbin=1
ttbin2=1
timelist=give_time(tbin=file[1].header['TBIN'],ttbin=ttbin,ttbin2=ttbin2,length=1024*1024,file=0/16384)-22.99
freqlist=give_freq(ffbin,filedata[0]['DAT_FREQ'])
V_mask0=give_mask(V_data,method='rfi',thresh=3)
index0=find_neartime(timelist,0)
index1=find_neartime(timelist,33)
V_mask1=give_mask(V_data,method='time',time=np.arange(index0,index1,1))
I_data_bkgd1=trend_remove(I_data*V_mask0*V_mask1,method='fit',index=2,output='bkgd')
V_data_bkgd1=trend_remove(V_data*V_mask0*V_mask1,method='fit',index=2,output='bkgd')
I_data_cut=I_data-I_data_bkgd1
V_data_cut=V_data-V_data_bkgd1
plot_mask=give_mask(I_data_cut,method='channel',thresh=2)

# Produce Fig. S1
plt.figure(num=1,figsize=(10,8),dpi=400)
plt.rcParams.update({'font.size': 10})
plt.subplot(3,1,1)
cmap=plt.get_cmap('viridis')
cmap.set_bad('lightgrey',alpha=0.5)
a=plt.imshow((I_data_cut*plot_mask).T,aspect='auto',vmin=0,vmax=120,origin='lower',\
                 extent=[timelist[0],timelist[-1],1000,1500],cmap=cmap)
plt.xlim([0,33])
plt.colorbar(a,label='Stokes I: flux density [mJy]',aspect=40,pad=0.02)
plt.ylabel('Frequency [MHz]')
plt.text(-3.6,1480,'A',fontsize=13,color='black',weight='bold')
plt.subplot(3,1,2)
cmap=plt.get_cmap('viridis_r')
cmap.set_bad('lightgrey',alpha=0.5)
a=plt.imshow((V_data_cut*plot_mask).T,aspect='auto',vmin=-120,vmax=0,origin='lower',\
                 extent=[timelist[0],timelist[-1],1000,1500],cmap=cmap)
plt.xlim([0,33])
plt.colorbar(a,label='Stokes V: flux density [mJy]',aspect=40,pad=0.02)
plt.ylabel('Frequency [MHz]')
plt.text(-3.6,1480,'B',fontsize=13,color='black',weight='bold')
plt.subplot(3,1,3)
cmap=plt.get_cmap('bwr_r')
cmap.set_bad('lightgrey',alpha=0.5)
a=plt.imshow(((V_data_cut/I_data_cut)*plot_mask*100).T,aspect='auto',vmin=-140,vmax=-60,origin='lower',\
                 extent=[timelist[0],timelist[-1],1000,1500],cmap=cmap)
plt.xlim([0,33])
plt.colorbar(a,label='Polarization degree [%]',aspect=40,pad=0.02)
plt.xlabel('Time [s] since 2022-03-19 15:05:40 UT')
plt.ylabel('Frequency [MHz]')
plt.text(-3.6,1480,'C',fontsize=13,color='black',weight='bold')
fig1 = plt.gcf()
plt.show()
fig1.savefig(img_dir+'supfigure1.pdf',format='pdf',bbox_inches='tight')
