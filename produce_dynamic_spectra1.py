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

# Fine structure dynamic spectra
index_range=[189,194]
for index in range(index_range[0],index_range[1]):
    raw_I1=np.int16(filedata['data'][index,:,0,:,0])+np.int16(filedata['data'][index,:,1,:,0])
    raw_Q1=np.int16(filedata['data'][index,:,0,:,0])-np.int16(filedata['data'][index,:,1,:,0])
    raw_U1=np.int16(filedata['data'][index,:,2,:,0])*2
    raw_V1=np.int16(filedata['data'][index,:,3,:,0])*2 
    if index==index_range[0]:
        raw_I=raw_I1
        raw_Q=raw_Q1
        raw_U=raw_U1
        raw_V=raw_V1
    else:
        raw_I=np.vstack((raw_I,raw_I1))
        raw_Q=np.vstack((raw_Q,raw_Q1))
        raw_U=np.vstack((raw_U,raw_U1))
        raw_V=np.vstack((raw_V,raw_V1))
[raw_I,raw_Q,raw_U,raw_V]=obs2true(raw_I,raw_Q,raw_U,raw_V,gamma,phi,au2t)
G=np.load(file_dir+'G_cal.npy')
raw_I=np.multiply(1/G,raw_I)*1e3
raw_V=np.multiply(1/G,raw_V)*1e3
ffbin=1
ttbin2=2
sub_rawI=trebin(raw_I,ttbin2)
sub_rawI=frebin(sub_rawI,ffbin=ffbin)
sub_rawV=trebin(raw_V,ttbin2)
sub_rawV=frebin(sub_rawV,ffbin=ffbin)
sub_rawI=trend_remove(sub_rawI,average=10,method='fit',index=2)
sub_rawV=trend_remove(sub_rawV,average=10,method='fit',index=2)
I_mask2=give_mask(sub_rawI,method='channel',thresh=20)
sub_rawV1=sub_rawV
sub_mask1=I_mask2

index_range=[204,209]
for index in range(index_range[0],index_range[1]):
    raw_I1=np.int16(filedata['data'][index,:,0,:,0])+np.int16(filedata['data'][index,:,1,:,0])
    raw_Q1=np.int16(filedata['data'][index,:,0,:,0])-np.int16(filedata['data'][index,:,1,:,0])
    raw_U1=np.int16(filedata['data'][index,:,2,:,0])*2
    raw_V1=np.int16(filedata['data'][index,:,3,:,0])*2
    if index==index_range[0]:
        raw_I=raw_I1
        raw_Q=raw_Q1
        raw_U=raw_U1
        raw_V=raw_V1
    else:
        raw_I=np.vstack((raw_I,raw_I1))
        raw_Q=np.vstack((raw_Q,raw_Q1))
        raw_U=np.vstack((raw_U,raw_U1))
        raw_V=np.vstack((raw_V,raw_V1))
[raw_I,raw_Q,raw_U,raw_V]=obs2true(raw_I,raw_Q,raw_U,raw_V,gamma,phi,au2t)
G=np.load(file_dir+'G_cal.npy')
raw_I=np.multiply(1/G,raw_I)*1e3
raw_V=np.multiply(1/G,raw_V)*1e3
ffbin=1
ttbin2=2
sub_rawI=trebin(raw_I,ttbin2)
sub_rawI=frebin(sub_rawI,ffbin=ffbin)
sub_rawV=trebin(raw_V,ttbin2)
sub_rawV=frebin(sub_rawV,ffbin=ffbin)
sub_rawI=trend_remove(sub_rawI,average=10,method='fit',index=2)
sub_rawV=trend_remove(sub_rawV,average=10,method='fit',index=2)
I_mask2=give_mask(sub_rawI,method='channel',thresh=4)
sub_rawV2=sub_rawV
sub_mask2=I_mask2

index_range=[244,249]
for index in range(index_range[0],index_range[1]):
    raw_I1=np.int16(filedata['data'][index,:,0,:,0])+np.int16(filedata['data'][index,:,1,:,0])
    raw_Q1=np.int16(filedata['data'][index,:,0,:,0])-np.int16(filedata['data'][index,:,1,:,0])
    raw_U1=np.int16(filedata['data'][index,:,2,:,0])*2
    raw_V1=np.int16(filedata['data'][index,:,3,:,0])*2
    if index==index_range[0]:
        raw_I=raw_I1
        raw_Q=raw_Q1
        raw_U=raw_U1
        raw_V=raw_V1
    else:
        raw_I=np.vstack((raw_I,raw_I1))
        raw_Q=np.vstack((raw_Q,raw_Q1))
        raw_U=np.vstack((raw_U,raw_U1))
        raw_V=np.vstack((raw_V,raw_V1))
[raw_I,raw_Q,raw_U,raw_V]=obs2true(raw_I,raw_Q,raw_U,raw_V,gamma,phi,au2t)
G=np.load(file_dir+'G_cal.npy')
raw_I=np.multiply(1/G,raw_I)*1e3
raw_V=np.multiply(1/G,raw_V)*1e3
ffbin=1
ttbin2=2
sub_rawI=trebin(raw_I,ttbin2)
sub_rawI=frebin(sub_rawI,ffbin=ffbin)
sub_rawV=trebin(raw_V,ttbin2)
sub_rawV=frebin(sub_rawV,ffbin=ffbin)
sub_rawI=trend_remove(sub_rawI,average=10,method='fit',index=2)
sub_rawV=trend_remove(sub_rawV,average=10,method='fit',index=2)
I_mask2=give_mask(sub_rawI,method='channel',thresh=40)
sub_rawV3=sub_rawV
sub_mask3=I_mask2

# Produce Fig. 1
fig=plt.figure(figsize=(10,8),dpi=400)
plt.rcParams.update({'font.size': 10})
cmap=plt.get_cmap('viridis_r')
cmap.set_bad('lightgrey',alpha=0.5)

ax0=fig.add_axes([0.1,0.7,0.85,0.2])
a0=ax0.imshow((V_data_cut*plot_mask).T,aspect='auto',vmin=-120,vmax=0,origin='lower',\
                 extent=[timelist[0],timelist[-1],1000,1500],cmap=cmap)
ax0.set_xlim([0,33])
ax0.set_ylabel('Frequency [MHz]')
color="tomato"
ax0.plot([timelist[189*1024//ttbin],timelist[189*1024//ttbin],\
          timelist[194*1024//ttbin],timelist[194*1024//ttbin],timelist[189*1024//ttbin]],\
        [1500,1000,1000,1500,1500],'--',color=color,clip_on=False,linewidth=2)
ax0.text(0.5*(timelist[189*1024//ttbin]+timelist[194*1024//ttbin])-0.3,1020,'B',fontsize=13,weight='bold',color=color)
ax0.plot([timelist[204*1024//ttbin],timelist[204*1024//ttbin],\
          timelist[209*1024//ttbin],timelist[209*1024//ttbin],timelist[204*1024//ttbin]],\
        [1500,1000,1000,1500,1500],'--',color=color,clip_on=False,linewidth=2)
ax0.text(0.5*(timelist[204*1024//ttbin]+timelist[209*1024//ttbin])-0.3,1020,'C',fontsize=13,weight='bold',color=color)
ax0.plot([timelist[244*1024//ttbin],timelist[244*1024//ttbin],\
          timelist[249*1024//ttbin],timelist[249*1024//ttbin],timelist[244*1024//ttbin]],\
        [1500,1000,1000,1500,1500],'--',color=color,clip_on=False,linewidth=2)
ax0.text(0.5*(timelist[244*1024//ttbin]+timelist[249*1024//ttbin])-0.3,1020,'D',fontsize=13,weight='bold',color=color)
ax0.text(15,1530,'Stokes V radio emission flux density [mJy]',clip_on=False,fontsize=10)
ax0.text(-2.7,1480,'A',clip_on=False,fontsize=13,weight='bold')
ax00=fig.add_axes([0.8,0.91,0.15,0.015])
a00=plt.colorbar(a0,cax=ax00,orientation='horizontal')
a00.ax.invert_xaxis()
ax00.xaxis.set_ticks_position('top')
ax1=fig.add_axes([0.1,0.45,0.85,0.18])
a1=ax1.imshow((sub_rawV1*sub_mask1).T,vmin=-120,vmax=0,origin='lower',aspect='auto',\
             extent=[timelist[189*1024//ttbin],timelist[194*1024//ttbin],1000,1500],cmap=cmap)
ax1.set_ylim([freqlist[0],freqlist[-1]])
ax1.set_ylabel('Frequency [MHz]')
ax1.text(timelist[189*1024//ttbin]-2.7/33,1485,'B',clip_on=False,fontsize=13,weight='bold')
ax1.plot([timelist[189*1024//ttbin]+0.05,timelist[189*1024//ttbin]+0.05+300/(11.5*1e3)],[1400,1100],\
         '--',color=color,linewidth=2.3)
ax1.text(timelist[189*1024//ttbin]+0.05,1050,'11.5 GHz$\cdot$s$^{-1}$',color=color,fontsize=12)
ax2=fig.add_axes([0.1,0.23,0.85,0.18])
a2=ax2.imshow((sub_rawV2*sub_mask2).T,vmin=-120,vmax=0,origin='lower',aspect='auto',\
             extent=[timelist[204*1024//ttbin],timelist[209*1024//ttbin],1000,1500],cmap=cmap)
ax2.set_ylim([freqlist[0],freqlist[-1]])
ax2.set_ylabel('Frequency [MHz]')
ax2.text(timelist[204*1024//ttbin]-2.7/33,1485,'C',clip_on=False,fontsize=13,weight='bold')
ax2.plot([timelist[204*1024//ttbin]+0.05,timelist[204*1024//ttbin]+0.05+300/(8.7*1e3)],[1400,1100],\
         '--',color=color,linewidth=2.3)
ax2.text(timelist[204*1024//ttbin]+0.05,1050,'8.7 GHz$\cdot$s$^{-1}$',color=color,fontsize=12)
ax3=fig.add_axes([0.1,0.01,0.85,0.18])
a3=ax3.imshow((sub_rawV3*sub_mask3).T,vmin=-120,vmax=0,origin='lower',aspect='auto',\
             extent=[timelist[244*1024//ttbin],timelist[249*1024//ttbin],1000,1500],cmap=cmap)
ax3.set_ylim([freqlist[0],freqlist[-1]])
ax3.set_ylabel('Frequency [MHz]')
ax3.set_xlabel('Time [s] since 2022-03-19 15:05:40 UT')
ax3.text(timelist[244*1024//ttbin]-2.7/33,1485,'D',clip_on=False,fontsize=13,weight='bold')
ax3.plot([timelist[244*1024//ttbin]+0.05,timelist[244*1024//ttbin]+0.05+300/(7.5*1e3)],[1400,1100],\
         '--',color=color,linewidth=2.3)
ax3.text(timelist[244*1024//ttbin]+0.05,1050,'7.5 GHz$\cdot$s$^{-1}$',color=color,fontsize=12)
fig1 = plt.gcf()
plt.show()
fig1.savefig(img_dir+'figure1.pdf',format='pdf',bbox_inches='tight')






