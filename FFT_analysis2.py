import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy.io import fits
from FAST_reduction_funcs import *
from scipy.interpolate import interp2d
from scipy.optimize import curve_fit
from scipy.fft import fft,fft2,fftshift,fftfreq,ifft,ifft2
from tqdm import tqdm

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
# Original data could be requested from https://fast.bao.ac.cn/
# Noise calibration data could be found on https://fast.bao.ac.cn/cms/category/telescope_performence_en/noise_diode_calibration_report_en/
file_dir="/Users/jiale/Desktop/projects/PT2021_0019/0319/"
cal_dir=file_dir+"high/"
sav_file='data/FAST/'
img_dir='publication_figures/'

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
I_mask=give_mask_lc(I_spec-medfilt(I_spec,4))
I_spec1=fix_nan_lc(I_spec*I_mask)
Q_mask=give_mask_lc(Q_spec-medfilt(Q_spec,4))
Q_spec1=fix_nan_lc(Q_spec*Q_mask)
U_mask=give_mask_lc(U_spec-medfilt(U_spec,4))
U_spec1=fix_nan_lc(U_spec*U_mask)
V_mask=give_mask_lc(V_spec-medfilt(V_spec,4))
V_spec1=fix_nan_lc(V_spec*V_mask)
[gamma,phi,au2t]=diff_cal(I_spec1,Q_spec1,U_spec1,V_spec1,noise_T=noise_T)

# Bandpass calibration
# Using Calibrator 3C286
G=np.load(file_dir+'G_cal.npy')

ffbin=1
ttbin2=1
timelist=give_time(tbin=file[1].header['TBIN'],ttbin=ttbin,ttbin2=ttbin2,length=1024*1024,file=0/16384)-22.99 # Time in s starting from 2022-03-19 15:05:40 UT
freqlist=give_freq(ffbin,filedata[0]['DAT_FREQ'])

driftlist=[]
errlist=[]
tlist=[]
threshlist=[50,30,50,25,50,30,20,8,2.5,4,10,30,40,40,40,40,40,40,40,40,40,40,40,40]
for i in tqdm(range(0,23)):
    index_range=[159+5*i,164+5*i]
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
    raw_I=np.multiply(1/G,raw_I)*1e3
    raw_V=np.multiply(1/G,raw_V)*1e3
    ffbin=1
    ttbin2=2
    sub_rawI=frebin(trebin(raw_I,ttbin2),ffbin=ffbin)
    sub_rawV=frebin(trebin(raw_V,ttbin2),ffbin=ffbin)
    sub_rawI=trend_remove(sub_rawI,average=10,method='fit',index=2)
    sub_rawV=trend_remove(sub_rawV,average=10,method='fit',index=2)
    V_mask2=give_mask(sub_rawI,method='channel',thresh=threshlist[i])
    sub_fixV=fix_nan(sub_rawV*V_mask2,mode='freq')
    
    fft_img=fftshift(fft2(sub_fixV))
    sz=np.shape(fft_img)
    f_time=fftshift(fftfreq(sz[0],d=0.000196608*ttbin2))
    f_freq=fftshift(fftfreq(sz[1],d=freqlist[1]-freqlist[0]))
    
    klist=np.arange(0.01,0.15,0.002)
    vlist=[]
    for k in klist:
        vlist.append(radon_trans(np.abs(fft_img)**2,k,400))
    vlist=np.array(normalize(vlist))
    slope=klist*(f_freq[1]-f_freq[0])/(f_time[1]-f_time[0])*1e3
    index=np.where(vlist>0.5)
    popt, pcov = curve_fit(gaussian, slope[index], vlist[index], p0=[0.5,0.1,0.03])
    mu_fit = popt[1]
    mu_err = popt[2]
    driftlist.append(1/mu_fit)
    errlist.append(1/mu_fit**2*mu_err)
    tlist.append((timelist[index_range[0]*1024//ttbin]+timelist[index_range[1]*1024//ttbin])/2)
driftlist=np.array(driftlist)
errlist=np.array(errlist)
tlist=np.array(tlist)

npz_name='drift_rate.npz'
np.savez(sav_file+npz_name,driftlist=driftlist,errlist=errlist,tlist=tlist)

# Produce Fig. S2
fig=plt.figure(figsize=(3.55,2.5),dpi=400)
plt.rcParams.update({'font.size': 8})
plt.errorbar(tlist,driftlist,yerr=errlist,fmt='o',color='blue',linewidth=1,markersize=3)
plt.fill_between([tlist[6]-0.5,tlist[6]+0.5],[0,0],[26,26],alpha=0.2,color='black',edgecolor=None)
plt.text(tlist[6]-0.4,1.5,'B',fontsize=9)
plt.fill_between([tlist[9]-0.5,tlist[9]+0.5],[0,0],[26,26],alpha=0.2,color='black',edgecolor=None)
plt.text(tlist[9]-0.4,1.5,'C',fontsize=9)
plt.fill_between([tlist[17]-0.5,tlist[17]+0.5],[0,0],[26,26],alpha=0.2,color='black',edgecolor=None)
plt.text(tlist[17]-0.4,1.5,'D',fontsize=9)
plt.ylabel('Drift rate [GHz$\cdot$s$^{-1}$]')
plt.ylim([0,20])
plt.xlabel('Time [s] since 2022-03-19 15:05:40 UT')
fig.savefig(img_dir+'sfigure2.pdf',format='pdf',bbox_inches='tight')