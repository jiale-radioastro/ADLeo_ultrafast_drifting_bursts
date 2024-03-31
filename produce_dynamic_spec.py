from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
from tqdm import tqdm
from scipy.ndimage import median_filter as medfilt
from scipy.stats import median_abs_deviation

################################################subtract noise diode signal#####################################################
def noise_onoff(num,ttbin=1,file=0,filelen=512*1024,tperiod=80*1024,tnoise=5*1024):  
    # default: noise diode signals are periodically injected. e.g. on-1s off-15s on-1s off-15s ......   
    # To determine the index of 'num'th cycle of noise 'on' and noise 'off'
    # noise 'on': noise_on[0] - noise_on[1]   noise 'off': noise_off[0] - noise_off[1]
    # num: the sequence number of the cycle in the fits file. cycle starts with the noise injection
    # ttbin: first round of time rebin. default 1
    # file: the sequence number of the fits file. default 0
    # filelen: the length of a single file in time domain. default 512*1024
    # tperiod: the length of the cycle period. default 80*1024
    # tnoise: the length of the noise injection time. default 5*1024
    # the default value of tinterval and tnoise accords to the noise injection mode: 1s noise every 16s, time cadence 0.2ms
    tperiod=tperiod//ttbin
    tnoise=tnoise//ttbin
    prelen=filelen//ttbin*file
    prenum=int(np.ceil(prelen/tperiod))              # the number of previous cycles
    num=num+prenum                    # the real sequence number of the cycle
    noise_on=[num*tperiod-prelen,num*tperiod+tnoise-prelen]
    noise_off=[num*tperiod+tnoise-prelen,num*tperiod+2*tnoise-prelen]
    return noise_on,noise_off

def noise_count(file=0,currlen=512*1024,filelen=512*1024,tperiod=80*1024,tnoise=5*1024):
    # count the number of noise diode signals in a file
    prelen=filelen*file
    n_start=np.ceil(prelen/tperiod)
    n_end=np.ceil((prelen+currlen)/tperiod)
    return int(n_end-n_start)

################################################polarization and flux calibration#############################################
def obs2true(I,Q,U,V,gamma=0,phi=0,au2t=1):
    # calibrate the observed flux in four Stokes parameters
    # reference:  Britton (2000), van Straten (2004), Liao et al. (2016)
    I2=I*np.cosh(2*gamma)-Q*np.sinh(2*gamma)
    Q2=-I*np.sinh(2*gamma)+Q*np.cosh(2*gamma)
    U2=U*np.cos(2*phi)-V*np.sin(2*phi)
    V2=U*np.sin(2*phi)+V*np.cos(2*phi)
    return I2*au2t,Q2*au2t,U2*au2t,V2*au2t

def diff_cal(I,Q,U,V,noise_T=12.5*np.ones(1024)):
    # determine the differential gain and phase of the orthogonal feeds using the noise diode signals
    # calculate /gamma and /phi value in Mueller matrix
    # reference:  Britton (2000), van Straten (2004), Liao et al. (2016)
    gamma=np.arctanh(Q/I)/2
    phi=np.angle(U-V*complex(0,1))/2
    I2=I*np.cosh(2*gamma)-Q*np.sinh(2*gamma)
    return gamma,phi,noise_T/I2

def read_dat(datfile):
    # read data array from .dat file
    datstr = np.genfromtxt(datfile,delimiter='\t',dtype=str)
    datdata=[]
    for linestr in datstr:
        tmp=linestr.split(' ')
        while '' in tmp:
            tmp.remove('')
        for num in tmp:
            datdata.append(float(num))
     
    return datdata

def give_noise_T(adat,bdat,freqdat,freq):
    # derive the noise temperature from the .dat file
    # .dat files can be downloaded from the noise diode calibration report
    # (https://fast.bao.ac.cn/cms/category/telescope_performance/noise_diode_calibration_report/)
    # T_noise=np.sqrt(T_A*T_B)
    datstr = np.genfromtxt(adat,delimiter='\t',dtype=str)
    adatdata=read_dat(adat)
    bdatdata=read_dat(bdat)
    freqdatdata=read_dat(freqdat)
    raw_T=np.sqrt(np.array(adatdata)*np.array(bdatdata))
    raw_freq=np.array(freqdatdata)
    return np.interp(freq,raw_freq,raw_T)

################################################data concat and reduction####################################################
def concatdata(data,pol):
    # concatenate data in the given polarization
    # input data dimension time time pol freq 0   512 1024 4 1024 1
    # output data dimension time freq  512*1024 1024
    # notice that the raw data is in the format of uint8, better change to int
    # polarization channels are XX YY CR CI, CR=Re{XY*}, CI=Im{XY*}
    # Stokes parameters are defined as I=XX+YY, Q=XX-YY, U=CR*2, V=CI*2
    # Circular polarization convention from van Straten et al.(2010) positive for LCP
    if pol=="XX":
        return np.int16(np.concatenate(data['data'][:,:,0,:,0]))
    elif pol=="YY":
        return np.int16(np.concatenate(data['data'][:,:,1,:,0]))
    elif pol=="CR":
        return np.int16(np.concatenate(data['data'][:,:,2,:,0]))
    elif pol=="CI":
        return np.int16(np.concatenate(data['data'][:,:,3,:,0]))
    elif pol=="I":
        return np.int16(np.concatenate(data['data'][:,:,0,:,0]))+np.int16(np.concatenate(data['data'][:,:,1,:,0]))
    elif pol=="Q":
        return np.int16(np.concatenate(data['data'][:,:,0,:,0]))-np.int16(np.concatenate(data['data'][:,:,1,:,0]))
    elif pol=="U":
        return np.int16(np.concatenate(data['data'][:,:,2,:,0]))*2
    elif pol=="V":
        return np.int16(np.concatenate(data['data'][:,:,3,:,0]))*2
    
def give_sp(filedata,ttbin):
    # obtain four Stokes parameters from filedata
    I_data=trebin(concatdata(filedata,"I"),ttbin)
    Q_data=trebin(concatdata(filedata,"Q"),ttbin)
    U_data=trebin(concatdata(filedata,"U"),ttbin)
    V_data=trebin(concatdata(filedata,"V"),ttbin)
    return [I_data, Q_data, U_data, V_data]
    
def trebin(data,ttbin):
    # rebin data in time dimension
    # data dimension time freq
    if ttbin==1:
        return data
    else:
        sz=np.shape(data)
        if len(sz)>1:
            data2=np.zeros([sz[0]//ttbin,sz[1]])
            for i in range(sz[0]//ttbin):
                data2[i,:]=np.nanmean(data[ttbin*i:ttbin*(i+1),:],axis=0)
            return data2
        else:
            data2=np.zeros([sz[0]//ttbin])
            for i in range(sz[0]//ttbin):
                data2[i]=np.nanmean(data[ttbin*i:ttbin*(i+1)])
            return data2

def frebin(data,ffbin):
    # rebin data in freq dimension
    # data dimension time freq
    if ffbin==1:
        return data
    else:
        sz=np.shape(data)
        data2=np.zeros([sz[0],sz[1]//ffbin])
        for i in range(sz[1]//ffbin):
            data2[:,i]=np.nanmean(data[:,ffbin*i:ffbin*(i+1)],axis=1)
        return data2

################################################give timelist and freqlist######################################################
def give_time(tbin=0.000196608,ttbin=1,ttbin2=1,length=512*1024,prelength=512*1024,file=0):
    # give time list in s
    # tbin: time interval in s
    # ttbin: first round of time rebin
    # ttbin2: second round of time rebin
    # length: the length of the current file in time domain
    # prelength: the length of the previous files in time domain
    num=length//ttbin//ttbin2
    pretime=prelength*file*tbin
    return pretime+tbin*ttbin*ttbin2*np.linspace(0,num-1,num)

def give_freq(ffbin,freqlist=np.linspace(1000,1000+1023*0.48828125,1024),length=1024):
    # give frequency list in MHz
    # ffbin: frequency rebin
    # length: the length of the current file in frequency domain
    # freqlist: the raw frequency list. can be derived from 'filedata[0]['DAT_FREQ']'
    num=length//ffbin
    return np.array([np.mean(freqlist[ffbin*i:ffbin*(i+1)]) for i in range(num)])

def find_neartime(time0,timelist):
    try:
        value0=time0.mjd
        valuelist=timelist.mjd
    except:
        value0=time0
        valuelist=timelist
    difflist=np.abs(valuelist-value0)
    return np.where(difflist==np.nanmin(difflist))[0][0]

#################################################background subtraction#####################################################
def trend_remove(dspec,method='fit',index=1,average=1,kernel=1001,output='residual'):
    # remove the long-term background flux variation
    # three methods available
    # 'fit' method: use polynomial fitting to determine the background. 
    # index: the index of the polynomial fitting
    # 'medfilt' method: use median filter to determine the background
    # 'kernel': the median filter time window
    # 'startend' method: draw a line between the start and end points to estimate the background
    # average: the averaging length
    sz=np.shape(dspec)
    dspec_residual=np.zeros([sz[0],sz[1]])
    dspec_bkgd=np.zeros([sz[0],sz[1]])
    for freqi in tqdm(range(sz[1])):
        lc=dspec[:,freqi]
        if method == 'startend':
            y0=np.mean(lc[range(average)])
            x0=0.5*(0+average-1)
            y1=np.mean(lc[range(sz[0]-average,sz[0])])
            x1=0.5*(sz[0]-average+sz[0]-1)
            k=(y1-y0)/(x1-x0)
            dspec_residual[:,freqi]=lc-k*(np.arange(sz[0])-x0)-y0
            dspec_bkgd[:,freqi]=k*(np.arange(sz[0])-x0)+y0
        elif method == 'fit':
            idx = np.isfinite(lc)
            if len(lc[idx])>index*2:
                kb=np.polyfit(np.arange(sz[0])[idx],lc[idx],index)
                f1=np.poly1d(kb)
                dspec_residual[:,freqi]=lc-f1(np.arange(sz[0]))
                dspec_bkgd[:,freqi]=f1(np.arange(sz[0]))
            else:
                dspec_residual[:,freqi]=lc
                dspec_bkgd[:,freqi]=lc*0
        elif method == 'medfilt':
            dspec_bkgd[:,freqi]=medfilt(lc,kernel)
            dspec_residual[:,freqi]=lc-dspec_bkgd[:,freqi]
    if output=='residual':
        return dspec_residual
    if output=='bkgd':
        return dspec_bkgd
    if output=='both':
        return [dspec_residual,dspec_bkgd]

#####################################################data flagging#####################################################
def fix_nan_lc(lc,time=[]):
    lc=np.array(lc)
    if np.isnan(lc).any():
        if len(time)==0:
            time=np.arange(0,len(lc),1)
        time0=time[~np.isnan(lc)]
        lc0=lc[~np.isnan(lc)]
        if len(lc0)<2:
            return np.ones(len(lc))
        return np.interp(time,time0,lc0)
    else:
        return lc

def fix_nan(dspec,mode='time'):
    sz=np.shape(dspec)
    dspec_new=dspec[:,:]
    if mode=='time':
        for freqi in range(sz[1]):
            dspec_new[:,freqi]=fix_nan_lc(dspec[:,freqi])
    if mode=='freq':
        for timei in range(sz[0]):
            dspec_new[timei,:]=fix_nan_lc(dspec[timei,:])
    return dspec_new

def give_std_lc(lc,kernel=5):
    std_list=[]
    for i in range(len(lc)):
        index0=np.max([0,i-kernel])
        index1=np.min([i+kernel,len(lc)])
        std_list.append(np.nanstd(lc[index0:index1]))
    return np.array(std_list)

def give_mask_lc(lc,kernel=5,thresh=5):
    std_list=give_std_lc(lc,kernel=kernel)
    MAD=median_abs_deviation(std_list,nan_policy='omit')
    median=np.nanmedian(std_list)
    mask=np.ones(len(lc))
    mask[std_list>median+thresh*MAD]=np.nan
    return mask

def give_mask(dspec,kernel=5,thresh=5,method='rfi',time=[],freq=[]):
    #  five modes made available now
    # "rfi": flag random short-duration RFI signals
    # "burst":  flag strong radio burst region
    # "channel":   flag channels that are corrupted by RFIs
    # "time":   flag specific time indexes
    # "freq":   flag specific frequency indexes
    sz=np.shape(dspec)
    mask=np.ones(sz)
    if method=='rfi':
        for freqi in tqdm(range(sz[1])):
            mask_lc=give_mask_lc(dspec[:,freqi],kernel=kernel,thresh=thresh)
            mask[:,freqi]=mask_lc
    if method=='burst':
        index=np.where(dspec>thresh)
        mask[index]=np.nan
    if method=='channel':
        std_list=[]
        for i in range(sz[1]):
            std_list.append(np.nanstd(dspec[:,i]))
        std_list=np.array(std_list)
        median=np.nanmedian(std_list)
        MAD=median_abs_deviation(std_list,nan_policy='omit')
        for i in range(sz[1]):
            if std_list[i]>median+thresh*MAD:
                mask[:,i]=np.nan*mask[:,i]
    if method=='time':
        for timeri in time:
            mask[timeri,:]=np.nan*mask[timeri,:]
    if method=='freq':
        for freqi in freq:
            mask[:,freqi]=np.nan*mask[:,freqi]
    return mask

file_dir="/Users/jiale/Desktop/projects/PT2021_0019/0319/"

# read observation data
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
cal_dir=file_dir+"high/"
noise_T=give_noise_T(cal_dir+'T_noise_W_high_01a.dat',cal_dir+'T_noise_W_high_01b.dat',cal_dir+'freq.dat',filedata[0]['DAT_FREQ'])

# do calibrations
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
[I_data,Q_data,U_data,V_data]=obs2true(I_data,Q_data,U_data,V_data,gamma,phi,au2t)
G=np.load(file_dir+'G_cal.npy')
I_data=np.multiply(1/G,I_data)*1e3
Q_data=np.multiply(1/G,Q_data)*1e3
U_data=np.multiply(1/G,U_data)*1e3
V_data=np.multiply(1/G,V_data)*1e3

# flag data and subtract background
ffbin=1
ttbin2=1
timelist=give_time(tbin=file[1].header['TBIN'],ttbin=ttbin,ttbin2=ttbin2,length=1024*1024,file=0/16384)-22.99
freqlist=give_freq(ffbin,filedata[0]['DAT_FREQ'])

index0=find_neartime(timelist,0)
index1=find_neartime(timelist,33)
V_mask0=give_mask(V_data,method='rfi',thresh=3)
V_mask1=give_mask(V_data,method='time',time=np.arange(index0,index1,1))
I_data_bkgd1=trend_remove(I_data*V_mask0*V_mask1,method='fit',index=2,output='bkgd')
V_data_bkgd1=trend_remove(V_data*V_mask0*V_mask1,method='fit',index=2,output='bkgd')
I_data_cut=I_data-I_data_bkgd1
V_data_cut=V_data-V_data_bkgd1
plot_mask=give_mask(I_data_cut,method='channel',thresh=2)

# produce and save dynamic spectra
plt.figure(figsize=(10,8),dpi=400)
plt.rcParams.update({'font.size': 10})
plt.subplot(3,1,1)
a=plt.imshow((I_data_cut*plot_mask).T,aspect='auto',vmin=0,vmax=120,origin='lower',\
                 extent=[timelist[0],timelist[-1],1000,1500],cmap='coolwarm')
plt.xlim([0,33])
plt.colorbar(a,label='Stokes I: flux density [mJy]',aspect=40,pad=0.02)
plt.ylabel('Frequency [MHz]')
plt.text(-3.4,1480,'a',fontsize=15,color='black',weight='bold')
plt.subplot(3,1,2)
a=plt.imshow((V_data_cut*plot_mask).T,aspect='auto',vmin=0,vmax=120,origin='lower',\
                 extent=[timelist[0],timelist[-1],1000,1500],cmap='coolwarm')
plt.xlim([0,33])
plt.colorbar(a,label='Stokes V: flux density [mJy]',aspect=40,pad=0.02)
plt.ylabel('Frequency [MHz]')
plt.text(-3.4,1480,'b',fontsize=15,color='black',weight='bold')
plt.subplot(3,1,3)
a=plt.imshow(((V_data_cut/I_data_cut)*plot_mask*100).T,aspect='auto',vmin=60,vmax=140,origin='lower',\
                 extent=[timelist[0],timelist[-1],1000,1500],cmap='bwr')
plt.xlim([0,33])
plt.colorbar(a,label='Polarization degree [%]',aspect=40,pad=0.02)
plt.xlabel('Time [s] since 2022-03-19 15:05:40 UT')
plt.ylabel('Frequency [MHz]')
plt.text(-3.4,1480,'c',fontsize=15,color='black',weight='bold')

fig1 = plt.gcf()
plt.show()
fig1.savefig('/users/jiale/desktop/supfigure1.pdf',format='pdf',bbox_inches='tight')
