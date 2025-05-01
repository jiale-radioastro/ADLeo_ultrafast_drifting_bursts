import numpy as np
from tqdm import tqdm
from scipy.ndimage import median_filter as medfilt
from scipy.stats import median_abs_deviation

################################################subtract noise diode signal#####################################################
def noise_onoff(num,ttbin=1,file=0,filelen=512*1024,tperiod=80*1024,tnoise=5*1024):
    # To determine the index of 'num'th cycle of noise 'on' and noise 'off'
    # default: noise diode signals are periodically injected. e.g. on-1s off-15s on-1s off-15s ...... 
    # default: sampling time: 0.2 ms, more accurately, 196.608 us
    # noise 'on': noise_on[0] - noise_on[1]   noise 'off': noise_off[0] - noise_off[1]
    # num: the sequence number of the cycle in the fits file. cycle starts with the noise injection
    # ttbin: first round of time rebin. default 1
    # file: the sequence number of the fits file, remember FAST file names start from 1. default 0
    # filelen: the length of a single file in time domain, related to first two dim lens. default 512*1024
    # tperiod: the length of the cycle period, noise period/sampling time. default 80*1024
    # tnoise: the length of the noise injection time, injection time/sampling time. default 5*1024
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
    # currlen: the length of the current file in time domain
    # filelen: the length of the previous files in the time domain
    # others: refer to noise_onoff()
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
    phi=np.arctan2(-V,U)/2
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
    # polarization channels are XX YY CR CI, CR=Re{X*Y}, CI=Im{X*Y} 
    # reference: https://iopscience.iop.org/article/10.1088/1674-4527/acea1f/meta
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
    # data 2-d: time freq
    # data 1-d: time
    # suggest chosing the multiple of 2
    if ttbin==1:
        return data
    else:
        sz=np.shape(data)
        if len(sz)==2:
            data2=np.zeros([sz[0]//ttbin,sz[1]])
            for i in range(sz[0]//ttbin):
                data2[i,:]=np.nanmean(data[ttbin*i:ttbin*(i+1),:],axis=0)
            return data2
        elif len(sz)==1:
            data2=np.zeros([sz[0]//ttbin])
            for i in range(sz[0]//ttbin):
                data2[i]=np.nanmean(data[ttbin*i:ttbin*(i+1)])
            return data2
        
def frebin(data,ffbin):
    # rebin data in freq dimension
    # data 2-d: time freq
    # data 1-d: freq
    # suggest chosing the multiple of 2
    if ffbin==1:
        return data
    else:
        sz=np.shape(data)
        if len(sz)==2:
            data2=np.zeros([sz[0],sz[1]//ffbin])
            for i in range(sz[1]//ffbin):
                data2[:,i]=np.nanmean(data[:,ffbin*i:ffbin*(i+1)],axis=1)
            return data2
        elif len(sz)==1:
            data2=np.zeros([sz[0]//ffbin])
            for i in range(sz[0]//ffbin):
                data2[i]=np.nanmean(data[ffbin*i:ffbin*(i+1)])
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

def give_freq(ffbin,freqlist=np.linspace(1000,1000+1023*0.48828125,1024)):
    # give frequency list in MHz
    # ffbin: frequency rebin
    # freqlist: the raw frequency list. can be derived from 'filedata[0]['DAT_FREQ']'
    return frebin(freqlist,ffbin)

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
    for freqi in range(sz[1]):
        lc=dspec[:,freqi]
        if method == 'startend':
            y0=np.mean(lc[0:average])
            x0=0.5*(0+average-1)
            y1=np.mean(lc[sz[0]-average,sz[0]])
            x1=0.5*(sz[0]-average+sz[0]-1)
            kb=np.polyfit([x0,x1],[y0,y1],1)
            f1=np.poly1d(kb)
            dspec_residual[:,freqi]=lc-f1(np.arange(sz[0]))
            dspec_bkgd[:,freqi]=f1(np.arange(sz[0]))
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
    
def find_neartime(time0,timelist):
    try:
        value0=time0.mjd
        valuelist=timelist.mjd
    except:
        value0=time0
        valuelist=timelist
    difflist=np.abs(valuelist-value0)
    return np.where(difflist==np.nanmin(difflist))[0][0]

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
    # mask certain regions in the dynamic spectra
    # 5 methods available now
    # rfi: mask sporadic rfi in the data, can be time-consuming
    # burst: mask strong radio bursts
    # channel: mask channels corrupted by rfis
    # time: mask specfic time indexes
    # freq: mask specfic freq indexes
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
