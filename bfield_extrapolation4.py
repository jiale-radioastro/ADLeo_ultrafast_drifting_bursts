import numpy as np
from PFSS_funcs import *

file_dir='data/magnetic_field/'
suffixlist=['2019a','2019b','2020a','2020b']

for suffix in suffixlist:
    datfile=file_dir+'outMagCoeff_ADLeo_'+suffix+'.dat'
    datstr = np.genfromtxt(datfile,delimiter='\t',dtype=str)
    alpha_lm_zdi=np.zeros([9,9],dtype=np.complex_)
    for linestr in datstr[2:46]:
        tmp=linestr.split(' ')
        while '' in tmp:
            tmp.remove('')
        data=[float(num) for num in tmp]
        alpha_lm_zdi[int(data[0]),int(data[1])]=complex(data[2],data[3])

    nlat=180*2
    nlon=360*2

    lon0=np.linspace(-np.pi, np.pi, nlon)
    sin_lat0=np.linspace(-1, 1, nlat)
    lat0=np.arcsin(sin_lat0)
    colat0=-lat0+np.pi/2
    _,sin_lat0=np.meshgrid(lon0,sin_lat0,indexing='ij')
    lon0, colat0=np.meshgrid(lon0,colat0,indexing='ij')

    sz=np.shape(lon0)
    br0,_,_=pfss_vec(np.ones(sz),colat0,lon0,alpha_lm_zdi,r_s=4)
    br,bphi,btheta=pfss_vec(np.ones(sz)*1.05,colat0,lon0,alpha_lm_zdi,r_s=4)
    babs1=np.sqrt(br**2+btheta**2+bphi**2)
    lb1=pfss_lb(np.ones(sz)*1.05,colat0,lon0,alpha_lm_zdi,r_s=4)
    npz_name='bmap'+suffix+'_1.05'+'.npz'
    np.savez(file_dir+npz_name,lon0=lon0,colat0=colat0,br0=br0,babs1=babs1,lb1=lb1)

    br,bphi,btheta=pfss_vec(np.ones(sz)*1.1,colat0,lon0,alpha_lm_zdi,r_s=4)
    babs2=np.sqrt(br**2+btheta**2+bphi**2)
    lb2=pfss_lb(np.ones(sz)*1.1,colat0,lon0,alpha_lm_zdi,r_s=4)
    npz_name='bmap'+suffix+'_1.10'+'.npz'
    np.savez(file_dir+npz_name,lon0=lon0,colat0=colat0,br0=br0,babs1=babs2,lb1=lb2)