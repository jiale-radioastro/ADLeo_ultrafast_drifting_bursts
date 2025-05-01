import numpy as np
import matplotlib.pyplot as plt
from PFSS_funcs import *
            
# Magnetic field extrapolation with only the ZDI Map

dat_dir='/Users/jiale/Desktop/projects/PT2021_0019/0319/zdi/'
datfile=dat_dir+'outMagCoeff_ADLeo_2019b.dat'
datstr = np.genfromtxt(datfile,delimiter='\t',dtype=str)
alpha_lm_zdi=np.zeros([9,9],dtype=np.complex_)
for linestr in datstr[2:46]:
    tmp=linestr.split(' ')
    while '' in tmp:
        tmp.remove('')
    data=[float(num) for num in tmp]
    alpha_lm_zdi[int(data[0]),int(data[1])]=complex(data[2],data[3])
    
lat_list=[10,70]
for lat in lat_list:
    nlat=180*2
    nlon=360*2
    lon0=np.linspace(-np.pi, np.pi, nlon)
    sin_lat0=np.linspace(-1, 1, nlat)
    lat0=np.arcsin(sin_lat0)
    colat0=-lat0+np.pi/2
    _,sin_lat0=np.meshgrid(lon0,sin_lat0,indexing='ij')
    lon0, colat0=np.meshgrid(lon0,colat0,indexing='ij')

    sz=np.shape(lon0)
    br=np.zeros(sz)
    br,_,_=pfss_vec(np.ones(sz),colat0,lon0,alpha_lm_zdi,r_s=4)

    # latitudinal cross-section
    lon1=np.arange(-180,180,0.5)/180*np.pi
    colat1=np.ones(len(lon1))*(90-lat)/90*np.pi/2
    r1=np.arange(1.02,2.01,0.01)
    colat1,_=np.meshgrid(colat1,r1,indexing='ij')
    lon1,r1=np.meshgrid(lon1,r1,indexing='ij')

    sz1=lon1.shape
    babs1=np.ones(sz1)
    lb1=np.ones([sz1[0],sz1[1],3])
    br1,btheta1,bphi1=pfss_vec(r1,colat1,lon1,alpha_lm_zdi,r_s=4)
    babs1=np.sqrt(br1**2+btheta1**2+bphi1**2)
    lb1=pfss_lb(r1,colat1,lon1,alpha_lm_zdi,r_s=4)

    npz_name='bmap'+'_lat'+str(lat)+'_nospot'+'.npz'
    npz_file='/Users/jiale/Desktop/projects/PT2021_0019/0319/starspot/'+npz_name
    np.savez(npz_file,lon0=lon0,colat0=colat0,br=br,lon1=lon1,colat1=colat1,r1=r1,babs1=babs1,lb1=lb1)
    print(npz_name+' completed')

