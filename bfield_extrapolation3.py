import numpy as np
from PFSS_funcs import *

file_dir='data/magnetic_field/'
datfile=file_dir+'outMagCoeff_ADLeo_2019b.dat'
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
npz_name='bmap'+'_r'+str(1.05)+'_nospot'+'.npz'
np.savez(file_dir+npz_name,lon0=lon0,colat0=colat0,br0=br0,babs1=babs1,lb1=lb1)

br,bphi,btheta=pfss_vec(np.ones(sz)*1.1,colat0,lon0,alpha_lm_zdi,r_s=4)
babs2=np.sqrt(br**2+btheta**2+bphi**2)
lb2=pfss_lb(np.ones(sz)*1.1,colat0,lon0,alpha_lm_zdi,r_s=4)
npz_name='bmap'+'_r'+str(1.10)+'_nospot'+'.npz'
np.savez(file_dir+npz_name,lon0=lon0,colat0=colat0,br0=br0,babs1=babs2,lb1=lb2)

lat=10
rho=8
bmax=6000

npy_name='alpha_lm'+'_lat'+str(lat)+'_rho'+str(rho)+'_bmax'+str(bmax)+'.npy'
npy_file=file_dir+npy_name
alpha_lm=np.load(npy_file)
alpha_lm_hybrid=alpha_lm.copy()
alpha_lm_hybrid[:9,:9]+=alpha_lm_zdi

nlat=180*2
nlon=360*2

lon0=np.linspace(-np.pi, np.pi, nlon)
sin_lat0=np.linspace(-1, 1, nlat)
lat0=np.arcsin(sin_lat0)
colat0=-lat0+np.pi/2
_,sin_lat0=np.meshgrid(lon0,sin_lat0,indexing='ij')
lon0, colat0=np.meshgrid(lon0,colat0,indexing='ij')

sz=np.shape(lon0)
br0,_,_=pfss_vec(np.ones(sz),colat0,lon0,alpha_lm_hybrid,r_s=4)
br,bphi,btheta=pfss_vec(np.ones(sz)*1.05,colat0,lon0,alpha_lm_hybrid,r_s=4)
babs1=np.sqrt(br**2+btheta**2+bphi**2)
lb1=pfss_lb(np.ones(sz)*1.05,colat0,lon0,alpha_lm_hybrid,r_s=4)
npz_name='bmap'+'_r'+str(1.05)+'_rho'+str(rho)+'_bmax'+str(bmax)+'.npz'
np.savez(file_dir+npz_name,lon0=lon0,colat0=colat0,br0=br0,babs1=babs1,lb1=lb1)

br,bphi,btheta=pfss_vec(np.ones(sz)*1.1,colat0,lon0,alpha_lm_hybrid,r_s=4)
babs2=np.sqrt(br**2+btheta**2+bphi**2)
lb2=pfss_lb(np.ones(sz)*1.1,colat0,lon0,alpha_lm_hybrid,r_s=4)
npz_name='bmap'+'_r'+str(1.10)+'_rho'+str(rho)+'_bmax'+str(bmax)+'.npz'
np.savez(file_dir+npz_name,lon0=lon0,colat0=colat0,br0=br0,babs1=babs2,lb1=lb2)