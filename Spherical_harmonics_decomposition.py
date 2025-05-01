import numpy as np
from PFSS_funcs import *

# Spherical harmonic coefficients of the starspot models

npy_dir='/Users/jiale/Desktop/projects/PT2021_0019/0319/starspot/'

lat_list=[10,70]
rho_list=[30,15,8]
bmax_list=[3000,6000,9000]
for lat in lat_list:
    for rho in rho_list:
        for bmax in bmax_list:
            nlat=180*5
            nlon=360*5
            lon0=np.linspace(-np.pi, np.pi, nlon)
            sin_lat0=np.linspace(-1, 1, nlat)
            lat0=np.arcsin(sin_lat0)
            colat0=-lat0+np.pi/2
            _,sin_lat0=np.meshgrid(lon0,sin_lat0,indexing='ij')
            lon0, colat0=np.meshgrid(lon0,colat0,indexing='ij')

            sz=np.shape(lon0)
            br=np.zeros(sz)
            br+=sunspot_br(sin_lat0,lon0,bmax,s0=np.sin(lat/180*np.pi),phi0=0,gamma=0,rho=rho/180*np.pi,a=0.56)
            alpha_lm=br2alpha_lm(colat0,lon0,br,lmax=81)

            npy_name='alpha_lm'+'_lat'+str(lat)+'_rho'+str(rho)+'_bmax'+str(bmax)+'.npy'
            npy_file=npy_dir+npy_name
            np.save(npy_file,alpha_lm)
