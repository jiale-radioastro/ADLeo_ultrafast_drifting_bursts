import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lpmv,lpmn
from tqdm import tqdm
import matplotlib

# coordinate transformation
def cart2sphe(x,y,z):
    # return r, theta, phi
    # theta is the angle with the z-axis (co-latitude)
    r=(x**2+y**2+z**2)**0.5
    theta=np.arctan2((x**2+y**2)**0.5,z)
    phi=np.arctan2(y,x)
    return (r,theta,phi)

def sphe2cart_vec(dr,dtheta,dphi,r,theta,phi):
    dx=dr*np.sin(theta)*np.cos(phi)+dtheta*np.cos(theta)*np.cos(phi)-dphi*np.sin(phi)
    dy=dr*np.sin(theta)*np.sin(phi)+dtheta*np.cos(theta)*np.sin(phi)+dphi*np.cos(phi)
    dz=dr*np.cos(theta)-dtheta*np.sin(theta)
    return (dx,dy,dz)

# calculate vector magnetic field based on pfss model
def pfss_vec(r,theta,phi,alpha_lm,r_s,lmax=0):
    # r in the unit of stellar radius, theta and phi in the unit of radian
    # r, theta, phi should be in the same shape
    # alpha_lm should include the factor of 2 when m>0
    # output in br,btheta and bphi
    ds=1e-7
    if lmax==0:
        lmax=np.shape(alpha_lm)[0]
    sz=np.shape(r)
    br=np.zeros(sz)
    btheta=np.zeros(sz)
    bphi=np.zeros(sz)
    for l in range(1,lmax):
        fl=(l+1+l*(r/r_s)**(2*l+1))/(l+1+l*(1/r_s)**(2*l+1))
        gl=(1-(r/r_s)**(2*l+1))/(l+1+l*(1/r_s)**(2*l+1))
        for m in range(0,l+1):
            c_lm=np.sqrt((2*l+1)/(4*np.pi)*np.math.factorial(l-m)/np.math.factorial(l+m))
            p_lm=lpmv(m,l,np.cos(theta))
            dp_lm=(lpmv(m,l,np.cos(theta+ds))-lpmv(m,l,np.cos(theta-ds)))/(2*ds)
            br+=np.real(c_lm*alpha_lm[l,m]*p_lm*fl*r**(-l-2)*np.exp(1j*m*phi))
            btheta+=np.real(-c_lm*alpha_lm[l,m]*dp_lm*gl*r**(-l-2)*np.exp(1j*m*phi))
            bphi+=np.real(-c_lm*alpha_lm[l,m]*p_lm/np.sin(theta)*1j*m*gl*r**(-l-2)*np.exp(1j*m*phi))
    return (br,btheta,bphi)

# calculate the magnetic scale height
def pfss_lb(r,theta,phi,alpha_lm,r_s,lmax=0):
    br,btheta,bphi=pfss_vec(r,theta,phi,alpha_lm,r_s,lmax)
    babs=np.sqrt(br**2+btheta**2+bphi**2)
    ds=1e-7
    dr=ds*br/babs
    dtheta=ds*btheta/babs/r
    dphi=ds*bphi/babs/(r*np.sin(theta))
    br1,btheta1,bphi1=pfss_vec(r+dr,theta+dtheta,phi+dphi,alpha_lm,r_s,lmax)
    babs1=np.sqrt(br1**2+btheta1**2+bphi1**2)
    br2,btheta2,bphi2=pfss_vec(r-dr,theta-dtheta,phi-dphi,alpha_lm,r_s,lmax)
    babs2=np.sqrt(br2**2+btheta2**2+bphi2**2)
    return np.abs(babs/((babs2-babs1)/(2*ds)))

# spherical harmonics decomposition
def br2alpha_lm(theta,phi,br,lmax=50):
    alpha_lm=np.zeros([lmax,lmax],dtype=np.complex_)
    do_lm=np.abs(np.diff(phi,axis=0)[:,1:]*np.diff(theta,axis=1)[1:,:])
    for l in tqdm(range(1,lmax)):
        for m in range(0,l+1):
            c_lm=np.sqrt((2*l+1)/(4*np.pi)*np.math.factorial(l-m)/np.math.factorial(l+m))
            p_lm=lpmv(m,l,np.cos(theta))
            int_lm=br*p_lm*np.exp(-1j*m*phi)*np.sin(theta)
            int_lm=(int_lm[:-1,:-1]+int_lm[1:,:-1]+int_lm[:-1,1:]+int_lm[1:,1:])/4
            ## see Vidotto 2016 for reason to do it
            if m!=0:
                int_lm*=2
            alpha_lm[l,m]=c_lm*np.sum(int_lm*do_lm)
    return alpha_lm

# construct sun(star)spot
def sunspot_br(s,phi,bmax,s0=0,phi0=0,gamma=0,rho=np.pi/6,a=0.56):
    b0=bmax*np.sqrt(2*np.exp(1))/a
    #rho=sep/(2*a)*np.sqrt(2)
    x=np.cos(phi)*np.sqrt(1-s**2)
    y=np.sin(phi)*np.sqrt(1-s**2)
    z=s
    sz=np.shape(x)
    matrix1=np.matrix([[1,0,0],\
                     [0,np.cos(gamma),-np.sin(gamma)],\
                     [0,np.sin(gamma),np.cos(gamma)]])
    matrix2=np.matrix([[np.sqrt(1-s0**2),0,s0],\
                      [0,1,0],\
                      [-s0,0,np.sqrt(1-s0**2)]])
    matrix3=np.matrix([[np.cos(phi0),np.sin(phi0),0],\
                      [-np.sin(phi0),np.cos(phi0),0],\
                      [0,0,1]])
    matrix=matrix1 @ matrix2 @ matrix3
    x1=np.ones(sz)
    y1=np.ones(sz)
    z1=np.ones(sz)
    for i in range(sz[0]):
        for j in range(sz[1]):
            tmp=matrix @ np.array([x[i,j],y[i,j],z[i,j]]).T
            x1[i,j]=tmp[0,0]
            y1[i,j]=tmp[0,1]
            z1[i,j]=tmp[0,2]
    phi1=np.angle(x1+y1*1j)
    s1=z1
    
    return -b0*phi1/rho*np.exp(-(phi1**2+2*(np.arcsin(s1))**2)/(a*rho)**2)

def find_neartime(time0,timelist):
    try:
        value0=time0.mjd
        valuelist=timelist.mjd
    except:
        value0=time0
        valuelist=timelist
    difflist=np.abs(valuelist-value0)
    return np.where(difflist==np.nanmin(difflist))[0][0]

# create customized colormaps
cmap = plt.cm.RdBu_r  # define the colormap
cmaplist = [cmap(i) for i in range(cmap.N)]
cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, cmap.N)
bounds1 = np.linspace(-900, 900, 16)
norm1 = matplotlib.colors.BoundaryNorm(bounds1, cmap1.N)
cmap = plt.cm.RdYlBu  # define the colormap
cmaplist = [cmap(i) for i in range(cmap.N)]
cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, cmap.N)
bounds2 = np.linspace(0.035, 0.485, 16)
norm2 = matplotlib.colors.BoundaryNorm(bounds2, cmap2.N)

# calculate magnetic field properties and save them
dat_dir='/Users/jiale/Desktop/projects/PT2021_0019/0319/zdi/'
suffixlist=['2019a','2019b','2020a','2020b']
for suffix in suffixlist:
    datfile=dat_dir+'outMagCoeff_ADLeo_'+suffix+'.dat'
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
    br1=np.zeros(sz)
    br1,_,_=pfss_vec(np.ones(sz),colat0,lon0,alpha_lm_zdi,r_s=4)
    br,bphi,btheta=pfss_vec(np.ones(sz)*1.05,colat0,lon0,alpha_lm_zdi,r_s=4)
    babs1=np.sqrt(br**2+btheta**2+bphi**2)
    lb1=pfss_lb(np.ones(sz)*1.05,colat0,lon0,alpha_lm_zdi,r_s=4)

    npz_name='bmap'+suffix+'.npz'
    npz_file='/Users/jiale/Desktop/projects/PT2021_0019/0319/zdi/'+npz_name
    np.savez(npz_file,lon0=lon0,colat0=colat0,br=br1,babs=babs1,lb=lb1)
    print(npz_name+' completed')

# visualization
fig=plt.figure(figsize=(10,6),dpi=400)
plt.rcParams.update({'font.size': 10})
ax0=fig.add_axes([0.07,0.6,0.2,0.32],projection='polar')
ax1=fig.add_axes([0.07+0.22,0.6,0.2,0.32],projection='polar')
ax2=fig.add_axes([0.07+0.44,0.6,0.2,0.32],projection='polar')
ax3=fig.add_axes([0.07+0.66,0.6,0.2,0.32],projection='polar')
ax4=fig.add_axes([0.07,0.23,0.2,0.32],projection='polar')
ax5=fig.add_axes([0.07+0.22,0.23,0.2,0.32],projection='polar')
ax6=fig.add_axes([0.07+0.44,0.23,0.2,0.32],projection='polar')
ax7=fig.add_axes([0.07+0.66,0.23,0.2,0.32],projection='polar')
ax8=fig.add_axes([0.09,0.05,0.16,0.15])
ax9=fig.add_axes([0.09+0.22,0.05,0.16,0.15])
ax10=fig.add_axes([0.09+0.44,0.05,0.16,0.15])
ax11=fig.add_axes([0.09+0.66,0.05,0.16,0.15])
ax12=fig.add_axes([0.96,0.6,0.01,0.32])
ax13=fig.add_axes([0.96,0.23,0.01,0.32])

axlist=[ax0,ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11]
colorlist=['darkgreen','darkblue','darkorange','darkmagenta']
for i in [0,1,2,3]:
    suffix=suffixlist[i]
    npz_name='bmap'+suffix+'.npz'
    npz_file='/Users/jiale/Desktop/projects/PT2021_0019/0319/zdi/'+npz_name
    npz_data=np.load(npz_file)
    lon0=npz_data['lon0']
    colat0=npz_data['colat0']
    br=npz_data['br']
    babs=npz_data['babs']
    lb=npz_data['lb']
    sz=np.shape(lon0)
    
    a0 = axlist[i].pcolor(lon0[:,:], colat0[:,:], br[:,:],cmap=cmap1,norm=norm1,rasterized=True)
    axlist[i].plot(lon0[:,0],np.ones(sz[0])*np.pi/6,'--',color='grey',linewidth=1,dashes=(10, 10))
    axlist[i].plot(lon0[:,0],np.ones(sz[0])*np.pi/3,'--',color='grey',linewidth=1,dashes=(10, 10))
    axlist[i].plot(lon0[:,0],np.ones(sz[0])*np.pi/2,color='grey',linewidth=1)
    axlist[i].set_theta_offset(-np.pi/2)
    axlist[i].set_theta_direction(-1)
    axlist[i].set_rticks([])
    axlist[i].set_xticks([np.pi/4*1,np.pi/4*3,np.pi/4*5,np.pi/4*7])
    axlist[i].set_ylim([0,120/180*np.pi])
    axlist[i].set_title(suffix)
    #axlist[4*i].text(-0.08,0.95,'a',fontsize=15,weight='bold',transform=ax0.transAxes)
    axlist[i].grid(False)
    if i==0:
        axlist[i].text(-0.2,0.05,'Radial magnetic field', rotation='vertical',fontsize=12,weight='bold',transform=axlist[i].transAxes)
        a01=plt.colorbar(a0,cax=ax12,label='$B_r$ [G]')
        a01.set_ticks([-720,-240,240,720])
    
    a1 = axlist[i+4].pcolor(lon0[:,:], colat0[:,:], lb[:,:],cmap=cmap2,norm=norm2,rasterized=True)
    axlist[i+4].plot(lon0[:,0],np.ones(sz[0])*np.pi/6,'--',color='grey',linewidth=1,dashes=(10, 10))
    axlist[i+4].plot(lon0[:,0],np.ones(sz[0])*np.pi/3,'--',color='grey',linewidth=1,dashes=(10, 10))
    axlist[i+4].plot(lon0[:,0],np.ones(sz[0])*np.pi/2,color='grey',linewidth=1)
    axlist[i+4].contour(lon0[:,:], colat0[:,:], babs[:,:],levels=[1000/2.8,1500/2.8],colors='black')
    axlist[i+4].contour(lon0[:,:], colat0[:,:], babs[:,:],levels=[1000/2.8/2,1500/2.8/2],colors='darkgreen')
    axlist[i+4].set_theta_offset(-np.pi/2)
    axlist[i+4].set_theta_direction(-1)
    axlist[i+4].set_rticks([])
    axlist[i+4].set_xticks([np.pi/4*1,np.pi/4*3,np.pi/4*5,np.pi/4*7])
    axlist[i+4].set_ylim([0,120/180*np.pi])
    #axlist[i+4].set_title('$L_B$ at $r=1.05\;r_0$')
    #axlist[i+4].text(-0.08,0.95,'b',fontsize=15,weight='bold',transform=ax1.transAxes)
    axlist[i+4].grid(False)
    if i==0:
        axlist[i+4].text(-0.2,0.25,'Scale height', rotation='vertical',fontsize=12,weight='bold',transform=axlist[i+4].transAxes)
        a11=plt.colorbar(a1,cax=ax13,label='$L_B$ $[r_\star]$')
        a11.set_ticks([0.08,0.2,0.32,0.44])
    
    axlist[i+8].fill_between([0,0.12],[1,1],[1e5,1e5],color='Grey',alpha=0.3,edgecolor=None)
    axlist[i+8].fill_between([0,0.07],[1,1],[1e5,1e5],color='Grey',alpha=0.7,edgecolor=None)
    index=find_neartime(colat0[0,:],np.pi/180*120)
    lb1list=lb[:,index:].flatten()
    babs1list=babs[:,index:].flatten()
    lb1vlist=[lb1list[i] for i in range(len(lb1list)) if \
              (babs1list[i]>1000/2.8 and babs1list[i]<1500/2.8) or (babs1list[i]>500/2.8 and babs1list[i]<750/2.8)]
    _,bins,_=axlist[i+8].hist(lb1list,bins=50,color=colorlist[i],alpha=0.3,edgecolor='black',range=[0,0.8],rasterized=True)
    _=axlist[i+8].hist(lb1vlist,color=colorlist[i],edgecolor='black',bins=bins,rasterized=True)
    axlist[i+8].spines['top'].set_visible(False)
    axlist[i+8].spines['right'].set_visible(False)
    axlist[i+8].set_yscale('log')
    axlist[i+8].set_xlim([0,0.8])
    axlist[i+8].set_ylim([10,1e3])
    axlist[i+8].set_yticks([1e2,1e5])
    axlist[i+8].set_xlabel('$L_B$ [$r_\star$]')
    axlist[i+8].set_xticks([0,0.2,0.4,0.6,0.8])
    if i==0:
        axlist[i+8].text(-0.35,0.1,'Statistics', rotation='vertical',fontsize=12,weight='bold',transform=axlist[i+8].transAxes)

fig1 = plt.gcf()
plt.show()
fig1.savefig('/users/jiale/desktop/supfigure3.pdf',format='pdf',bbox_inches='tight')
