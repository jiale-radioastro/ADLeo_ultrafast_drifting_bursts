import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lpmv,lpmn
from tqdm import tqdm
import matplotlib

def pfss_vec(r,theta,phi,alpha_lm,r_s,lmax=0):
    # r: the radial distance, in the unit of stellar radius
    # theta: colatitude, phi: longitude, both in the unit of radian
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

def br2alpha_lm(theta,phi,br,lmax=50):
    alpha_lm=np.zeros([lmax,lmax],dtype=np.complex_)
    do_lm=np.abs(np.diff(phi,axis=0)[:,1:]*np.diff(theta,axis=1)[1:,:])
    for l in tqdm(range(1,lmax)):
        for m in range(0,l+1):
            c_lm=np.sqrt((2*l+1)/(4*np.pi)*np.math.factorial(l-m)/np.math.factorial(l+m))
            p_lm=lpmv(m,l,np.cos(theta))
            int_lm=br*p_lm*np.exp(-1j*m*phi)*np.sin(theta)
            int_lm=(int_lm[:-1,:-1]+int_lm[1:,:-1]+int_lm[:-1,1:]+int_lm[1:,1:])/4
            ## see Vidotto 2016 for reasons to add the factor
            if m!=0:
                int_lm*=2
            alpha_lm[l,m]=c_lm*np.sum(int_lm*do_lm)
    return alpha_lm

def sunspot_br(s,phi,bmax,s0=0,phi0=0,gamma=0,rho=np.pi/6,a=0.56):
    b0=bmax*np.sqrt(2*np.exp(1))/a
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

# Produce Fig. 2 
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
br,bphi,btheta=pfss_vec(np.ones(sz)*1.1,colat0,lon0,alpha_lm_zdi,r_s=4)
babs2=np.sqrt(br**2+btheta**2+bphi**2)
lb2=pfss_lb(np.ones(sz)*1.1,colat0,lon0,alpha_lm_zdi,r_s=4)

lat=10
rho=8
bmax=6000

npy_name='alpha_lm'+'_lat'+str(lat)+'_rho'+str(rho)+'_bmax'+str(bmax)+'.npy'
npy_file='/Users/jiale/Desktop/projects/PT2021_0019/0319/starspot/'+npy_name
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
br2=np.zeros(sz)
br2,_,_=pfss_vec(np.ones(sz),colat0,lon0,alpha_lm_hybrid,r_s=4)
br,bphi,btheta=pfss_vec(np.ones(sz)*1.05,colat0,lon0,alpha_lm_hybrid,r_s=4)
babs3=np.sqrt(br**2+btheta**2+bphi**2)
lb3=pfss_lb(np.ones(sz)*1.05,colat0,lon0,alpha_lm_hybrid,r_s=4)
br,bphi,btheta=pfss_vec(np.ones(sz)*1.1,colat0,lon0,alpha_lm_hybrid,r_s=4)
babs4=np.sqrt(br**2+btheta**2+bphi**2)
lb4=pfss_lb(np.ones(sz)*1.1,colat0,lon0,alpha_lm_hybrid,r_s=4)

# Create the colormaps
cmap = plt.cm.RdBu_r
cmaplist = [cmap(i) for i in range(cmap.N)]
cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, cmap.N)
nbins=40
maxbin=np.log(5)
minbin=np.log(0.5)
debin=(maxbin-minbin)/(nbins+0.5)
bar1=-1*np.exp(np.linspace(maxbin,minbin+debin,nbins))
nbins=80
maxbin=0.5
minbin=-0.5
debin=(maxbin-minbin)/(nbins+1)
bar2=np.linspace(minbin+debin,maxbin-debin,nbins)
nbins=40
maxbin=np.log(5)
minbin=np.log(0.5)
debin=(maxbin-minbin)/(nbins+0.5)
bar3=np.exp(np.linspace(minbin+debin,maxbin,nbins))
bounds1=np.concatenate((bar1,bar2,bar3))
norm1 = matplotlib.colors.BoundaryNorm(bounds1, cmap1.N)

cmap = plt.cm.RdYlBu 
cmaplist = [cmap(i) for i in range(cmap.N)]
cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, cmap.N)
bounds2 = np.linspace(0, 0.5, 256)
norm2 = matplotlib.colors.BoundaryNorm(bounds2, cmap2.N)

fig=plt.figure(figsize=(10,7),dpi=400)
plt.rcParams.update({'font.size': 10})

ax0=fig.add_axes([0.16,0.71,0.25,0.29],projection='polar')
ax1=fig.add_axes([0.42,0.71,0.25,0.29],projection='polar')
ax2=fig.add_axes([0.68,0.71,0.25,0.29],projection='polar')
ax3=fig.add_axes([0.47,0.58,0.2,0.1])
ax4=fig.add_axes([0.73,0.58,0.2,0.1])
ax5=fig.add_axes([0.16,0.14,0.25,0.29],projection='polar')
ax6=fig.add_axes([0.42,0.14,0.25,0.29],projection='polar')
ax7=fig.add_axes([0.68,0.14,0.25,0.29],projection='polar')
ax8=fig.add_axes([0.47,0.01,0.2,0.1])
ax9=fig.add_axes([0.73,0.01,0.2,0.1])

ax51=fig.add_axes([0.32,0.145,0.11,0.075])
ax61=fig.add_axes([0.58,0.145,0.11,0.075])
ax71=fig.add_axes([0.84,0.145,0.11,0.075])

ax00=fig.add_axes([0.2,0.68,0.16,0.01])
ax01=fig.add_axes([0.2,0.6,0.16,0.01])


a0 = ax0.pcolor(lon0[:,:], colat0[:,:], br1[:,:]/1e3,cmap=cmap1,norm=norm1,rasterized=True)
ax0.plot(lon0[:,0],np.ones(sz[0])*np.pi/6,'--',color='grey',linewidth=1,dashes=(10, 10))
ax0.plot(lon0[:,0],np.ones(sz[0])*np.pi/3,'--',color='grey',linewidth=1,dashes=(10, 10))
ax0.plot(lon0[:,0],np.ones(sz[0])*np.pi/2,color='grey',linewidth=1)
ax0.set_theta_offset(-np.pi/2)
ax0.set_theta_direction(-1)
ax0.set_rticks([])
ax0.set_xticks([np.pi/4*1,np.pi/4*3,np.pi/4*5,np.pi/4*7])
ax0.set_ylim([0,120/180*np.pi])
ax0.set_title('Surface radial magnetic field')
ax0.text(-0.2,-0.15,'ZDI map', rotation='vertical',fontsize=15,weight='bold',transform=ax0.transAxes)
ax0.text(-0.08,0.95,'A',fontsize=13,weight='bold',transform=ax0.transAxes)
ax0.grid(False)


a1 = ax1.pcolor(lon0[:,:], colat0[:,:], lb1[:,:],cmap=cmap2,norm=norm2,rasterized=True)
ax1.plot(lon0[:,0],np.ones(sz[0])*np.pi/6,'--',color='grey',linewidth=1,dashes=(10, 10))
ax1.plot(lon0[:,0],np.ones(sz[0])*np.pi/3,'--',color='grey',linewidth=1,dashes=(10, 10))
ax1.plot(lon0[:,0],np.ones(sz[0])*np.pi/2,color='grey',linewidth=1)
ax1.contour(lon0[:,:], colat0[:,:], babs1[:,:],levels=[1000/2.8,1500/2.8],colors='black')
ax1.contour(lon0[:,:], colat0[:,:], babs1[:,:],levels=[1000/2.8/2,1500/2.8/2],colors='darkgreen')
ax1.set_theta_offset(-np.pi/2)
ax1.set_theta_direction(-1)
ax1.set_rticks([])
ax1.set_xticks([np.pi/4*1,np.pi/4*3,np.pi/4*5,np.pi/4*7])
ax1.set_ylim([0,120/180*np.pi])
ax1.set_title('$L_B$ at $r=1.05\;r_\star$')
ax1.text(-0.08,0.95,'B',fontsize=13,weight='bold',transform=ax1.transAxes)
ax1.grid(False)

a00=plt.colorbar(a0,cax=ax00,orientation='horizontal',label='Magnetic field strength [kG]')
a00.set_ticks([-5,-0.5,0,0.5,5])
a01=plt.colorbar(a1,cax=ax01,orientation='horizontal',label='Magnetic scale height [$r_\star$]')
a01.set_ticks([0,0.1,0.2,0.3,0.4,0.5])


a2 = ax2.pcolor(lon0[:,:], colat0[:,:], lb2[:,:],cmap=cmap2,norm=norm2,rasterized=True)
ax2.plot(lon0[:,0],np.ones(sz[0])*np.pi/6,'--',color='grey',linewidth=1,dashes=(10, 10))
ax2.plot(lon0[:,0],np.ones(sz[0])*np.pi/3,'--',color='grey',linewidth=1,dashes=(10, 10))
ax2.plot(lon0[:,0],np.ones(sz[0])*np.pi/2,color='grey',linewidth=1)
ax2.contour(lon0[:,:], colat0[:,:], babs2[:,:],levels=[1000/2.8,1500/2.8],colors='black')
ax2.contour(lon0[:,:], colat0[:,:], babs2[:,:],levels=[1000/2.8/2,1500/2.8/2],colors='darkgreen')
ax2.set_theta_offset(-np.pi/2)
ax2.set_theta_direction(-1)
ax2.set_rticks([])
ax2.set_xticks([np.pi/4*1,np.pi/4*3,np.pi/4*5,np.pi/4*7])
ax2.set_ylim([0,120/180*np.pi])
ax2.set_title('$L_B$ at $r=1.1\;r_\star$')
ax2.text(-0.08,0.95,'C',fontsize=13,weight='bold',transform=ax2.transAxes)
ax2.grid(False)


ax3.fill_between([0,0.15],[1,1],[1e5,1e5],color='Grey',alpha=0.3,edgecolor=None)
ax3.fill_between([0,0.08],[1,1],[1e5,1e5],color='Grey',alpha=0.7,edgecolor=None)
index=find_neartime(colat0[0,:],np.pi/180*120)
lb1list=lb1[:,index:].flatten()
babs1list=babs1[:,index:].flatten()
lb1vlist=[lb1list[i] for i in range(len(lb1list)) if \
          (babs1list[i]>1000/2.8 and babs1list[i]<1500/2.8) or (babs1list[i]>500/2.8 and babs1list[i]<750/2.8)]
_,bins,_=ax3.hist(lb1list,bins=50,color='darkblue',alpha=0.3,edgecolor='black',range=[0,0.8],rasterized=True)
_=ax3.hist(lb1vlist,color='darkblue',edgecolor='black',bins=bins,rasterized=True)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.set_yscale('log')
ax3.set_xlim([0,0.8])
ax3.set_ylim([10,1e3])
ax3.set_yticks([1e2,1e5])
ax3.set_xlabel('$L_B$ [$r_\star$]')
ax3.set_ylabel('Counts')
ax3.text(-0.28,0.95,'D',fontsize=13,weight='bold',transform=ax3.transAxes)


ax4.fill_between([0,0.15],[1,1],[1e5,1e5],color='Grey',alpha=0.3,edgecolor=None)
ax4.fill_between([0,0.08],[1,1],[1e5,1e5],color='Grey',alpha=0.7,edgecolor=None)
index=find_neartime(colat0[0,:],np.pi/180*120)
lb2list=lb2[:,index:].flatten()
babs2list=babs2[:,index:].flatten()
lb2vlist=[lb2list[i] for i in range(len(lb2list)) if \
          (babs2list[i]>1000/2.8 and babs2list[i]<1500/2.8) or (babs2list[i]>500/2.8 and babs2list[i]<750/2.8)]
_,bins,_=ax4.hist(lb2list,bins=50,color='darkblue',alpha=0.3,edgecolor='black',range=[0,0.8],rasterized=True)
_=ax4.hist(lb2vlist,color='darkblue',edgecolor='black',bins=bins,rasterized=True)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.set_yscale('log')
ax4.set_xlim([0,0.8])
ax4.set_ylim([10,1e3])
ax4.set_yticks([1e2,1e5])
ax4.set_xlabel('$L_B$ [$r_\star$]')
ax4.text(-0.28,0.95,'E',fontsize=13,weight='bold',transform=ax4.transAxes)



a5 = ax5.pcolor(lon0[:,:], colat0[:,:], br2[:,:]/1e3,cmap=cmap1,norm=norm1,rasterized=True)
ax5.plot(lon0[:,0],np.ones(sz[0])*np.pi/6,'--',color='grey',linewidth=1,dashes=(10, 10))
ax5.plot(lon0[:,0],np.ones(sz[0])*np.pi/3,'--',color='grey',linewidth=1,dashes=(10, 10))
ax5.plot(lon0[:,0],np.ones(sz[0])*np.pi/2,color='grey',linewidth=1)
ax5.plot(lon0[300:419,0],np.ones(119)*np.pi*(95/180),'--',color='grey',linewidth=1)
ax5.plot(lon0[300:419,0],np.ones(119)*np.pi*(80/180),'--',color='grey',linewidth=1)
ax5.plot(lon0[300:419,0],np.ones(119)*np.pi*(65/180),'--',color='grey',linewidth=1)
ax5.plot([lon0[300,0],lon0[300,0]],[np.pi*(65/180),np.pi*(95/180)],'--',color='grey',linewidth=1)
ax5.plot([lon0[419,0],lon0[419,0]],[np.pi*(65/180),np.pi*(95/180)],'--',color='grey',linewidth=1)
ax5.set_theta_offset(-np.pi/2)
ax5.set_theta_direction(-1)
ax5.set_rticks([])
ax5.set_xticks([np.pi/4*1,np.pi/4*3,np.pi/4*5,np.pi/4*7])
ax5.set_ylim([0,120/180*np.pi])
ax5.set_title('Surface radial magnetic field')
ax5.text(-0.2,-0.35,'ZDI map + starspots', rotation='vertical',fontsize=15,weight='bold',transform=ax5.transAxes)
ax5.grid(False)
ax5.text(-0.08,0.95,'F',fontsize=13,weight='bold',transform=ax5.transAxes)

ax51.spines['left'].set_linewidth=0.1
ax51.spines['right'].set_linewidth=0.1
ax51.spines['bottom'].set_linewidth=0.1
ax51.spines['top'].set_linewidth=0.1
ax51.pcolor(lon0[:,:], colat0[:,:], br2[:,:]/1e3,cmap=cmap1,norm=norm1,rasterized=True)
ax51.plot([30/180*np.pi,-30/180*np.pi],[np.pi*(80/180),np.pi*(80/180)],'--',color='grey',linewidth=1)
ax51.set_xlim([30/180*np.pi,-30/180*np.pi])
ax51.set_ylim(95/180*np.pi,65/180*np.pi)
ax51.set_xticks([])
ax51.set_yticks([])


a6 = ax6.pcolor(lon0[:,:], colat0[:,:], lb3[:,:],cmap=cmap2,norm=norm2,rasterized=True)
ax6.plot(lon0[:,0],np.ones(sz[0])*np.pi/6,'--',color='grey',linewidth=1,dashes=(10, 10))
ax6.plot(lon0[:,0],np.ones(sz[0])*np.pi/3,'--',color='grey',linewidth=1,dashes=(10, 10))
ax6.plot(lon0[300:419,0],np.ones(119)*np.pi*(95/180),'--',color='grey',linewidth=1)
ax6.plot(lon0[300:419,0],np.ones(119)*np.pi*(80/180),'--',color='grey',linewidth=1)
ax6.plot(lon0[300:419,0],np.ones(119)*np.pi*(65/180),'--',color='grey',linewidth=1)
ax6.plot([lon0[300,0],lon0[300,0]],[np.pi*(65/180),np.pi*(95/180)],'--',color='grey',linewidth=1)
ax6.plot([lon0[419,0],lon0[419,0]],[np.pi*(65/180),np.pi*(95/180)],'--',color='grey',linewidth=1)
ax6.plot(lon0[:,0],np.ones(sz[0])*np.pi/2,color='grey',linewidth=1)
ax6.contour(lon0[:,:], colat0[:,:], babs3[:,:],levels=[1000/2.8,1500/2.8],colors='black')
ax6.contour(lon0[:,:], colat0[:,:], babs3[:,:],levels=[1000/2.8/2,1500/2.8/2],colors='darkgreen')
ax6.set_theta_offset(-np.pi/2)
ax6.set_theta_direction(-1)
ax6.set_rticks([])
ax6.set_xticks([np.pi/4*1,np.pi/4*3,np.pi/4*5,np.pi/4*7])
ax6.set_ylim([0,120/180*np.pi])
ax6.set_title('$L_B$ at $r=1.05\;r_\star$')
ax6.grid(False)
ax6.text(-0.08,0.95,'G',fontsize=13,weight='bold',transform=ax6.transAxes)

ax61.spines['left'].set_linewidth=0.1
ax61.spines['right'].set_linewidth=0.1
ax61.spines['bottom'].set_linewidth=0.1
ax61.spines['top'].set_linewidth=0.1
ax61.pcolor(lon0[:,:], colat0[:,:], lb3[:,:],cmap=cmap2,norm=norm2,rasterized=True)
ax61.plot([30/180*np.pi,-30/180*np.pi],[np.pi*(80/180),np.pi*(80/180)],'--',color='grey',linewidth=1)
ax61.contour(lon0[:,:], colat0[:,:], babs3[:,:],levels=[1000/2.8,1500/2.8],colors='black')
ax61.contour(lon0[:,:], colat0[:,:], babs3[:,:],levels=[1000/2.8/2,1500/2.8/2],colors='darkgreen')
ax61.set_xlim([30/180*np.pi,-30/180*np.pi])
ax61.set_ylim(95/180*np.pi,65/180*np.pi)
ax61.set_xticks([])
ax61.set_yticks([])

a7 = ax7.pcolor(lon0[:,:], colat0[:,:], lb4[:,:],cmap=cmap2,norm=norm2,rasterized=True)
ax7.plot(lon0[:,0],np.ones(sz[0])*np.pi/6,'--',color='grey',linewidth=1,dashes=(10, 10))
ax7.plot(lon0[:,0],np.ones(sz[0])*np.pi/3,'--',color='grey',linewidth=1,dashes=(10, 10))
ax7.plot(lon0[:,0],np.ones(sz[0])*np.pi/2,color='grey',linewidth=1)
ax7.plot(lon0[300:419,0],np.ones(119)*np.pi*(95/180),'--',color='grey',linewidth=1)
ax7.plot(lon0[300:419,0],np.ones(119)*np.pi*(80/180),'--',color='grey',linewidth=1)
ax7.plot(lon0[300:419,0],np.ones(119)*np.pi*(65/180),'--',color='grey',linewidth=1)
ax7.plot([lon0[300,0],lon0[300,0]],[np.pi*(65/180),np.pi*(95/180)],'--',color='grey',linewidth=1)
ax7.plot([lon0[419,0],lon0[419,0]],[np.pi*(65/180),np.pi*(95/180)],'--',color='grey',linewidth=1)
ax7.contour(lon0[:,:], colat0[:,:], babs4[:,:],levels=[1000/2.8,1500/2.8],colors='black')
ax7.contour(lon0[:,:], colat0[:,:], babs4[:,:],levels=[1000/2.8/2,1500/2.8/2],colors='darkgreen')
ax7.set_theta_offset(-np.pi/2)
ax7.set_theta_direction(-1)
ax7.set_rticks([])
ax7.set_xticks([np.pi/4*1,np.pi/4*3,np.pi/4*5,np.pi/4*7])
ax7.set_ylim([0,120/180*np.pi])
ax7.set_title('$L_B$ at $r=1.1\;r_\star$')
ax7.grid(False)
ax7.text(-0.08,0.95,'H',fontsize=13,weight='bold',transform=ax7.transAxes)

ax71.spines['left'].set_linewidth=0.1
ax71.spines['right'].set_linewidth=0.1
ax71.spines['bottom'].set_linewidth=0.1
ax71.spines['top'].set_linewidth=0.1
ax71.pcolor(lon0[:,:], colat0[:,:], lb4[:,:],cmap=cmap2,norm=norm2,rasterized=True)
ax71.plot([30/180*np.pi,-30/180*np.pi],[np.pi*(80/180),np.pi*(80/180)],'--',color='grey',linewidth=1)
ax71.contour(lon0[:,:], colat0[:,:], babs4[:,:],levels=[1000/2.8,1500/2.8],colors='black')
ax71.contour(lon0[:,:], colat0[:,:], babs4[:,:],levels=[1000/2.8/2,1500/2.8/2],colors='darkgreen')
ax71.set_xlim([30/180*np.pi,-30/180*np.pi])
ax71.set_ylim(95/180*np.pi,65/180*np.pi)
ax71.set_xticks([])
ax71.set_yticks([])


ax8.fill_between([0,0.15],[1,1],[1e5,1e5],color='Grey',alpha=0.3,edgecolor=None)
ax8.fill_between([0,0.08],[1,1],[1e5,1e5],color='Grey',alpha=0.7,edgecolor=None)
index=find_neartime(colat0[0,:],np.pi/180*120)
lb3list=lb3[:,index:].flatten()
babs3list=babs3[:,index:].flatten()
lb3vlist=[lb3list[i] for i in range(len(lb3list)) if \
          (babs3list[i]>1000/2.8 and babs3list[i]<1500/2.8) or (babs3list[i]>500/2.8 and babs3list[i]<750/2.8)]
_,bins,_=ax8.hist(lb3list,bins=50,color='darkorange',alpha=0.3,edgecolor='black',range=[0,0.8],rasterized=True)
_=ax8.hist(lb3vlist,color='darkorange',edgecolor='black',bins=bins,rasterized=True)
ax8.spines['top'].set_visible(False)
ax8.spines['right'].set_visible(False)
ax8.set_yscale('log')
ax8.set_xlim([0,0.8])
ax8.set_ylim([10,1e3])
ax8.set_yticks([1e2,1e5])
ax8.set_xlabel('$L_B$ [$r_\star$]')
ax8.set_ylabel('Counts')
ax8.text(-0.28,0.95,'I',fontsize=13,weight='bold',transform=ax8.transAxes)


ax9.fill_between([0,0.15],[1,1],[1e5,1e5],color='Grey',alpha=0.3,edgecolor=None)
ax9.fill_between([0,0.08],[1,1],[1e5,1e5],color='Grey',alpha=0.7,edgecolor=None)
index=find_neartime(colat0[0,:],np.pi/180*120)
lb4list=lb4[:,index:].flatten()
babs4list=babs4[:,index:].flatten()
lb4vlist=[lb4list[i] for i in range(len(lb4list)) if \
          (babs4list[i]>1000/2.8 and babs4list[i]<1500/2.8) or (babs4list[i]>500/2.8 and babs4list[i]<750/2.8)]
_,bins,_=ax9.hist(lb4list,bins=50,color='darkorange',alpha=0.3,edgecolor='black',range=[0,0.8],rasterized=True)
_=ax9.hist(lb4vlist,color='darkorange',edgecolor='black',bins=bins,rasterized=True)
ax9.spines['top'].set_visible(False)
ax9.spines['right'].set_visible(False)
ax9.set_yscale('log')
ax9.set_xlim([0,0.8])
ax9.set_ylim([10,1e3])
ax9.set_yticks([1e2,1e5])
ax9.set_xlabel('$L_B$ [$r_\star$]')
ax9.text(-0.28,0.95,'J',fontsize=13,weight='bold',transform=ax9.transAxes)

fig1 = plt.gcf()
plt.show()
fig1.savefig('/users/jiale/desktop/figure2.pdf',format='pdf',bbox_inches='tight')

cmap = plt.cm.RdYlBu  # define the colormap
cmaplist = [cmap(i) for i in range(cmap.N)]

# create the new map
cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, cmap.N)

# define the bins and normalize
bounds2 = np.linspace(0, 0.8, 256)
norm2 = matplotlib.colors.BoundaryNorm(bounds2, cmap2.N)

rho_list=[0,30,15,8]
bmax=6000

fig=plt.figure(figsize=(10,9),dpi=400)
plt.rcParams.update({'font.size': 11})
ax0=fig.add_axes([0.02,0.8,0.22,0.16])
ax1=fig.add_axes([0.26,0.8,0.22,0.16])
ax2=fig.add_axes([0.5,0.8,0.22,0.16])
ax3=fig.add_axes([0.74,0.8,0.22,0.16])
ax00=fig.add_axes([0.97,0.8,0.01,0.16])

ax4=fig.add_axes([0.02,0.53,0.22,0.16])
ax5=fig.add_axes([0.26,0.53,0.22,0.16])
ax6=fig.add_axes([0.5,0.53,0.22,0.16])
ax7=fig.add_axes([0.74,0.53,0.22,0.16])
ax40=fig.add_axes([0.97,0.53,0.01,0.16])

ax8=fig.add_axes([0.02,0.26,0.22,0.16])
ax9=fig.add_axes([0.26,0.26,0.22,0.16])
ax10=fig.add_axes([0.5,0.26,0.22,0.16])
ax11=fig.add_axes([0.74,0.26,0.22,0.16])
ax80=fig.add_axes([0.97,0.26,0.01,0.16])

ax12=fig.add_axes([0.02,0.02,0.22,0.16])
ax13=fig.add_axes([0.26,0.02,0.22,0.16])
ax14=fig.add_axes([0.5,0.02,0.22,0.16])
ax15=fig.add_axes([0.74,0.02,0.22,0.16])
axlist=[ax0,ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14,ax15]

imgi=0
lbminlist=[]
for rho in rho_list:
    if rho==0:
        npz_name='bmap'+'_lat'+str(10)+'_nospot'+'.npz'
    else:   
        npz_name='bmap'+'_lat'+str(10)+'_rho'+str(rho)+'_bmax'+str(bmax)+'.npz'
    npz_file='/Users/jiale/Desktop/projects/PT2021_0019/0319/starspot/'+npz_name
    npz_data=np.load(npz_file)
    lon0=npz_data['lon0']
    colat0=npz_data['colat0']
    br=npz_data['br']
    a0=axlist[imgi].pcolor(lon0,np.cos(colat0),br/1e3,cmap=cmap1,norm=norm1,rasterized=True)
    axlist[imgi].plot([-np.pi,np.pi],[np.sin(10/180*np.pi),np.sin(10/180*np.pi)],'--',color='grey')
    axlist[imgi].plot([-np.pi,np.pi],[np.sin(70/180*np.pi),np.sin(70/180*np.pi)],'--',color='grey')
    axlist[imgi].set_xticks([-np.pi/3*2,-np.pi/3,0,np.pi/3,np.pi/3*2],[240,300,0,60,120])
    axlist[imgi].set_xlabel('Longitude [\N{degree sign}]')
    axlist[imgi].set_yticks([])
    axlist[imgi].set_xlim([lon0[-1,0],lon0[0,0]])
    if imgi in [0]:
        axlist[imgi].set_yticks([np.sin(-60/180*np.pi),np.sin(-30/180*np.pi),0,\
                                 np.sin(30/180*np.pi),np.sin(60/180*np.pi)],['-60','-30','0','30','60'])
        axlist[imgi].set_title('No starspots',fontsize=12)
        axlist[imgi].set_ylabel('Latitude [$\degree$]')
        axlist[imgi].text(-0.2,1.1,'A',fontsize=14,weight='bold',transform=axlist[imgi].transAxes)
    else:
        axlist[imgi].set_yticks([])
        axlist[imgi].set_title('Starspots '+r'$\rho$='+str(rho)+'\N{degree sign}',fontsize=12)
    imgi+=1
a00=plt.colorbar(a0,cax=ax00,label='$B_r$ [kG]')
a00.set_ticks([-5,-0.5,0,0.5,5])

for rho in rho_list:
    if rho==0:
        npz_name='bmap'+'_lat'+str(10)+'_nospot'+'.npz'
    else:   
        npz_name='bmap'+'_lat'+str(10)+'_rho'+str(rho)+'_bmax'+str(bmax)+'.npz'
    npz_file='/Users/jiale/Desktop/projects/PT2021_0019/0319/starspot/'+npz_name
    npz_data=np.load(npz_file)
    lon1=npz_data['lon1']
    colat1=npz_data['colat1']
    r1=npz_data['r1']
    babs1=npz_data['babs1']
    lb1=npz_data['lb1']
    a0=axlist[imgi].pcolor(lon1,r1,lb1,cmap=cmap2,norm=norm2,rasterized=True)
    axlist[imgi].set_xticks([-np.pi/3*2,-np.pi/3,0,np.pi/3,np.pi/3*2],[240,300,0,60,120])
    axlist[imgi].set_xlabel('Longitude [\N{degree sign}]')
    axlist[imgi].contour(lon1,r1,lb1,levels=[0.15],colors='black')
    #axlist[imgi].contour(lon1, r1[:,:], babs1[:,:],levels=[1000/2.8,1500/2.8],colors='black')
    #axlist[imgi].contour(lon1, r1[:,:], babs1[:,:],levels=[1000/2.8/2,1500/2.8/2],colors='darkgreen')
    lbmin=np.min(lb1,axis=0)
    lbminlist.append(lbmin)
    axlist[imgi].set_xlim([lon1[-1,0],lon1[0,0]])
    if imgi in [4]:
        axlist[imgi].set_yticks([1.2,1.4,1.6,1.8,2])
        axlist[imgi].set_title('Lat.=10\N{degree sign}, No starspots',fontsize=12)
        axlist[imgi].set_ylabel('$r$ [$r_{\star}$]')
        axlist[imgi].text(-0.2,1.1,'B',fontsize=14,weight='bold',transform=axlist[imgi].transAxes)
    else:
        axlist[imgi].set_yticks([1.2,1.4,1.6,1.8,2],["","","","",""])
        axlist[imgi].set_title('Lat.=10\N{degree sign}, '+r'$\rho$='+str(rho)+'\N{degree sign}',fontsize=12)
    imgi+=1
a01=plt.colorbar(a0,cax=ax40,label='$L_B$ [$r_\star$]')
a01.set_ticks([0,0.2,0.4,0.6])
    
for rho in rho_list:
    if rho==0:
        npz_name='bmap'+'_lat'+str(70)+'_nospot'+'.npz'
    else:   
        npz_name='bmap'+'_lat'+str(70)+'_rho'+str(rho)+'_bmax'+str(bmax)+'.npz'
    npz_file='/Users/jiale/Desktop/projects/PT2021_0019/0319/starspot/'+npz_name
    npz_data=np.load(npz_file)
    lon1=npz_data['lon1']
    colat1=npz_data['colat1']
    r1=npz_data['r1']
    babs1=npz_data['babs1']
    lb1=npz_data['lb1']
    
    a1=axlist[imgi].pcolor(lon1,r1,lb1,cmap=cmap2,norm=norm2,rasterized=True)
    axlist[imgi].contour(lon1,r1,lb1,levels=[0.15],colors='black')
    #axlist[imgi].contour(lon1, r1[:,:], babs1[:,:],levels=[1000/2.8,1500/2.8],colors='black')
    #axlist[imgi].contour(lon1, r1[:,:], babs1[:,:],levels=[1000/2.8/2,1500/2.8/2],colors='darkgreen')
    axlist[imgi].set_xticks([-np.pi/3*2,-np.pi/3,0,np.pi/3,np.pi/3*2],[240,300,0,60,120])
    lbmin=np.min(lb1,axis=0)
    lbminlist.append(lbmin)
    axlist[imgi].set_xlim([lon1[-1,0],lon1[0,0]])
    if imgi in [8]:
        axlist[imgi].set_yticks([1.2,1.4,1.6,1.8,2])
        axlist[imgi].set_title('Lat.=70\N{degree sign}, No starspots',fontsize=12)
        axlist[imgi].set_ylabel('$r$ [$r_\star$]')
        axlist[imgi].text(-0.2,1.1,'C',fontsize=14,weight='bold',transform=axlist[imgi].transAxes)
    else:
        axlist[imgi].set_yticks([1.2,1.4,1.6,1.8,2],["","","","",""])
        axlist[imgi].set_title('Lat.=70\N{degree sign}, '+r'$\rho$='+str(rho)+'\N{degree sign}',fontsize=12)
    axlist[imgi].set_xlabel('Longitude [\N{degree sign}]')
    imgi+=1
a02=plt.colorbar(a1,cax=ax80,label='$L_B$ [$r_\star$]')
a02.set_ticks([0,0.2,0.4,0.6])

for i in range(4):
    axlist[imgi].fill_between([1,2],[0,0],[0.15,0.15],color='grey',alpha=0.3,edgecolor=None)
    axlist[imgi].fill_between([1,2],[0,0],[0.08,0.08],color='grey',alpha=0.5,edgecolor=None)
    
    if i==0:
        axlist[imgi].plot(r1[0,:],lbminlist[i],'-',label='Lat.=10\N{degree sign}')
        axlist[imgi].plot(r1[0,:],lbminlist[i+4],'-',label='Lat.=70\N{degree sign}')
        axlist[imgi].set_ylabel('$L_B$ [$r_\star$]')
        axlist[imgi].legend()
        axlist[imgi].text(-0.2,1.1,'D',fontsize=14,weight='bold',transform=axlist[imgi].transAxes)
        axlist[imgi].set_yticks([0,0.2,0.4,0.6])
    else:
        axlist[imgi].plot(r1[0,:],lbminlist[i],'-')
        axlist[imgi].plot(r1[0,:],lbminlist[i+4],'-')
        axlist[imgi].set_yticks([0,0.2,0.4,0.6],["","","",""])
    axlist[imgi].set_ylim([0,0.8])
    axlist[imgi].set_xlim([1,2])
    axlist[imgi].set_xscale('log')
    axlist[imgi].set_xticks([1,1.2,1.4,1.6,1.8,2],['1','1.2','1.4','1.6','1.8','2'])
    axlist[imgi].set_xlabel('$r$ [$r_\star$]')
    imgi+=1
fig1 = plt.gcf()
plt.show()
fig1.savefig('/users/jiale/desktop/figure3.pdf',format='pdf',bbox_inches='tight')

