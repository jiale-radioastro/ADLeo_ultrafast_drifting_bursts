import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PFSS_funcs import *

# Produce Fig. 2 
dat_dir='/Users/jiale/Desktop/projects/PT2021_0019/0319/zdi/'
datfile=dat_dir+'outMagCoeff_ADLeo_2019b.dat'
datstr = np.genfromtxt(datfile,delimiter='\t',dtype=str)
img_dir='/Users/jiale/Desktop/projects/PT2021_0019/0319/publication_figures/'

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
fig1.savefig(img_dir+'figure2.pdf',format='pdf',bbox_inches='tight')

