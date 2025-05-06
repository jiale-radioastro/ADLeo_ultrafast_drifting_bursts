import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PFSS_funcs import *

# Produce Fig. S7 
file_dir='data/magnetic_field/'
img_dir='publication_figures/'

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


fig=plt.figure(figsize=(7.25,7.25),dpi=400)
plt.rcParams.update({'font.size': 8})
ax0=fig.add_axes([0.07,0.76,0.2,0.22],projection='polar')
ax1=fig.add_axes([0.07+0.22,0.76,0.2,0.22],projection='polar')
ax2=fig.add_axes([0.07+0.44,0.76,0.2,0.22],projection='polar')
ax3=fig.add_axes([0.07+0.66,0.76,0.2,0.22],projection='polar')
ax00=fig.add_axes([0.96,0.77,0.01,0.2])
ax4=fig.add_axes([0.07,0.51,0.2,0.22],projection='polar')
ax5=fig.add_axes([0.07+0.22,0.51,0.2,0.22],projection='polar')
ax6=fig.add_axes([0.07+0.44,0.51,0.2,0.22],projection='polar')
ax7=fig.add_axes([0.07+0.66,0.51,0.2,0.22],projection='polar')
ax40=fig.add_axes([0.96,0.52,0.01,0.2])
ax8=fig.add_axes([0.09,0.435,0.16,0.07])
ax9=fig.add_axes([0.09+0.22,0.435,0.16,0.07])
ax10=fig.add_axes([0.09+0.44,0.435,0.16,0.07])
ax11=fig.add_axes([0.09+0.66,0.435,0.16,0.07])
ax12=fig.add_axes([0.07,0.15,0.2,0.22],projection='polar')
ax13=fig.add_axes([0.07+0.22,0.15,0.2,0.22],projection='polar')
ax14=fig.add_axes([0.07+0.44,0.15,0.2,0.22],projection='polar')
ax15=fig.add_axes([0.07+0.66,0.15,0.2,0.22],projection='polar')
ax120=fig.add_axes([0.96,0.16,0.01,0.2])
ax16=fig.add_axes([0.09,0.075,0.16,0.07])
ax17=fig.add_axes([0.09+0.22,0.075,0.16,0.07])
ax18=fig.add_axes([0.09+0.44,0.075,0.16,0.07])
ax19=fig.add_axes([0.09+0.66,0.075,0.16,0.07])

axlist=[ax0,ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14,ax15,ax16,ax17,ax18,ax19,ax00,ax40,ax120]

file_dir='data/magnetic_field/'
suffixlist=['2019a','2019b','2020a','2020b']
colorlist=['darkgreen','darkblue','darkorange','darkmagenta']
for i in [0,1,2,3]:
    suffix=suffixlist[i]
    npz_name='bmap'+suffix+'_1.05'+'.npz'
    npz_file=file_dir+npz_name
    npz_data=np.load(npz_file)
    lon0=npz_data['lon0']
    colat0=npz_data['colat0']
    br=npz_data['br0']
    babs1=npz_data['babs1']
    lb1=npz_data['lb1']
    sz=np.shape(lon0)
    npz_name='bmap'+suffix+'_1.10'+'.npz'
    npz_file=file_dir+npz_name
    npz_data=np.load(npz_file)
    babs2=npz_data['babs1']
    lb2=npz_data['lb1']
    babs2[:,-1]=babs2[:,-2]*2-babs2[:,-3]
    
    a0 = axlist[i].pcolor(lon0[:,:], colat0[:,:], br[:,:]/1e3,cmap=cmap1,norm=norm1,rasterized=True)
    axlist[i].plot(lon0[:,0],np.ones(sz[0])*np.pi/6,'--',color='grey',linewidth=1,dashes=(10, 10))
    axlist[i].plot(lon0[:,0],np.ones(sz[0])*np.pi/3,'--',color='grey',linewidth=1,dashes=(10, 10))
    axlist[i].plot(lon0[:,0],np.ones(sz[0])*np.pi/2,color='grey',linewidth=1)
    axlist[i].set_theta_offset(-np.pi/2)
    axlist[i].set_theta_direction(-1)
    axlist[i].set_rticks([])
    axlist[i].set_xticks([np.pi/4*1,np.pi/4*3,np.pi/4*5,np.pi/4*7])
    axlist[i].set_ylim([0,120/180*np.pi])
    axlist[i].set_title(suffix)
    axlist[i].grid(False)
    
    if i==0:
        axlist[i].text(-0.2,0.05,'Radial magnetic field', rotation='vertical',fontsize=9,weight='bold',transform=axlist[i].transAxes)
        a01=plt.colorbar(a0,cax=ax00,label='$B_r$ [G]')
        a01.set_ticks([-5,-0.5,0,0.5,5])
    
    a1 = axlist[i+4].pcolor(lon0[:,:], colat0[:,:], lb1[:,:],cmap=cmap2,norm=norm2,rasterized=True)
    axlist[i+4].plot(lon0[:,0],np.ones(sz[0])*np.pi/6,'--',color='grey',linewidth=1,dashes=(10, 10))
    axlist[i+4].plot(lon0[:,0],np.ones(sz[0])*np.pi/3,'--',color='grey',linewidth=1,dashes=(10, 10))
    axlist[i+4].plot(lon0[:,0],np.ones(sz[0])*np.pi/2,color='grey',linewidth=1)
    axlist[i+4].contour(lon0[:,:], colat0[:,:], babs1[:,:],levels=[1000/2.8,1500/2.8],colors='black')
    axlist[i+4].contour(lon0[:,:], colat0[:,:], babs1[:,:],levels=[1000/2.8/2,1500/2.8/2],colors='darkgreen')
    axlist[i+4].set_theta_offset(-np.pi/2)
    axlist[i+4].set_theta_direction(-1)
    axlist[i+4].set_rticks([])
    axlist[i+4].set_xticks([np.pi/4*1,np.pi/4*3,np.pi/4*5,np.pi/4*7])
    axlist[i+4].set_ylim([0,120/180*np.pi])
    axlist[i+4].grid(False)
    
    if i==0:
        axlist[i+4].text(-0.2,-0.25,'Scale height at $r=1.05\;r_\star$', rotation='vertical',fontsize=9,weight='bold',transform=axlist[i+4].transAxes)
        a11=plt.colorbar(a1,cax=ax40,label='$L_B$ $[r_\star]$')
        a11.set_ticks([0,0.1,0.2,0.3,0.4,0.5])
        axlist[i].text(-0.08,1.05,'A',fontsize=9,weight='bold',transform=axlist[i].transAxes)
        axlist[i+4].text(-0.08,1.05,'B',fontsize=9,weight='bold',transform=axlist[i+4].transAxes)
        axlist[i+12].text(-0.08,1.05,'C',fontsize=9,weight='bold',transform=axlist[i+12].transAxes)
    
    axlist[i+8].fill_between([0,0.15],[1,1],[1e5,1e5],color='Grey',alpha=0.3,edgecolor=None)
    axlist[i+8].fill_between([0,0.08],[1,1],[1e5,1e5],color='Grey',alpha=0.7,edgecolor=None)
    index=find_neartime(colat0[0,:],np.pi/180*120)
    lb1list=lb1[:,index:].flatten()
    babs1list=babs1[:,index:].flatten()
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
    
    a2 = axlist[i+12].pcolor(lon0[:,:], colat0[:,:], lb2[:,:],cmap=cmap2,norm=norm2,rasterized=True)
    axlist[i+12].plot(lon0[:,0],np.ones(sz[0])*np.pi/6,'--',color='grey',linewidth=1,dashes=(10, 10))
    axlist[i+12].plot(lon0[:,0],np.ones(sz[0])*np.pi/3,'--',color='grey',linewidth=1,dashes=(10, 10))
    axlist[i+12].plot(lon0[:,0],np.ones(sz[0])*np.pi/2,color='grey',linewidth=1)
    axlist[i+12].contour(lon0[:,:], colat0[:,:], babs2[:,:],levels=[1000/2.8,1500/2.8],colors='black')
    axlist[i+12].contour(lon0[:,:], colat0[:,:], babs2[:,:],levels=[1000/2.8/2,1500/2.8/2],colors='darkgreen')
    axlist[i+12].set_theta_offset(-np.pi/2)
    axlist[i+12].set_theta_direction(-1)
    axlist[i+12].set_rticks([])
    axlist[i+12].set_xticks([np.pi/4*1,np.pi/4*3,np.pi/4*5,np.pi/4*7])
    axlist[i+12].set_ylim([0,120/180*np.pi])
    axlist[i+12].grid(False)
    
    if i==0:
        axlist[i+12].text(-0.2,-0.25,'Scale height at $r=1.1\;r_\star$', rotation='vertical',fontsize=9,weight='bold',transform=axlist[i+12].transAxes)
        a21=plt.colorbar(a2,cax=ax120,label='$L_B$ $[r_\star]$')
        a21.set_ticks([0,0.1,0.2,0.3,0.4,0.5])
    
    axlist[i+16].fill_between([0,0.15],[1,1],[1e5,1e5],color='Grey',alpha=0.3,edgecolor=None)
    axlist[i+16].fill_between([0,0.08],[1,1],[1e5,1e5],color='Grey',alpha=0.7,edgecolor=None)
    index=find_neartime(colat0[0,:],np.pi/180*120)
    lb2list=lb2[:,index:].flatten()
    babs2list=babs2[:,index:].flatten()
    lb2vlist=[lb2list[i] for i in range(len(lb2list)) if \
              (babs2list[i]>1000/2.8 and babs2list[i]<1500/2.8) or (babs2list[i]>500/2.8 and babs2list[i]<750/2.8)]
    _,bins,_=axlist[i+16].hist(lb2list,bins=50,color=colorlist[i],alpha=0.3,edgecolor='black',range=[0,0.8],rasterized=True)
    _=axlist[i+16].hist(lb2vlist,color=colorlist[i],edgecolor='black',bins=bins,rasterized=True)
    axlist[i+16].spines['top'].set_visible(False)
    axlist[i+16].spines['right'].set_visible(False)
    axlist[i+16].set_yscale('log')
    axlist[i+16].set_xlim([0,0.8])
    axlist[i+16].set_ylim([10,1e3])
    axlist[i+16].set_yticks([1e2,1e5])
    axlist[i+16].set_xlabel('$L_B$ [$r_\star$]')
    axlist[i+16].set_xticks([0,0.2,0.4,0.6,0.8])

fig.savefig(img_dir+'sfigure7.pdf',format='pdf',bbox_inches='tight')