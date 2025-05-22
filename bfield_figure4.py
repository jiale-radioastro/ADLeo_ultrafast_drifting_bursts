import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PFSS_funcs import *

# Produce Fig. S6
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

cmap = plt.cm.RdYlBu  # define the colormap
cmaplist = [cmap(i) for i in range(cmap.N)]
cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, cmap.N)
bounds2 = np.linspace(0, 0.8, 256)
norm2 = matplotlib.colors.BoundaryNorm(bounds2, cmap2.N)

rho=8
bmax_list=[0,3000,6000,9000]

fig=plt.figure(figsize=(7.25,6.5),dpi=400)
plt.rcParams.update({'font.size': 8})
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
for bmax in bmax_list:
    if bmax==0:
        npz_name='bmap'+'_lat'+str(10)+'_nospot'+'.npz'
    else:   
        npz_name='bmap'+'_lat'+str(10)+'_rho'+str(rho)+'_bmax'+str(bmax)+'.npz'
    npz_file=file_dir+npz_name
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
        axlist[imgi].set_title('No starspots')
        axlist[imgi].set_ylabel('Latitude [$\degree$]')
        axlist[imgi].text(-0.2,1.1,'A',fontsize=9,weight='bold',transform=axlist[imgi].transAxes)
    else:
        axlist[imgi].set_yticks([])
        axlist[imgi].set_title('Starspots '+r'$B_{max}$='+str(bmax/1e3)+' kG')
    imgi+=1
a00=plt.colorbar(a0,cax=ax00,label='$B_r$ [kG]')
a00.set_ticks([-5,-0.5,0,0.5,5])

for bmax in bmax_list:
    if bmax==0:
        npz_name='bmap'+'_lat'+str(10)+'_nospot'+'.npz'
    else:   
        npz_name='bmap'+'_lat'+str(10)+'_rho'+str(rho)+'_bmax'+str(bmax)+'.npz'
    npz_file=file_dir+npz_name
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
    lbmin=np.min(lb1,axis=0)
    lbminlist.append(lbmin)
    axlist[imgi].set_xlim([lon1[-1,0],lon1[0,0]])
    if imgi in [4]:
        axlist[imgi].set_yticks([1.2,1.4,1.6,1.8,2])
        axlist[imgi].set_title('Lat.=10\N{degree sign}, No starspots')
        axlist[imgi].set_ylabel('$r$ [$r_{\star}$]')
        axlist[imgi].text(-0.2,1.1,'B',fontsize=9,weight='bold',transform=axlist[imgi].transAxes)
    else:
        axlist[imgi].set_yticks([1.2,1.4,1.6,1.8,2],["","","","",""])
        axlist[imgi].set_title('Lat.=10\N{degree sign}, '+r'$B_{max}$='+str(bmax/1e3)+' kG')
    imgi+=1
a01=plt.colorbar(a0,cax=ax40,label='$L_B$ [$r_\star$]')
a01.set_ticks([0,0.2,0.4,0.6])
    
for bmax in bmax_list:
    if bmax==0:
        npz_name='bmap'+'_lat'+str(70)+'_nospot'+'.npz'
    else:   
        npz_name='bmap'+'_lat'+str(70)+'_rho'+str(rho)+'_bmax'+str(bmax)+'.npz'
    npz_file=file_dir+npz_name
    npz_data=np.load(npz_file)
    lon1=npz_data['lon1']
    colat1=npz_data['colat1']
    r1=npz_data['r1']
    babs1=npz_data['babs1']
    lb1=npz_data['lb1']
    
    a1=axlist[imgi].pcolor(lon1,r1,lb1,cmap=cmap2,norm=norm2,rasterized=True)
    axlist[imgi].contour(lon1,r1,lb1,levels=[0.15],colors='black')
    axlist[imgi].set_xticks([-np.pi/3*2,-np.pi/3,0,np.pi/3,np.pi/3*2],[240,300,0,60,120])
    lbmin=np.min(lb1,axis=0)
    lbminlist.append(lbmin)
    axlist[imgi].set_xlim([lon1[-1,0],lon1[0,0]])
    if imgi in [8]:
        axlist[imgi].set_yticks([1.2,1.4,1.6,1.8,2])
        axlist[imgi].set_title('Lat.=70\N{degree sign}, No starspots')
        axlist[imgi].set_ylabel('$r$ [$r_\star$]')
        axlist[imgi].text(-0.2,1.1,'C',fontsize=9,weight='bold',transform=axlist[imgi].transAxes)
    else:
        axlist[imgi].set_yticks([1.2,1.4,1.6,1.8,2],["","","","",""])
        axlist[imgi].set_title('Lat.=10\N{degree sign}, '+r'$B_{max}$='+str(bmax/1e3)+' kG')
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
        axlist[imgi].text(-0.2,1.1,'D',fontsize=9,weight='bold',transform=axlist[imgi].transAxes)
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
fig.savefig(img_dir+'sfigure6.pdf',format='pdf',bbox_inches='tight')