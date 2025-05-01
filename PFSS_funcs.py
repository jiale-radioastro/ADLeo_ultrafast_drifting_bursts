import numpy as np
from scipy.special import lpmv,lpmn
from tqdm import tqdm

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
