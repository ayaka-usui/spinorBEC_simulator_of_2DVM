# -*- coding: utf-8 -*-

import numpy as np
import qutip as qt
from scipy.sparse import csc_matrix, lil_matrix
from tqdm import tqdm

def operators(baza, dim):
    Jx=lil_matrix((dim,dim),dtype="complex")
    Qzx=lil_matrix((dim,dim),dtype="complex")
    Dxy=lil_matrix((dim,dim),dtype="complex")
    Qxy=lil_matrix((dim,dim),dtype="complex")
    Qyz=lil_matrix((dim,dim),dtype="complex")
    N0=lil_matrix((dim,dim),dtype="complex")
    Np=lil_matrix((dim,dim),dtype="complex")
    Nm=lil_matrix((dim,dim),dtype="complex")
    Jz=lil_matrix((dim,dim),dtype="complex")
    Y=lil_matrix((dim,dim),dtype="complex")
    Jy=lil_matrix((dim,dim),dtype="complex")
    for i in range(dim):
        for j in range(i,dim):
            if (baza[j,0]==baza[i,0] and baza[j,1]-1 == baza[i,1] and baza[j,2] +1== baza[i,2]): # \sigma^{\dagger} \tau_-
                Jx[i,j]=1/np.sqrt(2)*np.sqrt(baza[j,1]*(baza[j,2]+1));
                Jx[j,i]=np.conjugate(Jx[i,j])
                Jy[i,j]=1j/np.sqrt(2)*np.sqrt(baza[j,1]*(baza[j,2]+1));
                Jy[j,i]=np.conjugate(Jy[i,j])
                Qzx[i,j]=-1/np.sqrt(2)*np.sqrt(baza[j,1]*(baza[j,2]+1));
                Qzx[j,i]=np.conjugate(Qzx[i,j])
                Qyz[i,j]=-1j/np.sqrt(2)*np.sqrt(baza[j,1]*(baza[j,2]+1));
                Qyz[j,i]=np.conjugate(Qyz[i,j])
            
            if (baza[j,0]==baza[i,0] and baza[j,1]+1 == baza[i,1] and baza[j,2] -1== baza[i,2]): # \tau_-^{\dagger} \sigma
                Jx[i,j]=1/np.sqrt(2)*np.sqrt(baza[j,2]*(baza[j,1]+1));
                Jx[j,i]=np.conjugate(Jx[i,j])
                Jy[i,j]=-1j/np.sqrt(2)*np.sqrt(baza[j,2]*(baza[j,1]+1));
                Jy[j,i]=np.conjugate(Jy[i,j])
                Qzx[i,j]=-1/np.sqrt(2)*np.sqrt(baza[j,2]*(baza[j,1]+1));
                Qzx[j,i]=np.conjugate(Qzx[i,j])
                Qyz[i,j]=1j/np.sqrt(2)*np.sqrt(baza[j,2]*(baza[j,1]+1));
                Qyz[j,i]=np.conjugate(Qyz[i,j])
            
            if (baza[j,0]-1==baza[i,0] and baza[j,1]+1 == baza[i,1] and baza[j,2]== baza[i,2]): # \tau_+^{\dagger} \sigma
                Jx[i,j]=1/np.sqrt(2)*np.sqrt(baza[j,0]*(baza[j,1]+1));
                Jx[j,i]=np.conjugate(Jx[i,j])
                Jy[i,j]=1j/np.sqrt(2)*np.sqrt(baza[j,0]*(baza[j,1]+1));
                Jy[j,i]=np.conjugate(Jy[i,j])
                Qzx[i,j]=1/np.sqrt(2)*np.sqrt(baza[j,0]*(baza[j,1]+1));
                Qzx[j,i]=np.conjugate(Qzx[i,j])
                Qyz[i,j]=1j/np.sqrt(2)*np.sqrt(baza[j,0]*(baza[j,1]+1));
                Qyz[j,i]=np.conjugate(Qyz[i,j])
    
            if (baza[j,0]+1==baza[i,0] and baza[j,1]-1 == baza[i,1] and baza[j,2]== baza[i,2]): # \sigma^{\dagger} \tau_+
                Jx[i,j]=1/np.sqrt(2)*np.sqrt(baza[j,1]*(baza[j,0]+1));
                Jx[j,i]=np.conjugate(Jx[i,j])
                Jy[i,j]=-1j/np.sqrt(2)*np.sqrt(baza[j,1]*(baza[j,0]+1));
                Jy[j,i]=np.conjugate(Jy[i,j])
                Qzx[i,j]=1/np.sqrt(2)*np.sqrt(baza[j,1]*(baza[j,0]+1));
                Qzx[j,i]=np.conjugate(Qzx[i,j])
                Qyz[i,j]=-1j/np.sqrt(2)*np.sqrt(baza[j,1]*(baza[j,0]+1));
                Qyz[j,i]=np.conjugate(Qyz[i,j])
                
            if (baza[i,1] == baza[j,1] and baza[i,0] == baza[j,0]-1 and baza[i,2] == baza[j,2] +1): # \tau_+^{\dagger} \tau_-
                Dxy[i,j]=np.sqrt(baza[j,0]*(baza[j,2]+1));
                Dxy[j,i]=np.conjugate(Dxy[i,j])
                Qxy[i,j]=1j*np.sqrt(baza[j,0]*(baza[j,2]+1));
                Qxy[j,i]=np.conjugate(Qxy[i,j])
          
            if (baza[i,1] == baza[j,1] and baza[i,0] == baza[j,0]+1 and baza[i,2] == baza[j,2] -1): # \tau_-^{\dagger} \tau_+
                Dxy[i,j]=np.sqrt(baza[j,2]*(baza[j,0]+1));
                Dxy[j,i]=np.conjugate(Dxy[i,j])
                Qxy[i,j]=-1j*np.sqrt(baza[j,2]*(baza[j,0]+1));
                Qxy[j,i]=np.conjugate(Qxy[i,j])
                
            if i==j:
                Jz[i,i]=baza[i,0]-baza[i,2];
                N0[i,i]=baza[i,1];
                Np[i,i]=baza[i,0];
                Nm[i,i]=baza[i,2];
                Y[i,i]=1/np.sqrt(3)*(baza[i,2]+baza[i,0]-baza[i,1]*2);
            
    
    Jx=qt.Qobj(Jx)
    Jy=qt.Qobj(Jy)
    Jz=qt.Qobj(Jz)
    Qxy=qt.Qobj(Qxy)
    Dxy=qt.Qobj(Dxy)
    Y=qt.Qobj(Y)
    Qzx=qt.Qobj(Qzx)
    Qyz=qt.Qobj(Qyz)
    Np=qt.Qobj(Np)
    Nm=qt.Qobj(Nm)
    N0=qt.Qobj(N0)
    
    
        
        
    return (Jx,Qzx,Dxy,N0,Jz,Y,Qxy,Jy,Qyz,Np,Nm)

def operators_s2(N, number_operators = False):
    Jx=np.zeros([N+1,N+1],dtype="complex")
    Jy=np.zeros([N+1,N+1],dtype="complex")
    Jz=np.zeros([N+1,N+1],dtype="complex")
    Nl=np.zeros([N+1,N+1],dtype="complex")
    for k in range(1, N+2):
        for j in range(k, N+2):
            if j == k+1:
                Jx[j-1, k-1] = 1/2*np.sqrt((j-1)*(N-(k-1)))
                Jx[k-1,j-1] = Jx[j-1,k-1]
                Jy[j-1, k-1] = 1/(2j)*np.sqrt((j-1)*(N-(k-1)))
                Jy[k-1,j-1] = np.conjugate(Jy[j-1,k-1])
            if j==k:
                Jz[j-1,k-1]= 1/2*((j-1)- (N-(k-1)))
                Nl[j-1,k-1]= (j-1)
    Jx=qt.Qobj(Jx)
    Jy=qt.Qobj(Jy)
    Jz=qt.Qobj(Jz)
    Nl=qt.Qobj(Nl)
    if number_operators:
        return Jx, Jy, Jz, Nl
    else:          
        return Jx, Jy, Jz

def operators_m0(base, N):
    dim = int(N/2) +1
    Jx=lil_matrix((dim,dim),dtype="complex")
    Qzx=lil_matrix((dim,dim),dtype="complex")
    Dxy=lil_matrix((dim,dim),dtype="complex")
    Qxy=lil_matrix((dim,dim),dtype="complex")
    Qyz=lil_matrix((dim,dim),dtype="complex")
    N0=lil_matrix((dim,dim),dtype="complex")
    Np=lil_matrix((dim,dim),dtype="complex")
    Nm=lil_matrix((dim,dim),dtype="complex")
    Jz=lil_matrix((dim,dim),dtype="complex")
    Y=lil_matrix((dim,dim),dtype="complex")
    Jy=lil_matrix((dim,dim),dtype="complex")
    
    Jx_2 = lil_matrix((dim,dim),dtype="complex")
    Qzx_2 = lil_matrix((dim,dim),dtype="complex")
    Dxy_2 = lil_matrix((dim,dim),dtype="complex")
    Qxy_2 = lil_matrix((dim,dim),dtype="complex")
    Qyz_2 = lil_matrix((dim,dim),dtype="complex")
    N0_2 = lil_matrix((dim,dim),dtype="complex")
    Np_2 = lil_matrix((dim,dim),dtype="complex")
    Nm_2 =lil_matrix((dim,dim),dtype="complex")
    Jz_2 =lil_matrix((dim,dim),dtype="complex")
    Y_2 = lil_matrix((dim,dim),dtype="complex")
    Jy_2 = lil_matrix((dim,dim),dtype="complex")
    op = lil_matrix((dim,dim),dtype="complex")
    op_conj = lil_matrix((dim,dim),dtype="complex")
    Jy_2 = lil_matrix((dim,dim),dtype="complex")
    for i in range(0, dim):
        N0[i,i] = base[i,1]
        Np[i,i] = base[i,0]
        for j in range(i, dim):
            if(base[j,1] + 2 == base[i,1] ):
                op_conj[i,j] = np.sqrt(base[j,0]*(base[j,2])*(base[j,1]+1)*(base[j,1]+2))
    op = np.transpose(np.conj(op_conj))
    Nm = Np
                
    Y = 1/np.sqrt(3)*(Np + Nm - 2*N0)
    # Y_2 = 1/3*(4*N0 + Nm*Nm - 4*Nm*N0 + 2*Nm*Np - 4*N0*Np + 4*N0*N0 - 4*N0 + Np*Np)
    Y_2 = 1/3*(Nm*Nm - 4*Nm*N0 + 2*Nm*Np - 4*N0*Np + 4*N0*N0 + Np*Np)            
    Jx_2 = 1/2*(Nm + 2*N0 + Np + 2*Nm*N0 + 2*N0*Np+ 2*op + 2*op_conj)
    Jy_2 = Jx_2
    Dxy_2 = Nm + Np +2*Nm*Np
    Qxy_2 = Nm + Np +2*Nm*Np
    Qyz_2 = 1/2*(Nm + 2*N0 + Np +2*Nm*N0 + 2*N0*Np - 2*op - 2*op_conj)
    Qzx_2 = Qyz_2
    Jz_2 = (Np - Nm)*(Np - Nm)
    
    Jx=qt.Qobj(Jx)
    Jy=qt.Qobj(Jy)
    Jz=qt.Qobj(Jz)
    Qxy=qt.Qobj(Qxy)
    Dxy=qt.Qobj(Dxy)
    Y=qt.Qobj(Y)
    Qzx=qt.Qobj(Qzx)
    Qyz=qt.Qobj(Qyz)
    Np=qt.Qobj(Np)
    Nm=qt.Qobj(Nm)
    N0=qt.Qobj(N0)
    Jx_2 = qt.Qobj(Jx_2)
    Jy_2 = qt.Qobj(Jy_2)
    Jz_2 = qt.Qobj(Jz_2)
    Qxy_2 = qt.Qobj(Qxy_2)
    Dxy_2 = qt.Qobj(Dxy_2)
    Y_2 = qt.Qobj(Y_2)
    Qzx_2 = qt.Qobj(Qzx_2)
    Qyz_2 = qt.Qobj(Qyz_2)
    Np_2 = qt.Qobj(Np_2)
    Nm_2 = qt.Qobj(Nm_2)
    N0_2 = qt.Qobj(N0_2)
    op = qt.Qobj(op)
    op_conj = qt.Qobj(op_conj)
    
    return (N0,Y,Np,Nm,Jx_2,Qzx_2,Dxy_2,N0_2,Jz_2,Y_2,Qxy_2,Jy_2,Qyz_2,Np_2,Nm_2,op, op_conj)