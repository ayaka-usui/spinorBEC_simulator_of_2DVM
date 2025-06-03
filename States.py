# -*- coding: utf-8 -*-
import numpy as np
from scipy.sparse import csc_matrix
import scipy.special
import sys


################# LIGHT STATES ################################################
def vac_L(N,Nph,Nph_mean):
   light=np.zeros(Nph+1)
   light[0]=1
   return light
   
def coh_L(N,Nph,Nph_mean):
    light=np.zeros(Nph+1)
    for i in range(Nph+1):
        light[i]=np.exp(-1/2*Nph_mean)*(np.sqrt(Nph_mean)**i)/np.sqrt(float(np.math.factorial(float(i))))
    if np.linalg.norm(light) < (1-10**-4):
        sys.exit('TOO SMALL LIGHT SPACE')
    return light

def fock_L(N,Nph,Nph_mean):
    light=np.zeros(Nph+1)
    for i in range(Nph+1):
        if i==Nph_mean:
            light[i]=1
    return light
################### ATOMIC STATES #############################################
   
def coh_A(N,theta,phi):
    atom=np.zeros(N+1,dtype='complex')
    for i in range(N+1):
        atom[i]=np.sqrt(scipy.special.binom(N,i))*(np.cos(theta/2))**(i)*(np.sin(theta/2)*np.exp(1j*phi))**(N-i)
    if np.linalg.norm(atom) < (1-10**-4):
        print(np.linalg.norm(atom))
        sys.exit('NORM PROBLEM!')
    return atom

def fock_A(N,base,left_mode,right_mode):
    atom=np.zeros(N+1)
    if left_mode+right_mode != N:
        sys.exit('sum of modes occupation must be equal to N')
    for i in range(N+1):
        if base[i,0]==left_mode and base[i,1]==right_mode:
            atom[i]=1
    return atom
def sup_A(N,base,left_mode,right_mode,left_mode2,right_mode2):
    atom=np.zeros(N+1)
    if left_mode+right_mode != N:
        sys.exit('sum of modes occupation must be equal to N')
    for i in range(N+1):
        if base[i,0]==left_mode and base[i,1]==right_mode:
            atom[i]=1/np.sqrt(2)
        if base[i,0]==left_mode2 and base[i,1]==right_mode2:
            atom[i]=1/np.sqrt(2)
    return atom
################### INITIAL STATE #############################################
    
def init(dim,N,Nph,light,atom):
    init = np.zeros(dim,dtype='complex')
    k=0
    for i in range(Nph+1):
        for j in range(N+1):
            init[k]=light[i]*atom[j]
            k=k+1
    if np.linalg.norm(init) < (1-10**-4):
        sys.exit('NORM PROBLEM!')
    return init