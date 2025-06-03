# -*- coding: utf-8 -*-
import numpy as np
import Operators
from tqdm import tqdm
from scipy.linalg import expm
from scipy.linalg import sqrtm
import qutip as qt
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from functions import *
from scipy.special import binom
import os

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from numpy import linalg as LA
from pytictoc import TicToc
from qutip.sparse import sp_eigs

from matplotlib import ticker

def delta(k1,k2):

    if k1 == k2:
        return 1
    else: #k1 =! k2
        return 0

def bend_degree(gamma):

    if gamma <= 0.2:
        return 0.0
    else:
        return np.sqrt((5*gamma-1)/(3*gamma+1))
    
def sumlog(N0, N):
    
    result = 0.0

    for n in range(N0,N+1):
        result += np.log(n)
    
    return result

def covariance_matrix_spinhalf_m0(state,N,Sx_2,Sy_2,op,op_conj):
    
    COV = np.zeros([2,2],dtype=complex)
    COV[0,0] = qt.expect(Sx_2, state)*(4/N)
    COV[1,1] = qt.expect(Sy_2, state)*(4/N)
    COV[1,0] = qt.expect(op-op_conj, state)*(1j/N)
    COV[0,1] = COV[1,0]
    
    #
    Delta = (COV[0,0] - COV[1,1])**2 + (2*COV[1,0])**2
    lambda_minus = (COV[0,0] + COV[1,1] - np.sqrt(Delta))/2
    lambda_plus = (COV[0,0] + COV[1,1] + np.sqrt(Delta))/2

    return np.real(lambda_minus), np.real(lambda_plus)

def generate_base_m0(N):

    dim=int(N/2+1)
    base=np.zeros([dim,3])
    for k in range(dim):
        base[k,0] = k
        base[k,1] = N - 2*k
        base[k,2] = k

    return base, dim

def time_evolution_m0(N, gamma, Nt, ti, tf):

    # define base and operators
    base,dim = generate_base_m0(N)
    [N0,Y,Np,Nm,Jx_2,Qzx_2,Dxy_2,N0_2,Jz_2,Y_2,Qxy_2,Jy_2,Qyz_2,Np_2,Nm_2,op,op_conj] = Operators.operators_m0(base, N)
    Sx_spinhalf_2 = Jx_2/4
    Sy_spinhalf_2 = Qyz_2/4
    Sz_spinhalf = (-np.sqrt(3)*Y)/4 #(-np.sqrt(3)*Y - Dxy)/4
    Sz_spinhalf_2 = (3*Y_2 + Dxy_2)/16

    # define time
    timescale = np.linspace(ti, tf, Nt)

    # allocate space for entanglement criteria and observables
    # squ0_t = np.zeros(Nt)
    squ1_t = np.zeros(Nt)
    invQFI_t = np.zeros(Nt)
    expSz = np.zeros(Nt)
    squ1_t_0 = np.zeros(Nt)
    
    # initial state 0N0
    init = qt.Qobj(np.array([1 if base[i][1]==N else 0 for i in range(dim)]))
    
    # Hamiltonian for evolution
    # 0.2 for ciritical gamma
    H = -(1-gamma)*N0 - gamma/N*(1j*N0*np.pi/2).expm()*(Jx_2 + Jy_2 + Jz_2)*(-1j*N0*np.pi/2).expm()

    # prepare for time evolution operator
    evals, evecs = sp_eigs(H.data, H.isherm) # sparse=sparse,sort=sort, eigvals=eigvals, tol=tol, maxiter=maxiter)
    evecs = evecs.T
   
    for m in tqdm(range(0, Nt)):

        # time evolution
        t = timescale[m]
        # u_evo = (-1j * H * t).expm() # takes time
        u_evo = qt.Qobj(evecs@(np.diag(np.exp(-1j*evals*t))@evecs.conj().T))
        state = u_evo * init

        # squeezing and QFI
        lambda_minus, lambda_plus = covariance_matrix_spinhalf_m0(state,N,Sx_spinhalf_2,Sy_spinhalf_2,op,op_conj)
        # squ0_t[m] = (qt.expect(Sx_spinhalf_2, state))/qt.expect(Sz_spinhalf, state)**2*N # non-optimal squeezing parameter
        squ1_t[m] = lambda_minus/qt.expect(Sz_spinhalf, state)**2*(N/2)**2
        invQFI_t[m] = 1/lambda_plus
        expSz[m] = qt.expect(Sz_spinhalf, state)
        squ1_t_0[m] = lambda_minus
        
    return timescale, squ1_t, invQFI_t, expSz, squ1_t_0

#####################################

def test_squeezing_diff_maximum():

    N = 500
    gamma = 0.190
    Nt = 101
    tf = 1000.0

    timescale, squ1_t, invQFI_t, expSz, squ1_t_0 = time_evolution_m0(N,gamma,Nt,0.0,tf)

    fig = plt.figure()
    axs1 = plt.subplot(211)
    axs2 = plt.subplot(212)

    axs1.plot(timescale,np.log(squ1_t-invQFI_t),'k-',linewidth=3.0)
    axs1 = plt.gca()
    # ax.set_ylim([-5.0,2.0])
    axs1.axvline(x = timescale[np.argmax(squ1_t-invQFI_t)], color = 'r', linestyle = '--', label = 'axvline')
    
    axs2.plot(timescale,np.log(squ1_t),'k-',linewidth=3.0)
    axs2.plot(timescale,np.log(invQFI_t),'r--',linewidth=3.0)
    axs2 = plt.gca()
    # ax.set_ylim([-5.0,2.0])
    axs2.axvline(x = timescale[np.argmin(squ1_t)], color = 'b', linestyle = '--', label = 'axvline')
    axs2.axvline(x = timescale[np.argmin(invQFI_t)], color = 'b', linestyle = '--', label = 'axvline')
    axs2.axvline(x = timescale[np.argmax(squ1_t-invQFI_t)], color = 'r', linestyle = '--', label = 'axvline')
    plt.savefig(f'test_N{round(N)}_xi{round(gamma,3)}_0.png')

    # fig = plt.figure()
    # axs1 = plt.subplot(211)
    # axs2 = plt.subplot(212)
    # # axs1.plot(timescale,squ1_t,'k-',linewidth=3.0)
    # axs1.plot(timescale,squ1_t_0,'r--',linewidth=3.0)
    # axs2.plot(timescale,expSz/(N/2),'b:',linewidth=3.0)
    # ax = plt.gca()
    # plt.savefig(f'test_N{round(N)}_xi{round(gamma,2)}_1.png')

def save_squeezing_diff_maximum():

    array_N = np.array([1500])
    # array_N = np.linspace(6000, 10000, 5)

    # array_gamma = np.linspace(0.25, 0.3, 6)
    # array_gamma = np.linspace(0.199, 0.22, 22)
    array_gamma = np.linspace(0.18, 0.22, 41)
    # array_gamma = np.linspace(0.14, 0.17, 4)

    tf_0 = 1000.0 #400.0 #40.0
    Nt_0 = round(tf_0*10)+1

    for n in range(array_N.size):
        N = array_N[n]
        tf = tf_0
        Nt = Nt_0
        for j in range(array_gamma.size):

            #
            # if array_gamma[j] == 0.18 or array_gamma[j] == 0.19 or array_gamma[j] == 0.20 or array_gamma[j] == 0.21 or array_gamma[j] == 0.22:
            # if array_gamma[j] >= 0.18 and array_gamma[j] <= 0.22:
                # continue

            #
            # if N == 6000 and j <= 2:
            #     continue
            # if N == 6000 and j == 3:
            #     tf = 20.0
            #     Nt = round(tf*10)+1

            gamma = array_gamma[j]
            timescale, squ1_t, invQFI_t, expSz, squ1_t_0 = time_evolution_m0(N,gamma,Nt,0.0,tf)

            fig = plt.figure()
            axs1 = plt.subplot(211)
            axs2 = plt.subplot(212)
            axs1.plot(timescale,np.log(squ1_t-invQFI_t),'k-',linewidth=3.0)
            axs1 = plt.gca()
            axs1.axvline(x = timescale[np.argmax(squ1_t-invQFI_t)], color = 'r', linestyle = '--', label = 'axvline')
    
            axs2.plot(timescale,np.log(squ1_t),'k-',linewidth=3.0)
            axs2.plot(timescale,np.log(invQFI_t),'r--',linewidth=3.0)
            axs2 = plt.gca()
            axs2.axvline(x = timescale[np.argmin(squ1_t)], color = 'b', linestyle = '--', label = 'axvline')
            axs2.axvline(x = timescale[np.argmin(invQFI_t)], color = 'b', linestyle = '--', label = 'axvline')
            axs2.axvline(x = timescale[np.argmax(squ1_t-invQFI_t)], color = 'r', linestyle = '--', label = 'axvline')
            plt.savefig(f'test_N{round(N)}_xi{round(gamma,3)}.png')
            
            # if gamma <= 0.2:
            #     tf = tf_0
            #     Nt = Nt_0
            # else:
            #     tf = timescale[np.argmax(squ1_t-invQFI_t)] + 5.0
            #     Nt = round(tf*10)+1

            with open(f'squeezing_QFI_N{round(N)}_xi{round(gamma,3)}.npy', 'wb') as f:
                np.save(f, timescale)
                np.save(f, N)
                np.save(f, gamma)
                np.save(f, squ1_t)
                np.save(f, invQFI_t)
                np.save(f, expSz)
                np.save(f, squ1_t_0)
            
            np.savetxt(f'array_diff_QFI_N{round(N)}_xi{round(gamma,3)}.txt', squ1_t - invQFI_t)

def plot_squeezing_diff_maximum():

    N = 5000

    array_gamma = np.linspace(0.2, 0.3, 11)
    array_diff = np.zeros(array_gamma.size)

    for j in range(array_gamma.size):
        with open(f'squeezing_QFI_N{round(N)}_xi{round(array_gamma[j],2)}.npy', 'rb') as f:
            timescale = np.load(f)
            N_1 = np.load(f)
            gamma_1 = np.load(f)
            squ1_t = np.load(f)
            invQFI_t = np.load(f)
            expSz = np.load(f)
            squ1_t_0 = np.load(f)
            
        array_diff[j] = np.max(squ1_t - invQFI_t)
        
    fig, ax = plt.subplots(1, 1, figsize=(5,4))
    plt.plot(array_gamma,np.log(array_diff),'k-o')
    ax = plt.gca()
    # ax.set_ylim([-0.05,1.05])
    # ax.set_xlim([-0.05,1.05])
    plt.savefig(f'squeezing_max_diff_N{round(N)}.png')

#####################################
# test_squeezing_diff_maximum()
save_squeezing_diff_maximum()
# plot_squeezing_diff_maximum()

