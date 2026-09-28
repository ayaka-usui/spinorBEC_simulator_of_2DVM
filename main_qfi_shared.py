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
from matplotlib.colors import (Normalize, ColorConverter)

from numpy import linalg as LA
from pytictoc import TicToc
# from qutip.sparse import sp_eigs
from scipy.linalg import eig

from matplotlib import ticker

####################### for Fig. 2

def plot_energyspectrum(N, ind_data, gamma0, gamma1, M):

    base,dim = generate_base_m0(N)
    [N0,Y,Np,Nm,Jx_2,Qzx_2,Dxy_2,N0_2,Jz_2,Y_2,Qxy_2,Jy_2,Qyz_2,Np_2,Nm_2,op,op_conj] = Operators.operators_m0(base, N)
        
    # Hamiltonian
    array_gamma = np.linspace(gamma0, gamma1, M)
    # 0.2 for ciritical gamma
    
    Nex = dim
    # array_E = np.zeros([dim,M])
    array_E = np.zeros([Nex,M])

    for j in tqdm(range(M)):

        gamma = array_gamma[j]
        H = -(1-gamma)*N0 - gamma/N*(1j*N0*np.pi/2).expm()*(Jx_2 + Jy_2 + Jz_2)*(-1j*N0*np.pi/2).expm()
        evals = qt.Qobj.eigenenergies(H)
        array_E[:,j] = evals[0:Nex]
        array_E[:,j] = array_E[:,j] - array_E[0,j]*np.ones(Nex)
    
    # evals, ekets = qt.Qobj.eigenstates(H)
    # evals, evecs = sp_eigs(H.data, H.isherm) # sparse=sparse,sort=sort, eigvals=eigvals, tol=tol, maxiter=maxiter)
    # evecs = evecs.T

    mpl.rcParams.update({'font.size': 24})
    fig, ax = plt.subplots(layout='constrained')
    plt.plot(array_gamma,array_E.transpose()/N,'k-')
    ax = plt.gca()
    ax.set_xlim([-0.01,1.01])
    ax.set_ylim([-0.01,1.01])
    plt.axvline(x = 0.2, color = 'r', linestyle = '--', label = 'axvline')
    # ax.axis('equal')
    ax.set_aspect('equal', 'box')
    # ax.set_ylim(bottom=0)
    plt.savefig(f'test_spectrum.png')    

# plot_energyspectrum(100, 0, 0.0, 0.99, 100)

####################### for Figs. 4, 7, 9

def generate_base(N):
    dim = int((N + 1) * (N + 2) / 2)

    ## mode +, 0, -
    base = np.zeros([dim, 3],dtype=int)
    m = 0
    for k in range(N+1):
        for j in range(N-k+1): # (+1, 0 , -1)
            base[m,0] = k
            base[m,1] = j
            base[m,2] = N-k-j
            m+=1

    return base, dim

def transform_toxy_frompm_modex0(base,dim,N):
    
    mat = np.zeros([N+1,dim])
    coeff1 = 0

    for j in range(dim):
        
        nx = base[j, 0]
        n0 = base[j, 1]
        # ny = base[j, 2]

        if nx + n0 != N:
            continue
        
        coeff1 = (-1)**nx
        
        for sx in range(nx + 1):

            if sx <= np.floor(nx/2):
                logf = (sumlog(nx-sx+1,nx) - sumlog(1,sx))/2 - (nx/2)*np.log(2)
            else: # sx > np.floor(nx/2)
                logf = (sumlog(sx+1,nx) - sumlog(1,nx-sx))/2 - (nx/2)*np.log(2)
                    
            for m in range(dim):
                if base[m, 0] == nx-sx and base[m, 1] == n0 and base[m, 2] == sx:
                    mat[n0, m] += coeff1 * (-1)**sx * np.exp(logf)
                    break
    
    return mat

def define_base_operators_transform(N,ind_data):

    if ind_data == 0:

        # base
        base, dim = generate_base(N)
        # operators
        [Jx,Qzx,Dxy,N0,Jz,Y,Qxy,Jy,Qyz,Np,Nm]=Operators.operators(base, dim)
        #
        mat_toxy_sub = qt.Qobj(transform_toxy_frompm_modex0(base,dim,N)) # unitary transformation to mode x, 0, y in the symmetric subspace

    else: # ind_data == 1

        with open(f'base_operator_N{round(N)}.npy', 'rb') as f:
            base = np.load(f)
            Jx = np.load(f)
            Qzx = np.load(f)
            Dxy = np.load(f)
            N0 = np.load(f)
            Jz = np.load(f)
            Y = np.load(f)
            Qxy = np.load(f)
            Jy = np.load(f)
            Qyz = np.load(f)
            Np = np.load(f)
            Nm = np.load(f)

        with open(f'transform_N{round(N)}.npy', 'rb') as f:
            mat_toxy_sub = np.load(f)

        dim = int((N + 1) * (N + 2) / 2)
        Jx = qt.Qobj(Jx)
        Qzx = qt.Qobj(Qzx)
        Dxy = qt.Qobj(Dxy)
        N0 = qt.Qobj(N0)
        Jz = qt.Qobj(Jz)
        Y = qt.Qobj(Y)
        Qxy = qt.Qobj(Qxy)
        Jy = qt.Qobj(Jy)
        Qyz = qt.Qobj(Qyz)
        Np = qt.Qobj(Np)
        Nm = qt.Qobj(Nm)
        mat_toxy_sub = qt.Qobj(mat_toxy_sub)

    return base,dim,Jx,Qzx,Dxy,N0,Jz,Y,Qxy,Jy,Qyz,Np,Nm,mat_toxy_sub

def coherent_2(x, y, dim, base, N):
    
    coh = np.zeros(dim, dtype=complex)
    coeff1 = 0j
    coeff2 = 0.0
    logf1 = 0.0
    logf2 = 0.0
    
    for k in range(N + 1):
        for j in range(k + 1):
        
            coeff1 = x**(k-j) * y**j * 1j**j * (-1)**(k-j)
            logf1 = (1/2)*sumlog(N-k+1,N) - k/2 * np.log(2)

            for sy in range(j + 1):
                for sx in range(k - j + 1):
                
                    coeff2 = (-1)**sx
                    logf2 = 1/2*sumlog(1,k-sx-sy) - sumlog(1,j-sy) - sumlog(1,k-j-sx) + 1/2*sumlog(1,sx+sy) - sumlog(1,sx) - sumlog(1,sy)
                                        
                    for m in range(dim):
                        if base[m, 0] == k - sx - sy and base[m, 1] == N - k and base[m, 2] == sx + sy:
                            coh[m] += coeff1 * coeff2 * np.exp(logf1+logf2)
                            break
                        
    return coh / (np.sqrt(1+x**2+y**2))**N
    # return qt.Qobj(coh / np.linalg.norm(coh))
    # return coh / np.linalg.norm(coh) 

def wigner_cmap_0(W, levels=1024, shift=0, max_color='#09224F',
                mid_color='#FFFFFF', min_color='#530017',
                neg_color='#FF97D4', invert=False):
    """A custom colormap that emphasizes negative values by creating a
    nonlinear colormap.

    Parameters
    ----------
    W : array
        Wigner function array, or any array.
    levels : int
        Number of color levels to create.
    shift : float
        Shifts the value at which Wigner elements are emphasized.
        This parameter should typically be negative and small (i.e -1e-5).
    max_color : str
        String for color corresponding to maximum value of data.  Accepts
        any string format compatible with the Matplotlib.colors.ColorConverter.
    mid_color : str
        Color corresponding to zero values.  Accepts any string format
        compatible with the Matplotlib.colors.ColorConverter.
    min_color : str
        Color corresponding to minimum data values.  Accepts any string format
        compatible with the Matplotlib.colors.ColorConverter.
    neg_color : str
        Color that starts highlighting negative values.  Accepts any string
        format compatible with the Matplotlib.colors.ColorConverter.
    invert : bool
        Invert the color scheme for negative values so that smaller negative
        values have darker color.

    Returns
    -------
    Returns a Matplotlib colormap instance for use in plotting.

    Notes
    -----
    The 'shift' parameter allows you to vary where the colormap begins
    to highlight negative colors. This is beneficial in cases where there
    are small negative Wigner elements due to numerical round-off and/or
    truncation.

    """
    cc = ColorConverter()
    max_color = np.array(cc.to_rgba(max_color), dtype=float)
    mid_color = np.array(cc.to_rgba(mid_color), dtype=float)
    if invert:
        min_color = np.array(cc.to_rgba(neg_color), dtype=float)
        neg_color = np.array(cc.to_rgba(min_color), dtype=float)
    else:
        min_color = np.array(cc.to_rgba(min_color), dtype=float)
        neg_color = np.array(cc.to_rgba(neg_color), dtype=float)
    
    # get min and max values from Wigner function
    # bounds = [W.min(), W.max()]
    bounds = [-1.1246918935307402, 3.989214701805316] ##############

    # create empty array for RGBA colors
    adjust_RGBA = np.hstack((np.zeros((levels, 3)), np.ones((levels, 1))))
    zero_pos = int(np.round(levels * np.abs(shift - bounds[0])
                        / (bounds[1] - bounds[0])))
    num_pos = levels - zero_pos
    num_neg = zero_pos - 1
    # set zero values to mid_color
    adjust_RGBA[zero_pos] = mid_color
    # interpolate colors
    for k in range(0, levels):
        if k < zero_pos:
            interp = k / (num_neg + 1.0)
            adjust_RGBA[k][0:3] = (1.0 - interp) * \
                min_color[0:3] + interp * neg_color[0:3]
        elif k > zero_pos:
            interp = (k - zero_pos) / (num_pos + 1.0)
            adjust_RGBA[k][0:3] = (1.0 - interp) * \
                mid_color[0:3] + interp * max_color[0:3]
    # create colormap
    wig_cmap = mpl.colors.LinearSegmentedColormap.from_list('wigner_cmap',
                                                            adjust_RGBA,
                                                            N=levels)
    return wig_cmap

def wigner_cmap_2(levels=1024, shift=0, max_color='#09224F',
                mid_color='#FFFFFF', min_color='#530017',
                neg_color='#FF97D4', invert=False):
    """A custom colormap that emphasizes negative values by creating a
    nonlinear colormap.

    Parameters
    ----------
    W : array
        Wigner function array, or any array.
    levels : int
        Number of color levels to create.
    shift : float
        Shifts the value at which Wigner elements are emphasized.
        This parameter should typically be negative and small (i.e -1e-5).
    max_color : str
        String for color corresponding to maximum value of data.  Accepts
        any string format compatible with the Matplotlib.colors.ColorConverter.
    mid_color : str
        Color corresponding to zero values.  Accepts any string format
        compatible with the Matplotlib.colors.ColorConverter.
    min_color : str
        Color corresponding to minimum data values.  Accepts any string format
        compatible with the Matplotlib.colors.ColorConverter.
    neg_color : str
        Color that starts highlighting negative values.  Accepts any string
        format compatible with the Matplotlib.colors.ColorConverter.
    invert : bool
        Invert the color scheme for negative values so that smaller negative
        values have darker color.

    Returns
    -------
    Returns a Matplotlib colormap instance for use in plotting.

    Notes
    -----
    The 'shift' parameter allows you to vary where the colormap begins
    to highlight negative colors. This is beneficial in cases where there
    are small negative Wigner elements due to numerical round-off and/or
    truncation.

    """
    cc = ColorConverter()
    max_color = np.array(cc.to_rgba(max_color), dtype=float)
    mid_color = np.array(cc.to_rgba(mid_color), dtype=float)
    if invert:
        min_color = np.array(cc.to_rgba(neg_color), dtype=float)
        neg_color = np.array(cc.to_rgba(min_color), dtype=float)
    else:
        min_color = np.array(cc.to_rgba(min_color), dtype=float)
        neg_color = np.array(cc.to_rgba(neg_color), dtype=float)
    
    # get min and max values from Wigner function
    # bounds = [W.min(), W.max()]
    bounds = [-0.02309426262961361, 0.6356844214889492] ##############

    # create empty array for RGBA colors
    adjust_RGBA = np.hstack((np.zeros((levels, 3)), np.ones((levels, 1))))
    zero_pos = int(np.round(levels * np.abs(shift - bounds[0])
                        / (bounds[1] - bounds[0])))
    num_pos = levels - zero_pos
    num_neg = zero_pos - 1
    # set zero values to mid_color
    adjust_RGBA[zero_pos] = mid_color
    # interpolate colors
    for k in range(0, levels):
        if k < zero_pos:
            interp = k / (num_neg + 1.0)
            adjust_RGBA[k][0:3] = (1.0 - interp) * \
                min_color[0:3] + interp * neg_color[0:3]
        elif k > zero_pos:
            interp = (k - zero_pos) / (num_pos + 1.0)
            adjust_RGBA[k][0:3] = (1.0 - interp) * \
                mid_color[0:3] + interp * max_color[0:3]
    # create colormap
    wig_cmap = mpl.colors.LinearSegmentedColormap.from_list('wigner_cmap',
                                                            adjust_RGBA,
                                                            N=levels)
    return wig_cmap

def plot_sphere_W_0(P, THETA, PHI, fig=None, ax=None, figsize=(6, 6)):

    cmap = wigner_cmap_0(P)
    norm = mpl.colors.Normalize(-1.1246918935307402, 3.989214701805316)

    xx = np.sin(THETA) * np.cos(PHI)
    yy = np.sin(THETA) * np.sin(PHI)
    zz = np.cos(THETA)

    # Plot the surface
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(xx, yy, zz, rstride=1, cstride=1,
                    facecolors=cmap(norm(P)), linewidth=0)
    ax.set_box_aspect((np.ptp(xx), np.ptp(yy), np.ptp(zz)))

    cax, kw = mpl.colorbar.make_axes(ax, shrink=.66, pad=.02)
    cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
    # cb1.set_label('magnitude')

    ax.set(xticklabels=[],
           yticklabels=[],
           zticklabels=[])
    
    ax.view_init(45, 45)
    
    ax.set_axis_off()

    # ax.grid(False)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])

    # offset = 0.1
    # zsx = np.ones(np.size(phi))*0.5 #separatrix()
    # xsx = np.sqrt(1 - zsx**2)*(1+offset) * np.cos(phi)
    # ysx = np.sqrt(1 - zsx**2)*(1+offset) * np.sin(phi)
    # ax.plot(xsx, ysx, zsx, color='black', linewidth=4)

    return fig, ax 

def time_evolution_wigner(N, ind_data, gamma, Nt, ti, tf):

    time = TicToc()
    time.tic() #Start timer
    base,dim,Jx,Qzx,Dxy,N0,Jz,Y,Qxy,Jy,Qyz,Np,Nm,mat_toxy_sub = define_base_operators_transform(N,ind_data)    
    time.toc()

    Jx_spinhalf = Dxy/2
    Jy_spinhalf = Qxy/2
    Jz_spinhalf = Jz/2

    Sx_spinhalf = Jx/2
    Sy_spinhalf = Qyz/2
    Sz_spinhalf = (-np.sqrt(3)*Y - Dxy)/4

    Ax_spinhalf = Qzx/2
    Ay_spinhalf = Jy/2
    Az_spinhalf = (-np.sqrt(3)*Y + Dxy)/4

    # time
    timescale = np.linspace(ti, tf, Nt)

    #
    squ0_t = np.zeros(Nt)
    squ1_t = np.zeros(Nt)
    invQFI_t = np.zeros(Nt)

    squ1_t_0 = np.zeros(Nt)

    expSx = np.zeros(Nt)
    expSy = np.zeros(Nt)
    expSz = np.zeros(Nt)

    expAx = np.zeros(Nt)
    expAy = np.zeros(Nt)
    expAz = np.zeros(Nt)
    expA2 = np.zeros(Nt)

    # initial state 
    # 0N0
    # init = qt.Qobj(np.array([1 if base[i][1]==N else 0 for i in range(dim)]))
    # init = init.unit()
    # bended
    r0 = bend_degree(gamma)
    init = qt.Qobj(coherent_2(r0, 0.0, dim, base, N))
    # init = qt.Qobj(coherent_2(1.0, 0.0, dim, base, N))
    # init = qt.Qobj(coherent_2(0.0, 0.0, dim, base, N))

    normx_t = np.zeros(Nt)
    pop_fock_end_t = np.zeros(Nt)
    var_modex_t = np.zeros(Nt)
    N0_t = np.zeros(Nt)
    X_t = np.zeros(Nt)

    # wigner
    Nx = 200
    xvec = np.linspace(-6,6,Nx)
    dx = xvec[1]-xvec[0]
    # var_W_x_t = np.zeros(Nt)
    # X, PX = np.meshgrid(xvec, xvec)

    X = (qt.destroy(N+1) + qt.create(N+1)) / 2
    P = (qt.destroy(N+1) - qt.create(N+1)) / (2*1j)

    # spin husimi and spin wigner
    theta = np.linspace(0,np.pi,51)
    phi = np.linspace(0,2*np.pi,101)

    #
    Ncoherent = 31
    X0 = np.linspace(-1.5,1.5,Ncoherent)
    Y0 = np.linspace(-1.5,1.5,Ncoherent)

    # Hamiltonian for evolution
    # gamma = 0.9 #0.8 #0.2 #0
    # 0.2 for ciritical gamma
    H = -(1-gamma)*N0 - gamma/N*(1j*N0*np.pi/2).expm()*(Jx*Jx + Jy*Jy + Jz*Jz)*(-1j*N0*np.pi/2).expm()

    # time evolution operator
    time.tic()
    evals, evecs = eig(H.full())
    evals = evals.real
    time.toc()
        
    for m in tqdm(range(0, Nt)):

        # time evolution
        t = timescale[m]
        # u_evo = (-1j * H * t).expm()
        u_evo = qt.Qobj(evecs@(np.diag(np.exp(-1j*evals*t))@evecs.conj().T))
        state = u_evo * init
    
        psi = qt.Qobj(np.flipud(qt.Qobj.full(mat_toxy_sub*state)))
        normx_t[m] = qt.Qobj.norm(psi)
        psi_renorm = psi/qt.Qobj.norm(psi)

        # wigner function on phase space
        W_x_fun = qt.wigner(psi_renorm, xvec, xvec,g=2)

        # wigner function on 
        spin_W_fun, Theta_W, Phi_W = qt.spin_wigner(qt.ket2dm(psi_renorm), theta, phi)

        # plot
        mpl.rcParams.update({'font.size': 32})

        fig, ax = plt.subplots(layout='constrained')
        # wmap = qt.wigner_cmap(W_x_fun)
        # wmap = wigner_cmap_1()
        wmap = wigner_cmap_2()
        # levels = np.linspace(-0.31060881323359246, 0.6349265242391664,100)
        levels = np.linspace(-0.02309426262961361, 0.6356844214889492,100)
        cs = ax.contourf(xvec, xvec, W_x_fun, cmap=wmap, levels=levels)
        cb = fig.colorbar(cs, ax=ax)
        tick_locator = ticker.MaxNLocator(nbins=5)
        cb.locator = tick_locator
        cb.update_ticks()
        plt.xticks(np.arange(-5, 6, step=5))
        plt.yticks(np.arange(-5, 6, step=5))
        ax.set_aspect('equal')
        plt.savefig(f'wigner_phase_test_N{round(N)}_{round(m)}.png')

        print(W_x_fun.max())
        print(W_x_fun.min())

        # plot spin husimi
        plot_sphere_W_0(spin_W_fun, Theta_W, Phi_W)
        # plt.show()
        plt.savefig(f'wigner_spin_test_N{round(N)}_{round(m)}.png')
        
        print(spin_W_fun.max())
        print(spin_W_fun.min())

# time_evolution_wigner(50, 0, 0.3, 3, 0.0, 2*np.pi)

####################### for Figs. 5, 6

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
    Hdata = H.full()

    # prepare for time evolution operator
    # evals, evecs = sp_eigs(H.data, H.isherm) # sparse=sparse,sort=sort, eigvals=eigvals, tol=tol, maxiter=maxiter)
    # evecs = evecs.T
    evals, evecs = eig(H.full())
    evals = evals.real
   
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

####################### for Figs. 5, 6

def test_fig():

    N = 2000 #500
    gamma = 0.3 #0.190
    Nt = 1001 #101
    tf = 3000.0 #1000.0

    timescale, squ1_t, invQFI_t, expSz, squ1_t_0 = time_evolution_m0(N,gamma,Nt,0.0,tf)

    plt.plot(timescale,np.log(squ1_t-invQFI_t),'k-',linewidth=3.0)
    # plt.ylim(-6,0)

    plt.savefig(f'test_N{round(N)}_xi{round(gamma,3)}_0.png')

def test_squeezing_diff_maximum():

    N = 500 #500
    gamma = 0.1 #0.190
    Nt = 10001 #101
    tf = 1000.0 #1000.0

    timescale, squ1_t, invQFI_t, expSz, squ1_t_0 = time_evolution_m0(N,gamma,Nt,0.0,tf)


    fig = plt.figure()
    axs1 = plt.subplot(211)
    axs2 = plt.subplot(212)

    axs1.plot(timescale,np.log(squ1_t-invQFI_t),'k-',linewidth=3.0)
    axs1 = plt.gca()
    axs1.axvline(x = timescale[np.argmax(squ1_t-invQFI_t)], color = 'r', linestyle = '--', label = 'axvline')
    axs1.set_ylim([-6.0,0.0])
    
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

    array_N = np.array([3000,4000,5000])
    array_gamma = np.linspace(0.1, 0.2, 11)

    # array_N = np.array([1500])
    # # array_N = np.linspace(6000, 10000, 5)

    # # array_gamma = np.linspace(0.25, 0.3, 6)
    # # array_gamma = np.linspace(0.199, 0.22, 22)
    # array_gamma = np.linspace(0.18, 0.22, 41)
    # # array_gamma = np.linspace(0.14, 0.17, 4)

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

            with open(f'squeezing_QFI_N{round(N)}_xi{round(gamma,3)}_0.npy', 'wb') as f:
                np.save(f, timescale)
                np.save(f, N)
                np.save(f, gamma)
                np.save(f, squ1_t)
                np.save(f, invQFI_t)
                np.save(f, expSz)
                np.save(f, squ1_t_0)
            
            np.savetxt(f'array_diff_QFI_N{round(N)}_xi{round(gamma,3)}_0.txt', squ1_t - invQFI_t)

def plot_squeezing_diff_maximum():

    N = 2000

    array_gamma = np.linspace(0.1, 0.3, 21) #np.linspace(0.2, 0.3, 11)
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
    plt.savefig(f'squeezing_max_diff_N{round(N)}_0.png')

# test_fig()
# test_squeezing_diff_maximum()
# save_squeezing_diff_maximum()
# plot_squeezing_diff_maximum()
