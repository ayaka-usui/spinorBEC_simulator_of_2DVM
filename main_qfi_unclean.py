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

def delta(k1,k2):

    if k1 == k2:
        return 1
    else: #k1 =! k2
        return 0

def moving_average(vec,Nt,N0):

    vec_new = np.zeros(Nt-N0+1)

    for j in range(Nt-N0+1):
        vec_new[j] = np.mean(vec[j:j+N0-1])

    return vec_new

def bend_degree(gamma):

    if gamma <= 0.2:
        return 0.0
    else:
        return np.sqrt((5*gamma-1)/(3*gamma+1))

def spin_coherent_spin1(theta, phi, Jy, Jz, state_0):
    return (-1j * phi * Jz).expm() * (-1j * theta * Jy).expm() * state_0
    
def sumlog(N0, N):
    
    result = 0.0

    for n in range(N0,N+1):
        result += np.log(n)
    
    return result

def transform_toxy_frompm_modex0_m0(dim,N):

    # dim=int(N/2+1)
    mat = np.zeros([N+1,dim])

    for nx in range(N+1):

        if np.mod(nx,2) == 1:
            continue
        halfnx = int(nx/2)

        logf = (sumlog(halfnx+1,nx) - sumlog(1,halfnx))/2 - halfnx*np.log(2)
        mat[N-nx, halfnx] = (-1)**halfnx * np.exp(logf)
    
    return mat

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

def transform_toxy_frompm(base,dim):
    
    mat = np.zeros([dim,dim],dtype="complex")
    coeff1 = 0.0j
    coeff2 = 0.0

    for j in tqdm(range(dim)):
        
        nx = base[j, 0]
        n0 = base[j, 1]
        ny = base[j, 2]
        
        logf1 = -(nx+ny)*np.log(2)/2 + sumlog(1,nx)/2 + sumlog(1,ny)/2
        
        coeff1 = (-1)**nx *(1j)**ny
        
        for sx in range(nx + 1):

            logfx = - sumlog(1,sx) - sumlog(1,nx-sx)
            px = (-1)**sx

            for sy in range(ny + 1):
                
                logf = logf1 + logfx - sumlog(1,sy) - sumlog(1,ny-sy) + sumlog(1,nx+ny-sx-sy)/2 + sumlog(1,sx+sy)/2 # order of the terms affect the precision therefore this is fine only for small N
                coeff2 = px * np.exp(logf)
                    
                for m in range(dim):
                    if base[m, 0] == nx+ny-sx-sy and base[m, 1] == n0 and base[m, 2] == sx+sy:
                        mat[j, m] += coeff1*coeff2
                        break
                        
    return mat

def transform_toxy_frompm_old(base,dim):
    
    mat = np.zeros([dim,dim],dtype="complex")
    coeff1 = 0.0j
    coeff2 = 0.0
    
    for j in range(dim):
        
        nx = base[j, 0]
        n0 = base[j, 1]
        ny = base[j, 2]
        
        coeff1 = np.sqrt(1/np.math.factorial(nx)/np.math.factorial(ny)/(2**(nx+ny))) *(-1)**nx *(1j)**ny
        
        for sx in range(nx + 1):
            for sy in range(ny + 1):
                
                coeff2 = binom(nx, sx) * binom(ny, sy) * (-1)**sx * np.sqrt(np.math.factorial(nx+ny-sx-sy)*np.math.factorial(sx+sy)*1.0)
                # "*1.0" is for avoinding an error where sqrt does not like big integers  
                    
                for m in range(dim):
                    if base[m, 0] == nx+ny-sx-sy and base[m, 1] == n0 and base[m, 2] == sx+sy:
                        mat[j, m] += coeff1*coeff2
                        break
                        
    return mat

def transform_tosa_fromxy(base,dim):
    
    mat = np.zeros([dim,dim],dtype="complex")
    
    for j in tqdm(range(dim)):
        
        ns = base[j, 0]
        n0 = base[j, 1]
        na = base[j, 2]

        for m in range(dim):
            if base[m, 0] == na and base[m, 1] == n0 and base[m, 2] == ns:
                mat[j, m] = (-1j)**ns * (-1)**na
                break

    return mat

def transform_tosa_frompm(base,dim):
    
    mat = np.zeros([dim,dim])
    coeff1 = 0.0j
    coeff2 = 0.0
    
    for j in range(dim):
        
        ns = base[j, 0]
        n0 = base[j, 1]
        na = base[j, 2]
        
        coeff1 = np.sqrt(1/np.math.factorial(ns)/np.math.factorial(na)/2**(ns+na))
        
        for ss in range(ns + 1):
            for sa in range(na + 1):
                
                coeff2 = binom(ns, ss) * binom(na, sa) * (-1)**sa * np.sqrt(np.math.factorial(ns+na-ss-sa)*np.math.factorial(ss+sa)*1.0)
                    
                for m in range(dim):
                    if base[m, 0] == ns+na-ss-sa and base[m, 1] == n0 and base[m, 2] == ss+sa:
                        mat[j, m] += coeff1*coeff2
                        break
                        
    return mat
    
def transform_to2mode(base,dim,N,ind):

    mat = np.zeros([N+1,dim])

    for j in range(dim):
    
        ns = base[j, 0]
        n0 = base[j, 1]
        na = base[j, 2]
    
        if ind == 1:
            if ns + n0 == N:
                mat[n0,j] = 1
        elif ind == -1:
            if na + n0 == N:
                mat[n0,j] = 1

    return mat
    
def transform_to1mode(base,dim,N,rho,ind):

    rho_reduced = np.zeros([N+1,N+1])

    for j in range(N+1):
    
        for m in range(dim):
            if j == base[m, ind]:
                rho_reduced[j,j] += rho[m,m].real
                
    rho_reduced = qt.Qobj(rho_reduced)
        
    return rho_reduced

def coherent(x, y, dim, base, N):
    
    coh = np.zeros(dim, dtype=complex)
    coeff1 = 0j
    coeff2 = 0
    
    for k in range(N + 1):
        for j in range(k + 1):
        
            coeff1 = binom(N, k) * binom(k, j) * x**(k-j) * y**j * 1j**j * (-1)**(k-j) / (np.sqrt(2)**k)
        
            for sy in range(j + 1):
                for sx in range(k - j + 1):
                
                    coeff2 = binom(j, sy) * binom(k-j, sx) * (-1)**sx * np.sqrt(np.math.factorial(k-sx-sy)*np.math.factorial(N-k)*np.math.factorial(sx+sy)*1.0)
                    
                    for m in range(dim):
                        if base[m, 0] == k - sx - sy and base[m, 1] == N - k and base[m, 2] == sx + sy:
                            coh[m] += coeff1 * coeff2
                            break
                        
    # return qt.Qobj(coh / np.linalg.norm(coh))
    return coh / np.linalg.norm(coh) 

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

def husimi_xy_1(x_tab, y_tab, dim, base, N):

    X, Y = np.meshgrid(x_tab, y_tab)
    coh_xy = np.zeros((dim,len(x_tab),len(y_tab)), dtype=complex)
    
    for i in range(len(x_tab)):
        for j in range(len(y_tab)):
            coh_xy[:,i,j] = coherent(x_tab[i], y_tab[j], dim, base, N)
            
    return coh_xy

def husimi_xy_2(state, x_tab, y_tab, coh_xy):
    
    X, Y = np.meshgrid(x_tab, y_tab)
    q_tab = np.zeros_like(X)
    for i in range(len(x_tab)):
        for j in range(len(y_tab)):
            coh = qt.Qobj(coh_xy[:,i,j])
            q_tab[j, i] =  np.abs(coh.overlap(state))**2 #(coh.trans().conj() * qt.ket2dm(state) * coh)[0, 0]
    return q_tab

def xyaxis_spin_distribution_2d(THETA, PHI):

    Y = (THETA - np.pi / 2) / (np.pi / 2)
    X = (np.pi - PHI) / np.pi * np.sqrt(np.cos(THETA - np.pi / 2))

    return X, Y

def plot_sphere(P, THETA, PHI, fig=None, ax=None, figsize=(6, 6)):

    if fig is None or ax is None:
        fig = plt.figure(figsize=figsize)
        ax = Axes3D(fig, azim=-35, elev=35)
   
    cmap = cm.coolwarm #cm.copper #cm.PuBu #cm.RdYlBu
    norm = mpl.colors.Normalize(0.0, P.max())

    # if P.min() < -1e12:
    #     cmap = cm.RdBu
    #     norm = mpl.colors.Normalize(-P.max(), P.max())
    # else:
    #     cmap = cm.RdYlBu
    #     norm = mpl.colors.Normalize(P.min(), P.max())

    xx = np.sin(THETA) * np.cos(PHI)
    yy = np.sin(THETA) * np.sin(PHI)
    zz = np.cos(THETA)

    # Plot the surface
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"}) #rstride=1, cstride=1, rcount=200, ccount=200,
    ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, 
                    facecolors=cmap(norm(P)), linewidth=0, shade=False)
    
    cax, kw = mpl.colorbar.make_axes(ax, shrink=.66, pad=.02)
    cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
    # cb1.set_label('magnitude')

    ax.set(xticklabels=[],
           yticklabels=[],
           zticklabels=[])
    
    ax.view_init(90, 0)
    
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

def plot_sphere_W(P, THETA, PHI, fig=None, ax=None, figsize=(6, 6)):

    if fig is None or ax is None:
        fig = plt.figure(figsize=figsize)
        ax = Axes3D(fig, azim=-35, elev=35)
   
    # if P.min() < -1e12:
    #     cmap = cm.RdBu
    #     norm = mpl.colors.Normalize(-P.max(), P.max())
    # else:
    #     cmap = cm.RdYlBu
    #     norm = mpl.colors.Normalize(P.min(), P.max())

    cmap = qt.wigner_cmap(P)
    norm = mpl.colors.Normalize(P.min(), P.max())

    xx = np.sin(THETA) * np.cos(PHI)
    yy = np.sin(THETA) * np.sin(PHI)
    zz = np.cos(THETA)

    # Plot the surface
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(xx, yy, zz, rstride=1, cstride=1,
                    facecolors=cmap(norm(P)), linewidth=0)
    
    cax, kw = mpl.colorbar.make_axes(ax, shrink=.66, pad=.02)
    cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
    # cb1.set_label('magnitude')

    ax.set(xticklabels=[],
           yticklabels=[],
           zticklabels=[])
    
    ax.view_init(90, 0)
    
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

def wigner_cmap_1(levels=1024, shift=0, max_color='#09224F',
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
    bounds = [-0.31060881323359246, 0.6349265242391664] ##############

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

def plot_sphere_W_0(P, THETA, PHI, fig=None, ax=None, figsize=(6, 6)):

    # if fig is None or ax is None:
    #     fig = plt.figure(figsize=figsize)
    #     ax = Axes3D(fig, azim=-35, elev=35)
   
    # if P.min() < -1e12:
    #     cmap = cm.RdBu
    #     norm = mpl.colors.Normalize(-P.max(), P.max())
    # else:
    #     cmap = cm.RdYlBu
    #     norm = mpl.colors.Normalize(P.min(), P.max())

    # cmap = qt.wigner_cmap(P)
    # norm = mpl.colors.Normalize(P.min(), P.max())

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

def QFI_covariance_matrix_pure(state,Jx, Qyz, Jy, Qzx, Dxy, Qxy, Y, Jz):
    
    COV = np.zeros([8,8])
    COV[0,0] = qt.expect(Jx*Jx, state) #mean_value_rho(rho, Jx*Jx)
    COV[1,1] = qt.expect(Qyz*Qyz, state) #mean_value_rho(rho, Qyz*Qyz)
    COV[2,2] = COV[0,0]
    COV[3,3] = COV[1,1]
    COV[4,4] = qt.expect(Dxy*Dxy, state) #mean_value_rho(rho, Dxy*Dxy)
    COV[5,5] = COV[4,4]
    COV[6,6] = qt.expect(Y*Y, state) - qt.expect(Y, state)**2 #mean_value_rho(rho, Y*Y) - mean_value_rho(rho, Y)**2
    # COV[0,1] = COV[1,0] = mean_value_rho(rho, 1/2*(Jx*Qyz+Qyz*Jx))-mean_value_rho(rho, Qyz)*mean_value_rho(rho, Jx)
    COV[0,1] = qt.expect(1/2*(Jx*Qyz+Qyz*Jx), state) - qt.expect(Qyz, state)*qt.expect(Jx, state)
    COV[1,0] = COV[0,1]
    # COV[2,3] = COV[3,2] = -COV[1,0]
    COV[2,3] = -COV[1,0]
    COV[3,2] = COV[2,3]

    #
    eigenvalues, eigenvectors = LA.eig(COV)

    return max(eigenvalues)

def covariance_matrix_spinhalf(state,N,Sx,Sy):
    
    COV = np.zeros([2,2])
    COV[0,0] = qt.expect(Sx*Sx+Sx*Sx, state)*(2/N)
    COV[1,1] = qt.expect(Sy*Sy+Sy*Sy, state)*(2/N)
    COV[1,0] = qt.expect(Sx*Sy+Sy*Sx, state)*(2/N)
    COV[0,1] = COV[1,0]
    
    #
    Delta = (COV[0,0] - COV[1,1])**2 + (2*COV[1,0])**2
    lambda_minus = (COV[0,0] + COV[1,1] - np.sqrt(Delta))/2
    lambda_plus = (COV[0,0] + COV[1,1] + np.sqrt(Delta))/2

    return lambda_minus, lambda_plus

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

def var_mode_phase(X,P,state):

    COV = np.zeros([2,2])
    COV[0,0] = qt.expect(X*X+X*X, state)/2
    COV[1,1] = qt.expect(P*P+P*P, state)/2
    COV[1,0] = qt.expect(X*P+P*X, state)/2
    COV[0,1] = COV[1,0]

    eigenvalues, eigenvectors = LA.eig(COV)

    return max(eigenvalues)

def var_mode_phase_wigner(X,P,W_x_fun,dx):

    COV = np.zeros([2,2])
    COV[0,0] = 2*((X*X)*W_x_fun).sum()*dx*dx
    COV[1,1] = 2*((P*P)*W_x_fun).sum()*dx*dx
    COV[1,0] = ((X*P+P*X)*W_x_fun).sum()*dx*dx
    COV[0,1] = COV[1,0]

    eigenvalues, eigenvectors = LA.eig(COV)

    return max(eigenvalues)

def part1_QFI(ind,k1,k2):

    if ind == 0:
        return ( np.sqrt(k2+1)*delta(k1,k2+1) + np.sqrt(k2)*delta(k1,k2-1) )/2
    elif ind == 1:
        return (-np.sqrt(k2+1)*delta(k1,k2+1) + np.sqrt(k2)*delta(k1,k2-1) )/(2*1j)

def QFI_onemode_element(vecp,ind0,ind1):

    vecp_size = np.size(vecp)
    QFI = 0.0*1j

    for k1 in range(vecp_size):
        for k2 in range(vecp_size):
            if vecp[k1]+vecp[k2] != 0.0:
                QFI += (vecp[k1]-vecp[k2])**2 / (vecp[k1]+vecp[k2]) * part1_QFI(ind0,k1,k2) * part1_QFI(ind1,k2,k1)
    
    return QFI*2

def QFI_onemode(rho):

    # rho is a diagonal matrix
    vecp = np.diag(qt.Qobj.full(rho)).real

    MQFI = np.zeros([2,2],dtype=complex)
    MQFI[0,0] = QFI_onemode_element(vecp,0,0)
    MQFI[1,1] = QFI_onemode_element(vecp,1,1)
    MQFI[0,1] = QFI_onemode_element(vecp,0,1)
    MQFI[1,0] = MQFI[0,1]

    eigenvalues, eigenvectors = LA.eig(MQFI)

    return np.real(max(eigenvalues))

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

def save_transform(N):
    
    time = TicToc()
    time.tic() #Start timer

    # base
    base, dim = generate_base(N)

    mat_toxy_sub = qt.Qobj(transform_toxy_frompm_modex0(base,dim,N)) # unitary transformation to mode x, 0, y in the symmetric subspace
    
    # mat_tosa_fromxy = qt.Qobj(transform_tosa_fromxy(base,dim)) # unitary transformation to mode s, 0, a from mode x, 0, y

    # these do not take time
    # mat_S_N = qt.Qobj(transform_to2mode(base,dim,N,1)) # projection to symmetric subspace with total spin N
    # mat_A_N = qt.Qobj(transform_to2mode(base,dim,N,-1)) # projection to anti-symmetric subspace with total spin N

    with open(f'transform_N{round(N)}.npy', 'wb') as f:
        np.save(f, mat_toxy_sub)
        # np.save(f, mat_tosa_fromxy)
        # np.save(f, mat_S_N)
        # np.save(f, mat_A_N)

    time.toc()

def save_base_operators(N):
    
    time = TicToc()
    time.tic() #Start timer

    # base
    base, dim = generate_base(N)

    # operators
    [Jx,Qzx,Dxy,N0,Jz,Y,Qxy,Jy,Qyz,Np,Nm]=Operators.operators(base, dim)

    with open(f'base_operator_N{round(N)}.npy', 'wb') as f:
        np.save(f, base)
        # np.save(f, dim)
        np.save(f, Jx)
        np.save(f, Qzx)
        np.save(f, Dxy)
        np.save(f, N0)
        np.save(f, Jz)
        np.save(f, Y)
        np.save(f, Qxy)
        np.save(f, Jy)
        np.save(f, Qyz)
        np.save(f, Np)
        np.save(f, Nm)

    time.toc()

# save_transform(60)
# save_base_operators(60)

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

def time_evolution_subspace(N, ind_data, gamma, Nt, ti, tf):

    time = TicToc()
    time.tic() #Start timer
    base,dim,Jx,Qzx,Dxy,N0,Jz,Y,Qxy,Jy,Qyz,Np,Nm,mat_toxy_sub = define_base_operators_transform(N,ind_data)
    time.toc()
    
    # Jx_spinhalf = Dxy/2
    # Jy_spinhalf = Qxy/2
    # Jz_spinhalf = Jz/2

    Sx_spinhalf = Jx/2
    Sy_spinhalf = Qyz/2
    Sz_spinhalf = (-np.sqrt(3)*Y - Dxy)/4

    # Ax_spinhalf = Qzx/2
    # Ay_spinhalf = Jy/2
    # Az_spinhalf = (-np.sqrt(3)*Y + Dxy)/4

    # time
    timescale = np.linspace(ti, tf, Nt)

    #
    squ0_t = np.zeros(Nt)
    squ1_t = np.zeros(Nt)
    invQFI_t = np.zeros(Nt)
    squ0_sub_t = np.zeros(Nt)
    squ1_sub_t = np.zeros(Nt)
    invQFI_sub_t = np.zeros(Nt)

    Nx = 200
    xvec = np.linspace(-5,5,Nx)

    X = (qt.destroy(N+1) + qt.create(N+1)) / 2
    P = (qt.destroy(N+1) - qt.create(N+1)) / (2*1j)
    var_modex_t = np.zeros(Nt)
    var_modex_t_sub = np.zeros(Nt)

    # initial state 
    # 0N0
    init = qt.Qobj(np.array([1 if base[i][1]==N else 0 for i in range(dim)]))
    # bended
    # init = qt.Qobj(coherent(np.sqrt(0.0), np.sqrt(0.0), dim, base, N))
    normx_t = np.zeros(Nt)
    
    # Hamiltonian for evolution
    # 0.2 for ciritical gamma
    H = -(1-gamma)*N0 + gamma/N*(Jx*Jx + Jy*Jy + Jz*Jz)

    # time evolution operator
    time.tic()
    # evals, ekets = qt.Qobj.eigenstates(H)
    # evals, evecs = sp_eigs(H.data, H.isherm) # sparse=sparse,sort=sort, eigvals=eigvals, tol=tol, maxiter=maxiter)
    # evecs = evecs.T
    evals, evecs = eig(H.full())
    evals = evals.real
    time.toc()

    #
    H_sub = mat_toxy_sub*H*mat_toxy_sub.trans()
    init_sub = mat_toxy_sub*init
    Sx_spinhalf_sub = mat_toxy_sub*Sx_spinhalf*mat_toxy_sub.trans()
    Sy_spinhalf_sub = mat_toxy_sub*Sy_spinhalf*mat_toxy_sub.trans()
    Sz_spinhalf_sub = mat_toxy_sub*Sz_spinhalf*mat_toxy_sub.trans()
    
    for m in tqdm(range(0, Nt)):

        # time evolution
        t = timescale[m]
        # u_evo = (-1j * H * t).expm()
        u_evo = qt.Qobj(evecs@(np.diag(np.exp(-1j*evals*t))@evecs.conj().T))
        state = u_evo * init

        # squeezing and QFI
        # lambda_minus, lambda_plus = covariance_matrix_spinhalf(state,N,Sx_spinhalf,Sy_spinhalf)
        # squ0_t[m] = (qt.expect(Sx_spinhalf*Sx_spinhalf, state)-qt.expect(Sx_spinhalf, state)**2)/qt.expect(Sz_spinhalf, state)**2*N
        # squ1_t[m] = lambda_minus/qt.expect(Sz_spinhalf, state)**2*(N/2)**2
        # invQFI_t[m] = 1/lambda_plus

        #
        psi = qt.Qobj(np.flipud(qt.Qobj.full(mat_toxy_sub*state)))
        normx_t[m] = qt.Qobj.norm(psi)
        psi_renorm = psi/qt.Qobj.norm(psi)

        # subspace
        u_evo_sub = (-1j * H_sub * t).expm()
        state_sub = u_evo_sub * init_sub
        state_sub = qt.Qobj(np.flipud(qt.Qobj.full(state_sub)))

        # lambda_minus, lambda_plus = covariance_matrix_spinhalf(state_sub,N,Sx_spinhalf_sub,Sy_spinhalf_sub)
        # squ0_sub_t[m] = (qt.expect(Sx_spinhalf_sub*Sx_spinhalf_sub, state_sub)-qt.expect(Sx_spinhalf_sub, state_sub)**2)/qt.expect(Sz_spinhalf_sub, state_sub)**2*N
        # squ1_sub_t[m] = lambda_minus/qt.expect(Sz_spinhalf_sub, state_sub)**2*(N/2)**2
        # invQFI_sub_t[m] = 1/lambda_plus

        # wigner function on phase space
        W_x_fun = qt.wigner(psi_renorm, xvec, xvec,g=2)
        W_x_fun_sub = qt.wigner(state_sub, xvec, xvec,g=2)

        # variance of one mode on phase space
        var_modex_t[m] = var_mode_phase(X,P,psi_renorm)
        var_modex_t_sub[m] = var_mode_phase(X,P,state_sub)

        # plot
        mpl.rcParams.update({'font.size': 16})

        fig, ax = plt.subplots(1, 2, figsize=(8,4))
        wmap = qt.wigner_cmap(W_x_fun)
        cs = ax[0].contourf(xvec, xvec, W_x_fun, 100, cmap=wmap)
        fig.colorbar(cs, ax=ax[0])
        ax[0].set_aspect('equal')
        wmap = qt.wigner_cmap(W_x_fun_sub)
        cs = ax[1].contourf(xvec, xvec, W_x_fun_sub, 100, cmap=wmap)
        fig.colorbar(cs, ax=ax[1])
        ax[1].set_aspect('equal')
        plt.savefig(f'wigner_phase_comparision_{round(m)}.png')

    fig, ax = plt.subplots(1, 1, figsize=(4,4))
    plt.plot(timescale,var_modex_t,'k-o')
    plt.plot(timescale,var_modex_t_sub,'r--o')
    ax = plt.gca()
    # ax.set_ylim([0.0,1.0])
    ax.set_ylim(bottom=0)
    plt.savefig(f'comparision_var.png')

    fig, ax = plt.subplots(1, 1, figsize=(4,4))
    plt.plot(timescale,normx_t,'k-o')
    ax = plt.gca()
    # ax.set_ylim([0.0,1.0])
    ax.set_ylim(bottom=0)
    plt.savefig(f'norm_psix.png')

    # fig, ax = plt.subplots(1, 1, figsize=(4,4))
    # plt.plot(timescale,squ0_t,'r-o')
    # plt.plot(timescale,squ1_t,'b-o')
    # plt.plot(timescale,invQFI_t,'k-o')
    # plt.plot(timescale,squ0_sub_t,'r:o')
    # plt.plot(timescale,squ1_sub_t,'b:o')
    # plt.plot(timescale,invQFI_sub_t,'k:o')
    # ax = plt.gca()
    # # ax.set_ylim([0.0,1.0])
    # ax.set_ylim(bottom=0)
    # plt.savefig(f'squeezing_time.png')

# time_evolution_subspace(50, 1, 0.9, 11, 0.0, 4.0)

def generate_base_m0(N):

    dim=int(N/2+1)
    base=np.zeros([dim,3])
    for k in range(dim):
        base[k,0] = k
        base[k,1] = N - 2*k
        base[k,2] = k

    return base, dim

def time_evolution_m0(N, gamma, Nt, ti, tf):

    base,dim = generate_base_m0(N)
    [N0,Y,Np,Nm,Jx_2,Qzx_2,Dxy_2,N0_2,Jz_2,Y_2,Qxy_2,Jy_2,Qyz_2,Np_2,Nm_2,op,op_conj] = Operators.operators_m0(base, N)
    
    Sx_spinhalf_2 = Jx_2/4
    Sy_spinhalf_2 = Qyz_2/4
    Sz_spinhalf = (-np.sqrt(3)*Y)/4 #(-np.sqrt(3)*Y - Dxy)/4
    Sz_spinhalf_2 = (3*Y_2 + Dxy_2)/16

    mat_toxy_y0_m0 = qt.Qobj(transform_toxy_frompm_modex0_m0(dim,N))

    # time
    timescale = np.linspace(ti, tf, Nt)

    #
    squ0_t = np.zeros(Nt)
    squ1_t = np.zeros(Nt)
    invQFI_t = np.zeros(Nt)
    QFI_phase_t = np.zeros(Nt)
    normx_t = np.zeros(Nt)

    expSz = np.zeros(Nt)
    squ1_t_0 = np.zeros(Nt)
    N0_t = np.zeros(Nt)
    
    # initial state 
    # 0N0
    init = qt.Qobj(np.array([1 if base[i][1]==N else 0 for i in range(dim)]))
    
    # Hamiltonian for evolution
    # 0.2 for ciritical gamma
    # H = -(1-gamma)*N0 + gamma/N*(Jx_2 + Jy_2 + Jz_2)
    # H = -(1-gamma)/gamma*N0 + 1/N*(Jx_2 + Jy_2 + Jz_2)
    H = -(1-gamma)*N0 - gamma/N*(1j*N0*np.pi/2).expm()*(Jx_2 + Jy_2 + Jz_2)*(-1j*N0*np.pi/2).expm()

    # time evolution operator
    # evals, ekets = qt.Qobj.eigenstates(H)
    evals, evecs = sp_eigs(H.data, H.isherm) # sparse=sparse,sort=sort, eigvals=eigvals, tol=tol, maxiter=maxiter)
    evecs = evecs.T
   
    for m in tqdm(range(0, Nt)):

        # time evolution
        t = timescale[m]
        # u_evo = (-1j * H * t).expm()
        u_evo = qt.Qobj(evecs@(np.diag(np.exp(-1j*evals*t))@evecs.conj().T))
        state = u_evo * init

        # squeezing and QFI
        lambda_minus, lambda_plus = covariance_matrix_spinhalf_m0(state,N,Sx_spinhalf_2,Sy_spinhalf_2,op,op_conj)
        squ0_t[m] = (qt.expect(Sx_spinhalf_2, state))/qt.expect(Sz_spinhalf, state)**2*N
        squ1_t[m] = lambda_minus/qt.expect(Sz_spinhalf, state)**2*(N/2)**2
        invQFI_t[m] = 1/lambda_plus

        expSz[m] = qt.expect(Sz_spinhalf, state)
        squ1_t_0[m] = lambda_minus
        N0_t[m] = qt.expect(N0, state)

        #
        psi = qt.Qobj(np.flipud(qt.Qobj.full(mat_toxy_y0_m0*state)))
        normx_t[m] = qt.Qobj.norm(psi)

        # # one mode
        # rho_reduced = transform_to1mode(base,dim,N,qt.ket2dm(state),0)
        # QFI_phase_t[m] = QFI_onemode(rho_reduced)

    # plot
    mpl.rcParams.update({'font.size': 24})

    # fig, ax = plt.subplots(1, 1, figsize=(4,4))
    # plt.plot(timescale,expSz,'k-o')
    # # plt.plot(timescale,squ1_t_0,'r-*')
    # ax = plt.gca()
    # # ax.set_ylim([0.0,1.0])
    # # ax.set_ylim(bottom=0)
    # plt.savefig(f'test_m0_N{round(N)}_xi0{round(gamma*10)}.png')

    # fig, ax = plt.subplots(layout='constrained')
    # # plt.plot(timescale,normx_t,'k-o')
    # plt.plot(np.log10(timescale),normx_t,'k-o')
    # ax = plt.gca()
    # # ax.set_ylim([0.0,1.0])
    # # ax.set_ylim(bottom=0)
    # plt.savefig(f'norm_psix_m0_test_N{round(N)}_xi0{round(gamma*10)}.png')

    # fig, ax = plt.subplots(layout='constrained')
    # # plt.plot(timescale,N0_t,'k-o')
    # plt.plot(timescale*gamma,N0_t,'k-o')
    # ax = plt.gca()
    # # ax.set_ylim([0.0,1.0])
    # ax.set_ylim(bottom=0)
    # plt.savefig(f'N0_m0_N{round(N)}_xi{round(gamma,2)}.png')
    
    fig, ax = plt.subplots(layout='constrained')
    # plt.plot(timescale,squ0_t,'r--o')
    # plt.plot(timescale,squ1_t,'b--o')
    # plt.plot(timescale,invQFI_t,'k-o')
    plt.plot(timescale,np.log10(invQFI_t),'k-',linewidth=3.0)
    plt.plot(timescale,np.log10(squ1_t),'r--',linewidth=3.0)
    # plt.plot(timescale*gamma,np.log10(squ1_t-invQFI_t),'k-',linewidth=3.0)
    ax = plt.gca()
    # ax.set_ylim([0.0,1.0])
    # ax.set_ylim(bottom=0)
    plt.axvline(x = timescale[np.argmax(squ1_t-invQFI_t)], color = 'r', linestyle = '--', label = 'axvline')
    # plt.axvline(x = np.pi*2, color = 'r', linestyle = '--', label = 'axvline')
    plt.savefig(f'squeezing_time_m0_N{round(N)}_xi{round(gamma,2)}.png')
    # print(invQFI_t[Nt-1]-invQFI_t[Nt-2])
    
    fig, ax = plt.subplots(layout='constrained')
    # plt.plot(timescale,squ0_t,'r--o')
    # plt.plot(timescale,squ1_t,'b--o')
    # plt.plot(timescale,invQFI_t,'k-o')
    # plt.plot(timescale*gamma,np.log10(invQFI_t),'k-',linewidth=3.0)
    # plt.plot(timescale*gamma,np.log10(squ1_t),'r--',linewidth=3.0)
    plt.plot(timescale,np.log10(squ1_t-invQFI_t),'k-',linewidth=3.0)
    ax = plt.gca()
    # ax.set_ylim([-5.0,2.0])
    # ax.set_ylim(bottom=0)
    plt.axvline(x = timescale[np.argmax(squ1_t-invQFI_t)], color = 'r', linestyle = '--', label = 'axvline')
    # plt.axvline(x = np.pi*2, color = 'r', linestyle = '--', label = 'axvline')
    plt.savefig(f'diff_squeezing_QFI_time_m0_N{round(N)}_xi{round(gamma,2)}.png')

    # fig, ax = plt.subplots(1, 1, figsize=(5,4))
    # plt.plot(timescale[1:Nt-1],np.log10(squ1_t[1:Nt-1]-invQFI_t[1:Nt-1]),'k-o')
    # ax = plt.gca()
    # # ax.set_ylim([0.0,1.0])
    # # ax.set_ylim(bottom=0)
    # plt.savefig(f'squeezing_diff_time_m0_N{round(N)}_xi0{round(gamma*10)}.png')

    # vec0 = squ1_t[1:Nt]-invQFI_t[1:Nt]
    # vec1 = moving_average(vec0,Nt-1,int(np.floor(Nt/10)))
    # size_vec1 = np.size(vec1)

    # fig, ax = plt.subplots(1, 1, figsize=(5,4))
    # plt.plot(vec1,'k-o')
    # ax = plt.gca()
    # # ax.set_ylim([0.0,1.0])
    # # ax.set_ylim(bottom=0)
    # plt.savefig(f'squeezing_diff_moveave_m0_N{round(N)}_xi0{round(gamma*10)}.png')

    # fig, ax = plt.subplots(1, 1, figsize=(5,4))
    # plt.plot(timescale,QFI_phase_t,'k-o')
    # ax = plt.gca()
    # # ax.set_ylim([0.0,1.0])
    # # ax.set_ylim(bottom=0)
    # plt.savefig(f'QFI_phase_time_m0_N{round(N)}_xi0{round(gamma*10)}.png')

    return timescale, squ1_t, invQFI_t, N0_t, gamma, expSz, squ1_t_0
    # return squ1_t-invQFI_t

# time_evolution_m0(1000, 0.3, 101, 0.0, 10.0)
# time_evolution_m0(1000, 0.5, 101, 0.0, 30.0)

def time_evolution_appro(N, gamma, Nt, ti, tf):
    
    tau = qt.destroy(N+1)
    tau_dag = qt.create(N+1)
    nx = qt.qeye(N+1)

    Sx_spinhalf = (tau + tau_dag)/2*np.sqrt(N)
    Sy_spinhalf = (tau - tau_dag)/(2*1j)*np.sqrt(N)
    Sz_spinhalf = (N - nx)/2

    H = (-2*nx + tau**2 + tau_dag**2)*gamma
    init = qt.fock(N+1,0)

    timescale = np.linspace(ti, tf, Nt)
    squ1_t = np.zeros(Nt)
    invQFI_t = np.zeros(Nt)
    
    # result = qt.mesolve(H, vaccum, times, [], [])
    # state = result.states

    for m in tqdm(range(0, Nt)):

        t = timescale[m]
        u_evo = (-1j * H * t).expm()
        state = u_evo * init

        # squeezing and QFI
        lambda_minus, lambda_plus = covariance_matrix_spinhalf(state,N,Sx_spinhalf,Sy_spinhalf)
        squ1_t[m] = lambda_minus/qt.expect(Sz_spinhalf, state)**2*(N/2)**2
        invQFI_t[m] = 1/lambda_plus

    fig, ax = plt.subplots(layout='constrained')
    plt.plot(timescale*gamma,invQFI_t,'k-',linewidth=3.0)
    plt.plot(timescale*gamma,squ1_t,'r--',linewidth=3.0)
    ax = plt.gca()
    # ax.set_ylim([0.0,1.0])
    # ax.set_ylim(bottom=0)
    plt.savefig(f'squeezing_time_m0_N{round(N)}_app.png')

    fig, ax = plt.subplots(layout='constrained')
    plt.plot(timescale*gamma,np.log10(invQFI_t),'k-',linewidth=3.0)
    plt.plot(timescale*gamma,np.log10(squ1_t),'r--',linewidth=3.0)
    ax = plt.gca()
    # ax.set_ylim([0.0,1.0])
    # ax.set_ylim(bottom=0)
    plt.savefig(f'squeezing_time_log10_m0_N{round(N)}_app.png')

    return timescale, squ1_t, invQFI_t, gamma

# time_evolution_appro(1000, 0.3, 51, 0.0, 10.0)

def save_squeezing_QFI():
    
    N = 1000
    gamma = 0.3 #0.1 #0.3
    Nt = 301
    tf = 30.0 #10.0*3 #10.0
    # timescale, squ1_t, invQFI_t, gamma = time_evolution_appro(N, gamma, Nt, 0.0, tf)
    timescale, squ1_t, invQFI_t, N0_t, gamma, expSz, squ1_t_0 = time_evolution_m0(N, gamma, Nt, 0.0, tf)

    # with open(f'squeezing_app_N{round(N)}.npy', 'wb') as f:
    with open(f'squeezing_QFI_N{round(N)}_xi{round(gamma,2)}.npy', 'wb') as f:
        np.save(f, timescale)
        np.save(f, squ1_t)
        np.save(f, invQFI_t)
        np.save(f, N0_t)
        np.save(f, gamma)
        np.save(f, expSz)
        np.save(f, squ1_t_0)

def plot_squeezing_QFI():

    N = 1000

    # with open(f'squeezing_app_N{round(N)}.npy', 'rb') as f:
    #     timescale_app = np.load(f)
    #     squ1_t_app = np.load(f)
    #     invQFI_t_app = np.load(f)
    #     gamma_app = np.load(f)

    gamma = 0.3
    with open(f'squeezing_QFI_N{round(N)}_xi{round(gamma,2)}.npy', 'rb') as f:
        timescale_3 = np.load(f)
        squ1_t_3 = np.load(f)
        invQFI_t_3 = np.load(f)
        N0_t_3 = np.load(f)
        gamma_3 = np.load(f)
        expSz_3 = np.load(f)
        squ1_t_0_3 = np.load(f)

    gamma = 0.1
    with open(f'squeezing_QFI_N{round(N)}_xi{round(gamma,2)}.npy', 'rb') as f:
        timescale_1 = np.load(f)
        squ1_t_1 = np.load(f)
        invQFI_t_1 = np.load(f)
        N0_t_1 = np.load(f)
        gamma_1 = np.load(f)
        expSz_1 = np.load(f)
        squ1_t_0_1 = np.load(f) 

    # fig, ax = plt.subplots(layout='constrained')
    # plt.plot(timescale_1*gamma_1,np.log10(invQFI_t_1),'k-',linewidth=3.0)
    # plt.plot(timescale_1*gamma_1,np.log10(squ1_t_1),'k--',linewidth=3.0)
    # ax = plt.gca()
    # ax.set_ylim([-4.0,0.5])
    # # ax.set_ylim(bottom=0)
    # plt.savefig(f'test.png')

    mpl.rcParams.update({'font.size': 24})
    fig, ax = plt.subplots(layout='constrained')
    # plt.plot(timescale_app*gamma_app,np.log(invQFI_t_app),'b-',linewidth=6.0)
    # plt.plot(timescale_app*gamma_app,np.log(squ1_t_app),'r:',linewidth=3.0)
    plt.plot(timescale_3,np.log(invQFI_t_3),'k-',linewidth=3.0)
    plt.plot(timescale_3,np.log(squ1_t_3),'r--',linewidth=3.0)
    plt.plot(timescale_1,np.log(invQFI_t_1),'k-.',linewidth=3.0)
    plt.plot(timescale_1,np.log(squ1_t_1),'r:',linewidth=3.0)
    plt.axhline(y = 0, color = 'k', linestyle = ':',linewidth=2.0)
    # ax = plt.gca()
    # ax.set_ylim([0.0,1.0])
    # ax.set_ylim(bottom=0)
    plt.savefig(f'test_squeezing_QFI.png')

    mpl.rcParams.update({'font.size': 24})
    # fig, ax = plt.subplots(layout='constrained')
    fig = plt.figure()
    axs1 = plt.subplot(211)
    axs2 = plt.subplot(212)
    axs1.plot(timescale_3,squ1_t_0_3,'k-',linewidth=3.0)
    axs2.plot(timescale_3,expSz_3/(N/2),'k-',linewidth=3.0)
    axs2.set_ylim([0.0,1.1])
    # ax = plt.gca()
    plt.savefig(f'test_components_squeezing_N{round(N)}_xi{round(0.3,2)}.png')

    # fig, ax = plt.subplots(layout='constrained')
    # # plt.plot(timescale,N0_t,'k-o')
    # plt.plot(timescale_3*gamma_3,N0_t_3,'k-',linewidth=3.0)
    # plt.plot(timescale_1*gamma_1,N0_t_1,'k-.',linewidth=3.0)
    # ax = plt.gca()
    # # ax.set_ylim([0.0,1.0])
    # ax.set_ylim(bottom=0)
    # plt.savefig(f'N0_test.png')

# save_squeezing_QFI()
# plot_squeezing_QFI()

def test_squeezing_diff_maximum(N):

    gamma = 0.2
    Nt = 1001
    tf = 1000.0

    timescale, squ1_t, invQFI_t, N0_t, gamma, expSz, squ1_t_0 = time_evolution_m0(N,gamma,Nt,0.0,tf)

    fig, ax = plt.subplots(layout='constrained')
    plt.plot(timescale,np.log(squ1_t-invQFI_t),'k-',linewidth=3.0)
    ax = plt.gca()
    # ax.set_ylim([-5.0,2.0])
    plt.axvline(x = timescale[np.argmax(squ1_t-invQFI_t)], color = 'r', linestyle = '--', label = 'axvline')
    plt.savefig(f'test_N{round(N)}_xi{round(gamma,2)}.png')
    print(timescale[np.argmax(squ1_t-invQFI_t)])

    fig, ax = plt.subplots(layout='constrained')
    plt.plot(timescale,np.log(squ1_t),'k-',linewidth=3.0)
    plt.plot(timescale,np.log(invQFI_t),'r--',linewidth=3.0)
    ax = plt.gca()
    # ax.set_ylim([-5.0,2.0])
    plt.axvline(x = timescale[np.argmin(squ1_t)], color = 'b', linestyle = '--', label = 'axvline')
    plt.axvline(x = timescale[np.argmin(invQFI_t)], color = 'b', linestyle = '--', label = 'axvline')
    plt.axvline(x = timescale[np.argmax(squ1_t-invQFI_t)], color = 'r', linestyle = '--', label = 'axvline')
    plt.savefig(f'test_N{round(N)}_xi{round(gamma,2)}_1.png')

    # fig, ax = plt.subplots(layout='constrained')
    fig = plt.figure()
    axs1 = plt.subplot(211)
    axs2 = plt.subplot(212)    
    # axs1.plot(timescale,squ1_t,'k-',linewidth=3.0)
    axs1.plot(timescale,squ1_t_0,'r--',linewidth=3.0)
    axs2.plot(timescale,expSz/(N/2),'b:',linewidth=3.0)
    ax = plt.gca()
    plt.savefig(f'test_N{round(N)}_xi{round(gamma,2)}_2.png')

# test_squeezing_diff_maximum(100)

# plot_squeezing_diff_maximum(1000)

def plot_squeezing_diff_maximum_1(N):

    # array_gamma = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    array_gamma = np.linspace(0.1, 0.3, 21)
    
    # array_gamma = np.linspace(0.4, 0.9, 11)
    
    N_gamma = np.size(array_gamma)
    array_diff = np.zeros(N_gamma)
    
    # array_tf = np.ones(np.size(array_gamma))*1500.0
    # array_tf = np.ones(np.size(array_gamma))*1000.0
    # array_tf = np.ones(np.size(array_gamma))*500.0
    # array_tf = np.ones(np.size(array_gamma))*50.0
    array_tf = np.ones(np.size(array_gamma))*10.0
    # array_tf = np.ones(np.size(array_gamma))*np.pi/2

    # array_tf = [500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 1000.0, 1000.0]
    # array_tf = [200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 400.0, 400.0]
    # array_tf = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 200.0, 200.0]
    
    # Nt = 10001
    # Nt = 5001
    # Nt = 1001
    Nt = 501
    # Nt = 201
    # Nt = 2

    for j in range(np.size(array_gamma)):
        # vec0 = time_evolution_m0(N,array_gamma[j],Nt,0.0,array_tf[j])
        # array_diff[j] = max(vec0)
        vec0 = time_evolution_m0(N,array_gamma[j],Nt,0.0,array_tf[j]/array_gamma[j])
        array_diff[j] = vec0[-1]
        
    fig, ax = plt.subplots(1, 1, figsize=(5,4))
    plt.plot(array_gamma,np.log10(array_diff),'k--o')
    ax = plt.gca()
    # ax.set_ylim([-0.05,1.05])
    # ax.set_xlim([-0.05,1.05])
    plt.savefig(f'squeezing_max_diff_N{round(N)}_1.png')

    # fig, ax = plt.subplots(1, 1, figsize=(5,4))
    # plt.plot(array_gamma,np.log10(array_diff/N),'k--o')
    # ax = plt.gca()
    # # ax.set_ylim([-0.05,1.05])
    # # ax.set_xlim([-0.05,1.05])
    # plt.savefig(f'squeezing_max_diff_N{round(N)}_scaled_1.png')

    with open(f'squeezing_max_diff_N{round(N)}_1.npy', 'wb') as f:
        np.save(f, array_gamma)
        np.save(f, array_diff)
        np.save(f, array_tf)
        np.save(f, Nt)

# plot_squeezing_diff_maximum_1(100)
# plot_squeezing_diff_maximum_1(500)
# plot_squeezing_diff_maximum_1(1000)

def plot_squeezing_diff_maximum_100():

    N = 100

    array_gamma = np.linspace(0.1, 0.3, 21)
    N_gamma = np.size(array_gamma)
    array_diff = np.zeros(N_gamma)
    
    array_tf = 150.0
    Nt = 1501
    
    for j in range(np.size(array_gamma)):
        vec0 = time_evolution_m0(N,array_gamma[j],Nt,0.0,array_tf)
        array_diff[j] = max(vec0)
        
    fig, ax = plt.subplots(1, 1, figsize=(5,4))
    plt.plot(array_gamma,np.log(array_diff),'k--o')
    ax = plt.gca()
    # ax.set_ylim([-0.05,1.05])
    # ax.set_xlim([-0.05,1.05])
    plt.savefig(f'squeezing_max_diff_N{round(N)}.png')

    with open(f'squeezing_max_diff_N{round(N)}.npy', 'wb') as f:
        np.save(f, array_gamma)
        np.save(f, array_diff)
        np.save(f, array_tf)
        np.save(f, Nt)

# plot_squeezing_diff_maximum_100()

def plot_squeezing_diff_maximum_500():

    N = 500

    array_gamma = np.linspace(0.1, 0.3, 21)
    N_gamma = np.size(array_gamma)
    array_diff = np.zeros(N_gamma)
    
    # array_tf = 750.0
    # Nt = 7501
    
    array_tf = np.zeros(N_gamma)
    for j in range(N_gamma):
        if j <= 2:
            array_tf[j] = 750
        elif j <= 18:
            array_tf[j] = 500
        else:
            array_tf[j] = 250
    Nt = array_tf*10
    Nt = Nt.astype(int) + 1
    
    for j in range(np.size(array_gamma)):
        vec0 = time_evolution_m0(N,array_gamma[j],Nt[j],0.0,array_tf[j])
        array_diff[j] = max(vec0)
        
    fig, ax = plt.subplots(1, 1, figsize=(5,4))
    plt.plot(array_gamma,np.log(array_diff),'k--o')
    ax = plt.gca()
    # ax.set_ylim([-0.05,1.05])
    # ax.set_xlim([-0.05,1.05])
    plt.savefig(f'squeezing_max_diff_N{round(N)}.png')

    with open(f'squeezing_max_diff_N{round(N)}.npy', 'wb') as f:
        np.save(f, array_gamma)
        np.save(f, array_diff)
        np.save(f, array_tf)
        np.save(f, Nt)

# plot_squeezing_diff_maximum_500()

def plot_squeezing_diff_maximum_1000():

    N = 1000

    array_gamma = np.linspace(0.1, 0.3, 21)
    N_gamma = np.size(array_gamma)
    array_diff = np.zeros(N_gamma)
    
    # array_tf = 750.0
    # Nt = 7501
    
    array_tf = np.zeros(N_gamma)
    for j in range(N_gamma):
        if j <= 3:
            array_tf[j] = 1000
        elif j <= 5:
            array_tf[j] = 750
        elif j <= 8:
            array_tf[j] = 500
        else:
            array_tf[j] = 250
    Nt = array_tf*10
    Nt = Nt.astype(int) + 1
    
    for j in range(np.size(array_gamma)):
        vec0 = time_evolution_m0(N,array_gamma[j],Nt[j],0.0,array_tf[j])
        array_diff[j] = max(vec0)
        
    fig, ax = plt.subplots(1, 1, figsize=(5,4))
    plt.plot(array_gamma,np.log(array_diff),'k--o')
    ax = plt.gca()
    # ax.set_ylim([-0.05,1.05])
    # ax.set_xlim([-0.05,1.05])
    plt.savefig(f'squeezing_max_diff_N{round(N)}.png')

    with open(f'squeezing_max_diff_N{round(N)}.npy', 'wb') as f:
        np.save(f, array_gamma)
        np.save(f, array_diff)
        np.save(f, array_tf)
        np.save(f, Nt)

# plot_squeezing_diff_maximum_1000()

def plot_squeezing_diff_maximum(N):

    # array_gamma = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    array_gamma = np.linspace(0.1, 0.3, 21)
    # array_gamma = np.linspace(0.4, 0.9, 11)
    
    N_gamma = np.size(array_gamma)
    array_diff = np.zeros(N_gamma)
    
    array_tf = np.ones(np.size(array_gamma))*1500.0
    # array_tf = np.ones(np.size(array_gamma))*200.0
    # array_tf = np.ones(np.size(array_gamma))*100.0
    # array_tf = np.ones(np.size(array_gamma))*50.0
    # array_tf = np.ones(np.size(array_gamma))*30.0
    # array_tf = np.ones(np.size(array_gamma))*10.0

    # array_tf = [500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 1000.0, 1000.0]
    # array_tf = [200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 400.0, 400.0]
    # array_tf = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 200.0, 200.0]
    
    # Nt = 2001
    # Nt = 1001
    Nt = 501
    # Nt = 301
    # Nt = 201

    for j in range(np.size(array_gamma)):
        vec0 = time_evolution_m0(N,array_gamma[j],Nt,0.0,array_tf[j])
        array_diff[j] = max(vec0)
        
    fig, ax = plt.subplots(1, 1, figsize=(5,4))
    plt.plot(array_gamma,np.log(array_diff),'k--o')
    ax = plt.gca()
    # ax.set_ylim([-0.05,1.05])
    # ax.set_xlim([-0.05,1.05])
    plt.savefig(f'squeezing_max_diff_N{round(N)}.png')

    with open(f'squeezing_max_diff_N{round(N)}.npy', 'wb') as f:
        np.save(f, array_gamma)
        np.save(f, array_diff)
        np.save(f, array_tf)
        np.save(f, Nt)

# plot_squeezing_diff_maximum(10)
# plot_squeezing_diff_maximum(50)
# plot_squeezing_diff_maximum(100)
# plot_squeezing_diff_maximum(500)
# plot_squeezing_diff_maximum(1000)
# plot_squeezing_diff_maximum(2000)

def plot_squeezing_diff_maximum_all():

    # array_N = [10, 50, 100, 500, 1000]
    # array_N = [10, 100, 1000]
    array_N = [100, 500, 1000]

    mpl.rcParams.update({'font.size': 24})
    fig, ax = plt.subplots(layout='constrained')
    # fig, ax = plt.subplots(1, 1, figsize=(5,4))

    for j in range(np.size(array_N)):
        
        with open(f'squeezing_max_diff_N{round(array_N[j])}.npy', 'rb') as f:
            array_gamma = np.load(f)
            array_diff = np.load(f)
            array_tf = np.load(f)
            Nt = np.load(f)

        print(array_N[j])
        print(array_tf[j])
        print(Nt)

        # plt.plot(array_gamma,array_diff/array_N[j],'--o',label='N = %.0f' %(array_N[j]))
        plt.plot(array_gamma,array_diff,'--o',label='N = %.0f' %(array_N[j]),linewidth=3.0)
        ax = plt.gca()
        plt.legend(loc="upper left")
        # ax.set_ylim([0.095,1.05])
        # ax.set_xlim([0.095,0.305])
        
    plt.savefig(f'squeezing_max_diff_N_1005001000.png')

# plot_squeezing_diff_maximum_all()

def plot_squeezing_diff_maximum_all_1():

    with open(f'squeezing_max_diff_N100.npy', 'rb') as f:
        array_gamma_100 = np.load(f)
        array_diff_100 = np.load(f)
        array_tf_100 = np.load(f)
        Nt_100 = np.load(f)

    with open(f'squeezing_max_diff_N500.npy', 'rb') as f:
        array_gamma_500 = np.load(f)
        array_diff_500 = np.load(f)
        array_tf_500 = np.load(f)
        Nt_500 = np.load(f)

    with open(f'squeezing_max_diff_N1000.npy', 'rb') as f:
        array_gamma_1000 = np.load(f)
        array_diff_1000 = np.load(f)
        array_tf_1000 = np.load(f)
        Nt_1000 = np.load(f)
        
    mpl.rcParams.update({'font.size': 24})
    
    fig, ax = plt.subplots(layout='constrained')
    # plt.plot(array_gamma,array_diff/array_N[j],'--o',label='N = %.0f' %(array_N[j]))
    plt.plot(array_gamma_1000,array_diff_1000,'-o',label='N = 1000', linewidth=3.0, markersize=10)
    plt.plot(array_gamma_500,array_diff_500,'--o',label='N = 500', linewidth=3.0, markersize=10)
    plt.plot(array_gamma_100,array_diff_100,':o',label='N = 100', linewidth=3.0, markersize=10)
    plt.legend(loc="upper left")
    plt.savefig(f'squeezing_max_diff_N_1005001000.png')

    fig, ax = plt.subplots(layout='constrained')
    plt.plot(array_gamma_1000,array_diff_1000/1000,'-o',label='N = 1000', linewidth=3.0, markersize=10)
    plt.plot(array_gamma_500,array_diff_500/500,'--o',label='N = 500', linewidth=3.0, markersize=10)
    plt.plot(array_gamma_100,array_diff_100/100,':o',label='N = 100', linewidth=3.0, markersize=10)
    plt.legend(loc="upper left")
    plt.savefig(f'squeezing_max_diff_N_1005001000_scaled.png')

    fig, ax = plt.subplots(layout='constrained')
    plt.plot(array_gamma_1000,np.log10(array_diff_1000/1000),'-o',label='N = 1000', linewidth=3.0, markersize=10)
    plt.plot(array_gamma_500,np.log10(array_diff_500/500),'--o',label='N = 500', linewidth=3.0, markersize=10)
    plt.plot(array_gamma_100,np.log10(array_diff_100/100),':o',label='N = 100', linewidth=3.0, markersize=10)
    # plt.legend(loc="upper left")
    plt.savefig(f'test.png')

# plot_squeezing_diff_maximum_all_1()

def plot_squeezing_diff_maximum_all_2():

    with open(f'squeezing_max_diff_N100.npy', 'rb') as f:
        array_gamma_100 = np.load(f)
        array_diff_100 = np.load(f)
        array_tf_100 = np.load(f)
        Nt_100 = np.load(f)

    with open(f'squeezing_max_diff_N500.npy', 'rb') as f:
        array_gamma_500 = np.load(f)
        array_diff_500 = np.load(f)
        array_tf_500 = np.load(f)
        Nt_500 = np.load(f)

    with open(f'squeezing_max_diff_N1000.npy', 'rb') as f:
        array_gamma_1000 = np.load(f)
        array_diff_1000 = np.load(f)
        array_tf_1000 = np.load(f)
        Nt_1000 = np.load(f)

    ref_1000_1_x = [0.192, 0.292]
    ref_1000_1_y = [-0.0616, 7.9435]
    ref_1000_2_x = [0.1, 0.2]
    ref_1000_2_y = [-3.8114, -0.7658]

    mpl.rcParams.update({'font.size': 24})
    
    # fig, ax = plt.subplots(layout='constrained')
    # # plt.plot(array_gamma,array_diff/array_N[j],'--o',label='N = %.0f' %(array_N[j]))
    # plt.plot(array_gamma_1000,array_diff_1000,'-o',label='N = 1000', linewidth=3.0, markersize=10)
    # plt.plot(array_gamma_500,array_diff_500,'--o',label='N = 500', linewidth=3.0, markersize=10)
    # plt.plot(array_gamma_100,array_diff_100,':o',label='N = 100', linewidth=3.0, markersize=10)
    # plt.legend(loc="upper left")
    # plt.savefig(f'squeezing_max_diff_N_1005001000_1.png')

    # fig, ax = plt.subplots(layout='constrained')
    # plt.plot(array_gamma_1000,array_diff_1000/1000,'-o',label='N = 1000', linewidth=3.0, markersize=10)
    # plt.plot(array_gamma_500,array_diff_500/500,'--o',label='N = 500', linewidth=3.0, markersize=10)
    # plt.plot(array_gamma_100,array_diff_100/100,':o',label='N = 100', linewidth=3.0, markersize=10)
    # plt.legend(loc="upper left")
    # plt.savefig(f'squeezing_max_diff_N_1005001000_1_scaled.png')

    # fig, ax = plt.subplots(layout='constrained')
    # plt.plot(array_gamma_1000,np.log(array_diff_1000),'-o',label='N = 1000', linewidth=3.0, markersize=10)
    # plt.plot(array_gamma_500,np.log(array_diff_500),'--o',label='N = 500', linewidth=3.0, markersize=10)
    # plt.plot(array_gamma_100,np.log(array_diff_100),':o',label='N = 100', linewidth=3.0, markersize=10)
    # plt.plot(ref_1000_1_x,ref_1000_1_y,'-k', linewidth=3.0)
    # plt.plot(ref_1000_2_x,ref_1000_2_y,'-k', linewidth=3.0)
    # plt.legend(loc="upper left")
    # ax.set_ylim([-5,5.5])
    # plt.savefig(f'test.png')

    fig, ax = plt.subplots(layout='constrained')
    plt.plot(array_gamma_1000,np.log(array_diff_1000/1000),'-o',label='N = 1000', linewidth=3.0, markersize=10)
    plt.plot(array_gamma_500,np.log(array_diff_500/500),'--o',label='N = 500', linewidth=3.0, markersize=10)
    plt.plot(array_gamma_100,np.log(array_diff_100/100),':o',label='N = 100', linewidth=3.0, markersize=10)
    # plt.legend(loc="upper left")
    # ax.set_ylim([-5,5.5])
    plt.savefig(f'test_normalisedN.png')

# plot_squeezing_diff_maximum_all_2()

def export_text():

    with open(f'squeezing_max_diff_N100.npy', 'rb') as f:
        array_gamma_100 = np.load(f)
        array_diff_100 = np.load(f)
        array_tf_100 = np.load(f)
        Nt_100 = np.load(f)

    with open(f'squeezing_max_diff_N500.npy', 'rb') as f:
        array_gamma_500 = np.load(f)
        array_diff_500 = np.load(f)
        array_tf_500 = np.load(f)
        Nt_500 = np.load(f)

    with open(f'squeezing_max_diff_N1000.npy', 'rb') as f:
        array_gamma_1000 = np.load(f)
        array_diff_1000 = np.load(f)
        array_tf_1000 = np.load(f)
        Nt_1000 = np.load(f)
        
    np.savetxt("array_gamma_100.txt", array_gamma_100)
    np.savetxt("array_diff_100.txt", array_diff_100)
    np.savetxt("array_gamma_500.txt", array_gamma_500)
    np.savetxt("array_diff_500.txt", array_diff_500)
    np.savetxt("array_gamma_1000.txt", array_gamma_1000)
    np.savetxt("array_diff_1000.txt", array_diff_1000)

    # mpl.rcParams.update({'font.size': 24})
    # fig, ax = plt.subplots(layout='constrained')
    # plt.plot(array_gamma_1000,np.log(array_diff_1000/1000),'-o',label='N = 1000', linewidth=3.0, markersize=10)
    # plt.plot(array_gamma_500,np.log(array_diff_500/500),'--o',label='N = 500', linewidth=3.0, markersize=10)
    # plt.plot(array_gamma_100,np.log(array_diff_100/100),':o',label='N = 100', linewidth=3.0, markersize=10)
    # plt.savefig(f'test.png')

# export_text()

def plot_test():

    with open(f'squeezing_max_diff_N1000.npy', 'rb') as f:
        array_gamma = np.load(f)
        array_diff = np.load(f)
        array_tf = np.load(f)
        Nt = np.load(f)

        plt.plot(array_gamma,array_diff,'--o')
        ax = plt.gca()
        plt.legend(loc="upper left")
        # ax.set_ylim([0.095,1.05])
        # ax.set_xlim([0.095,0.305])
        
    plt.savefig('test.png')

# plot_test()

def squeezing_minmum(N,gamma,Nt,ti,tf):

    Nt, timescale, squ1_t, invQFI_t = time_evolution_m0(N,gamma,Nt,ti,tf)
    ind_squ = 0
    ind_QFI = 0

    for j in range(Nt):
        if squ1_t[j] < squ1_t[j+1]:
            ind_squ = j
            break

    for j in range(Nt):
        if invQFI_t[j] < invQFI_t[j+1]:
            ind_QFI = j
            break

    return squ1_t[ind_squ], invQFI_t[ind_QFI]

def plot_squeezing_minmum(N):

    array_gamma = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
    N_gamma = np.size(array_gamma)
    array_squ1 = np.zeros(N_gamma)
    array_invQFI = np.zeros(N_gamma)

    array_tf = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 10.0]
    Nt = 100

    for j in range(np.size(array_gamma)):
        array_squ1[j], array_invQFI[j] = squeezing_minmum(N,array_gamma[j],Nt,0.0,array_tf[j])
        
    # return array_gamma, array_squ1, array_invQFI

    fig, ax = plt.subplots(1, 1, figsize=(5,4))
    plt.plot(array_gamma,array_invQFI,'k--o')
    ax = plt.gca()
    ax.set_ylim([-0.05,1.05])
    ax.set_xlim([-0.05,1.05])
    plt.savefig(f'squeezing_min_N{round(N)}.png')

# plot_squeezing_minmum(1000)

def time_evolution(N, gamma, Nt, ti, tf):

    # base
    # registeredN = [10,20]
    # if N == 
    base, dim = generate_base(N)

    mat_toxy = qt.Qobj(transform_toxy_frompm(base,dim)) # unitary transformation to mode x, 0, y
    # mat_tosa = qt.Qobj(transform_tosa_frompm(base,dim)) # unitary transformation to mode s, 0, a
    mat_S_N = qt.Qobj(transform_to2mode(base,dim,N,1)) # projection to symmetric subspace with total spin N
    mat_A_N = qt.Qobj(transform_to2mode(base,dim,N,-1)) # projection to anti-symmetric subspace with total spin N

    # operators
    [Jx,Qzx,Dxy,N0,Jz,Y,Qxy,Jy,Qyz,Np,Nm]=Operators.operators(base, dim)
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

    # initial state 
    # 0N0
    init = qt.Qobj(np.array([1 if base[i][1]==N else 0 for i in range(dim)]))
    # init = init.unit()
    # bended
    # init = qt.Qobj(coherent(np.sqrt(0.0), np.sqrt(0.0), dim, base, N))
    normx_t = np.zeros(Nt)

    # one mode
    X = (qt.destroy(N+1) + qt.create(N+1)) / 2
    P = (qt.destroy(N+1) - qt.create(N+1)) / (2*1j)
    var_modex_t = np.zeros(Nt)

    # Hamiltonian for evolution
    # gamma = 0.9 #0.8 #0.2 #0
    # 0.2 for ciritical gamma
    # H = N0*gamma + (1-gamma)/(2*N)*(1j*N0*np.pi/2).expm()*(Jx*Jx + Jy*Jy + Jz*Jz)*(-1j*N0*np.pi/2).expm()
    # H = N0*gamma + (1-gamma)/(2*N)*(Jx*Jx + Jy*Jy + Jz*Jz)
    H = -(1-gamma)*N0 + gamma/N*(Jx*Jx + Jy*Jy + Jz*Jz)

    for m in tqdm(range(0, Nt)):

        # time evolution
        t = timescale[m]
        u_evo = (-1j * H * t).expm()
        state = u_evo * init
        
        # squeezing
        lambda_minus, lambda_plus = covariance_matrix_spinhalf(state,N,Sx_spinhalf,Sy_spinhalf)
        squ0_t[m] = (qt.expect(Sx_spinhalf*Sx_spinhalf, state)-qt.expect(Sx_spinhalf, state)**2)/qt.expect(Sz_spinhalf, state)**2*N
        squ1_t[m] = lambda_minus/qt.expect(Sz_spinhalf, state)**2*(N/2)**2

        # QFI
        invQFI_t[m] = 1/lambda_plus
    
        # transform mode +, 0, - to mode x, 0, y
        psi = mat_toxy*state #qt.ket2dm(mat_toxy*state) # this
    
        # project to a single model (ind=0,1,2 for mode x,0,y)
        rho_reduced_x = qt.Qobj(np.flipud(qt.Qobj.full(mat_S_N*psi)))
        rho_reduced_y = qt.Qobj(np.flipud(qt.Qobj.full(mat_A_N*psi)))

        # variance of one mode on phase space
        var_modex_t[m] = var_mode_phase(X,P,rho_reduced_x)
        #qt.expect(P*P, rho_reduced_x)
        # qt.expect(X*X, rho_reduced_x)-qt.expect(X, rho_reduced_x)**2

        #
        normx_t[m] = qt.Qobj.norm(rho_reduced_x)
    
    fig, ax = plt.subplots(1, 1, figsize=(4,4))
    plt.plot(timescale,var_modex_t,'k-o')
    ax = plt.gca()
    ax.set_ylim(bottom=0)
    plt.savefig(f'variance_phase.png')

    return timescale, invQFI_t, squ0_t, squ1_t, normx_t

def repeataiton_forgamma(N,Nt,ti,tf,Ngamma,gamma0,gamma1):

    array_gamma = np.linspace(gamma0, gamma1, Ngamma)
    squ0_gamma = np.zeros(Ngamma)
    squ1_gamma = np.zeros(Ngamma)
    invQFI_gamma = np.zeros(Ngamma)

    for m in range(0, Ngamma):

        timescale, invQFI_t, squ0_t, squ1_t, normx_t = time_evolution(N, array_gamma[m], Nt, ti, tf)
        squ0_gamma[m] = min(squ0_t)
        squ1_gamma[m] = min(squ1_t)
        invQFI_gamma[m] = min(invQFI_t)

        print(m)

    return array_gamma, squ0_gamma, squ1_gamma, invQFI_gamma

def repeataiton_forN(Nt,ti,tf,M,N0,N1,gamma):

    array_N = np.linspace(N0, N1, M)
    squ0_N = np.zeros(M)
    squ1_N = np.zeros(M)
    invQFI_N = np.zeros(M)

    for m in range(0, M):

        timescale, invQFI_t, squ0_t, squ1_t, normx_t = time_evolution(int(array_N[m]), gamma, Nt, ti, tf)
        squ0_N[m] = min(squ0_t)
        squ1_N[m] = min(squ1_t)
        invQFI_N[m] = min(invQFI_t)

        print(m)

    return array_N, squ0_N, squ1_N, invQFI_N

def time_evolution_wigner(N, ind_data, gamma, Nt, ti, tf):

    time = TicToc()
    time.tic() #Start timer
    base,dim,Jx,Qzx,Dxy,N0,Jz,Y,Qxy,Jy,Qyz,Np,Nm,mat_toxy_sub = define_base_operators_transform(N,ind_data)    
    time.toc()
    # mat_toxy = qt.Qobj(transform_toxy_frompm(base,dim)) # unitary transformation to mode x, 0, y
    # mat_tosa = qt.Qobj(transform_tosa_frompm(base,dim)) # unitary transformation to mode s, 0, a
    # mat_S_N = qt.Qobj(transform_to2mode(base,dim,N,1)) # projection to symmetric subspace with total spin N
    # mat_A_N = qt.Qobj(transform_to2mode(base,dim,N,-1)) # projection to anti-symmetric subspace with total spin N

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

    # print(qt.expect(Sx_spinhalf, init))
    # print(qt.expect(Sy_spinhalf, init))
    # print(qt.expect(Sz_spinhalf, init))

    # print(qt.expect(4*Sx_spinhalf**2, init)/N)
    # print(qt.expect(4*Sy_spinhalf**2, init)/N)
    # print(qt.expect(4*Sz_spinhalf**2, init)/N)

    # print(qt.expect(4*Ay_spinhalf**2, init)/N)
    # print(qt.expect(4*Jz_spinhalf**2, init)/N)

    # print(qt.expect(Sx_spinhalf**2+Sy_spinhalf**2+Sz_spinhalf**2, init))
    # print(qt.expect(Ax_spinhalf**2+Ay_spinhalf**2+Az_spinhalf**2, init))
    # print(qt.expect(Sx_spinhalf, init))
    # print(qt.expect(Sy_spinhalf, init))
    # print(qt.expect(Sz_spinhalf, init))
    # print(qt.expect(Ax_spinhalf, init))
    # print(qt.expect(Ay_spinhalf, init))
    # print(qt.expect(Az_spinhalf, init))

    # for gamma=0
    # qt.expect(Sz_spinhalf, init) is N/2
    # qt.expect(S^2, init) is N/2*(N/2+1)

    #
    Ncoherent = 31
    X0 = np.linspace(-1.5,1.5,Ncoherent)
    Y0 = np.linspace(-1.5,1.5,Ncoherent)
    # coherent_Q_fun = np.zeros((Ncoherent, Ncoherent))
    # coh_xy = husimi_xy_1(X0, Y0, dim, base, N)

    # Hamiltonian for evolution
    # gamma = 0.9 #0.8 #0.2 #0
    # 0.2 for ciritical gamma
    # H = N0*gamma + (1-gamma)/(2*N)*(1j*N0*np.pi/2).expm()*(Jx*Jx + Jy*Jy + Jz*Jz)*(-1j*N0*np.pi/2).expm()
    # H = N0*gamma + (1-gamma)/(2*N)*(Jx*Jx + Jy*Jy + Jz*Jz)
    # H = -(1-gamma)*N0 + gamma/N*(Jx*Jx + Jy*Jy + Jz*Jz)
    H = -(1-gamma)*N0 - gamma/N*(1j*N0*np.pi/2).expm()*(Jx*Jx + Jy*Jy + Jz*Jz)*(-1j*N0*np.pi/2).expm()
    # H = -(1-gamma)/gamma*N0 - 1/N*(1j*N0*np.pi/2).expm()*(Jx*Jx + Jy*Jy + Jz*Jz)*(-1j*N0*np.pi/2).expm()

    # print(qt.expect(H, init)/N)

    #
    # psi0 = qt.Qobj(np.array([1 if base[i][0]==N else 0 for i in range(dim)]))
    # print(qt.expect(gamma/N*(Jx*Jx + Jy*Jy + Jz*Jz),psi0))

    # time evolution operator
    time.tic()
    # evals, ekets = qt.Qobj.eigenstates(H)
    # evals, evecs = sp_eigs(H.data, H.isherm) # sparse=sparse,sort=sort, eigvals=eigvals, tol=tol, maxiter=maxiter)
    # evecs = evecs.T
    evals, evecs = eig(H.full())
    evals = evals.real
    time.toc()

    # print(evals[0]/N)
    
    # return 0
    
    for m in tqdm(range(0, Nt)):

        # time evolution
        t = timescale[m]
        # u_evo = (-1j * H * t).expm()
        u_evo = qt.Qobj(evecs@(np.diag(np.exp(-1j*evals*t))@evecs.conj().T))
        state = u_evo * init

        # squeezing
        lambda_minus, lambda_plus = covariance_matrix_spinhalf(state,N,Sx_spinhalf,Sy_spinhalf)
        squ0_t[m] = (qt.expect(Sx_spinhalf*Sx_spinhalf, state)-qt.expect(Sx_spinhalf, state)**2)/qt.expect(Sz_spinhalf, state)**2*N
        squ1_t[m] = lambda_minus/qt.expect(Sz_spinhalf, state)**2*(N/2)**2
        N0_t[m] = qt.expect(N0, state)

        # QFI
        invQFI_t[m] = 1/lambda_plus
    
        # transform mode +, 0, - to mode x, 0, y
        # project to a single model (ind=0,1,2 for mode x,0,y)
        # psi = mat_toxy*state #qt.ket2dm(mat_toxy*state)
        # rho_reduced_x = qt.Qobj(np.flipud(qt.Qobj.full(mat_S_N*psi)))
        # rho_reduced_y = qt.Qobj(np.flipud(qt.Qobj.full(mat_A_N*psi)))

        psi = qt.Qobj(np.flipud(qt.Qobj.full(mat_toxy_sub*state)))
        normx_t[m] = qt.Qobj.norm(psi)
        psi_renorm = psi/qt.Qobj.norm(psi)

        # wigner function on phase space
        W_x_fun = qt.wigner(psi_renorm, xvec, xvec,g=2)
        X_t[m] = qt.expect(X, psi)

        # W_x_fun = qt.wigner(rho_reduced_x, xvec, xvec,g=2)
        # W_y_fun = qt.wigner(rho_reduced_y, xvec, xvec,g=2)

        #
        expSx[m] = qt.expect(Sx_spinhalf, state)
        expSy[m] = qt.expect(Sy_spinhalf, state)
        expSz[m] = qt.expect(Sz_spinhalf, state)
        squ1_t_0[m] = lambda_minus

        # var_W_x_t[m] = np.sqrt(var_mode_phase_wigner(X,PX,W_x_fun,dx))/2
        # pop_fock_end_t[m] = np.abs(rho_reduced_x[N])

        # variance of one mode on phase space
        # var_modex_t[m] = var_mode_phase(X,P,rho_reduced_x)

        # if var_modex_t[m] < var_modex_t[m-1]:
        #     if var_modex_t[m-1] == max(var_modex_t):
        #         print(m-1)

        # trace out two modes
        # rho_traced_0y = transform_to1mode(base,dim,N,qt.ket2dm(state),0)
        # W_x_fun_1 = qt.wigner(rho_traced_0y, xvec, xvec,g=2)

        # wigner function on bloch sphere
        # psi_1 = mat_toxy_sub*state
        # psi_1_renorm = psi_1/qt.Qobj.norm(psi_1)

        spin_W_fun, Theta_W, Phi_W = qt.spin_wigner(qt.ket2dm(psi_renorm), theta, phi)

        # plot
        # mpl.rcParams.update({'font.size': 14})
        # mpl.rcParams.update({'font.size': 24})
        mpl.rcParams.update({'font.size': 32})

        # husimi with coherent state
        # coherent_Q_fun = husimi_xy_2(state, X0, Y0, coh_xy)

        # fig, ax = plt.subplots(1, 2, figsize=(8,4))
        # wmap = qt.wigner_cmap(W_x_fun)
        # cs = ax[0].contourf(xvec, xvec, W_x_fun, 100, cmap=wmap)
        # fig.colorbar(cs, ax=ax[0])
        # ax[0].set_aspect('equal')
        # cs = ax[1].contourf(X0, Y0, coherent_Q_fun)
        # fig.colorbar(cs, ax=ax[1])
        # ax[1].set_aspect('equal')
        # plt.savefig(f'overlap_cohernet_{round(m)}.png')
            
        # fig, ax = plt.subplots(1, 2, figsize=(8,4))
        # wmap = qt.wigner_cmap(W_x_fun)
        # cs = ax[0].contourf(xvec, xvec, W_x_fun, 100, cmap=wmap)
        # fig.colorbar(cs, ax=ax[0])
        # ax[0].set_aspect('equal')
        # # wmap = qt.wigner_cmap(W_y_fun)
        # # cs = ax[1].contourf(xvec, xvec, W_y_fun, 100, cmap=wmap)
        # # fig.colorbar(cs, ax=ax[1])
        # # ax[1].set_aspect('equal')
        # qt.plot_fock_distribution(rho_reduced_x,ax=ax[1],fig=fig)
        # plt.savefig(f'wigner_phase_{round(m)}.png')

        # fig, ax = plt.subplots(1, 2, figsize=(8,4))
        # wmap = qt.wigner_cmap(W_x_fun)
        # cs = ax[0].contourf(xvec, xvec, W_x_fun, 100, cmap=wmap)
        # fig.colorbar(cs, ax=ax[0])
        # ax[0].set_aspect('equal')
        # wmap = qt.wigner_cmap(W_x_fun_1)
        # cs = ax[1].contourf(xvec, xvec, W_x_fun_1, 100, cmap=wmap)
        # fig.colorbar(cs, ax=ax[1])
        # ax[1].set_aspect('equal')
        # plt.savefig(f'wigner_phase_N{round(N)}_{round(m)}.png')

        # fig, ax = plt.subplots(1, 2, figsize=(8,4))
        # wmap = qt.wigner_cmap(W_x_fun)
        # cs = ax[0].contourf(xvec, xvec, W_x_fun, 100, cmap=wmap)
        # fig.colorbar(cs, ax=ax[0])
        # ax[0].set_aspect('equal')
        # plt.savefig(f'wigner_phase_test_N{round(N)}_{round(m)}.png')

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
        # plot_sphere_W_0(spin_W_fun, Theta_W, Phi_W)
        # # plt.show()
        # plt.savefig(f'wigner_spin_test_N{round(N)}_{round(m)}.png')
        
        # print(spin_W_fun.max())
        # print(spin_W_fun.min())

    # fig, ax = plt.subplots(1, 1, figsize=(4,4))
    # plt.plot(timescale,expSz,'k-o')
    # plt.plot(timescale,squ1_t_0,'r-*')
    # # plt.plot(timescale,expAz,'b--')
    # # plt.plot(timescale,expA2,'k:')
    # ax = plt.gca()
    # # ax.set_ylim([0.0,1.0])
    # # ax.set_ylim(bottom=0)
    # plt.savefig(f'test_N{round(N)}_xi0{round(gamma*10)}.png')

    # fig, ax = plt.subplots(1, 1, figsize=(4,4))
    # plt.plot(timescale,expSx,'k-o')
    # plt.plot(timescale,expSy,'r-*')
    # plt.plot(timescale,expSz,'b--')
    # # plt.plot(timescale,expA2,'k:')
    # ax = plt.gca()
    # # ax.set_ylim([0.0,1.0])
    # # ax.set_ylim(bottom=0)
    # plt.savefig(f'test2_N{round(N)}_xi0{round(gamma*10)}.png')

    # fig, ax = plt.subplots(layout='constrained')
    # plt.plot(timescale,normx_t,'k-o')
    # ax = plt.gca()
    # # ax.set_ylim([0.0,1.0])
    # ax.set_ylim(bottom=0)
    # plt.savefig(f'norm_psix_test_N{round(N)}_xi0{round(gamma*10)}.png')

    # fig, ax = plt.subplots(layout='constrained')
    # plt.plot(timescale,N0_t,'k-o')
    # ax = plt.gca()
    # # ax.set_ylim([0.0,1.0])
    # ax.set_ylim(bottom=0)
    # plt.savefig(f'N0_N{round(N)}_xi0{round(gamma*10)}.png')

    mpl.rcParams.update({'font.size': 32})
    fig, ax = plt.subplots(layout='constrained')
    plt.plot(timescale,X_t,'k-')
    ax = plt.gca()
    # ax.set_ylim([0.0,1.0])
    # ax.set_ylim(bottom=0)
    # ax.set_aspect('equal')
    plt.savefig(f'Xt_N{round(N)}_xi0{round(gamma*10)}.png')

    # fig, ax = plt.subplots(1, 1, figsize=(4,4))
    # # plt.plot(timescale,squ0_t,'r--o')
    # plt.plot(timescale,squ1_t,'b--o')
    # plt.plot(timescale,invQFI_t,'k-o')
    # ax = plt.gca()
    # # ax.set_ylim([0.0,3.0])
    # ax.set_ylim(bottom=0)
    # plt.savefig(f'squeezing_test_time_N{round(N)}_xi0{round(gamma*10)}.png')

    # print(expA2[0])

    # fig, ax = plt.subplots(1, 1, figsize=(5,4))
    # plt.plot(timescale,squ1_t-invQFI_t,'k-o')
    # ax = plt.gca()
    # # ax.set_ylim([0.0,1.0])
    # ax.set_ylim(bottom=0)
    # plt.savefig(f'squeezing_diff_time_N{round(N)}.png')
    # print(squ1_t[Nt-1])
    # print(invQFI_t[Nt-1])

    # fig, ax = plt.subplots(1, 1, figsize=(4,4))
    # plt.plot(timescale,var_W_x_t,'k-o')
    # ax = plt.gca()
    # # ax.set_ylim([0.0,1.0])
    # ax.set_ylim(bottom=0)
    # plt.savefig(f'var_W_x_time.png')

    # fig, ax = plt.subplots(1, 1, figsize=(4,4))
    # plt.plot(timescale,var_modex_t,'k-o')
    # ax = plt.gca()
    # ax.set_ylim(bottom=0)
    # plt.savefig(f'variance_phase.png')

    # fig, ax = plt.subplots(1, 1, figsize=(4,4))
    # plt.plot(timescale,pop_fock_end_t,'k-o')
    # ax = plt.gca()
    # # ax.set_ylim([0.0,1.0])
    # ax.set_ylim(bottom=0)
    # plt.savefig(f'pop_fock_end.png')
    
    # return timescale, invQFI_t, squ0_t, squ1_t, normx_t

# time_evolution_wigner(50, 0, 0.5, 5, 0.0, 2.0)

# time_evolution_wigner(10, 0, 0.3, 3, 0.0, 2*np.pi)
time_evolution_wigner(50, 0, 0.3, 3, 0.0, 2*np.pi)

# time_evolution_wigner(50, 1, 0.0, 41, 0.0, 2*np.pi)

# time_evolution_wigner(50, 1, 0.5, 5, 0.0, 2.0)

# main
# array_gamma, squ0_gamma, squ1_gamma, invQFI_gamma = repeataiton_forgamma(50,41,0.0,4.0,10,0.1,1.0)
# fig, ax = plt.subplots(1, 1, figsize=(4,4))
# plt.plot(array_gamma,invQFI_gamma,'k-o')
# plt.plot(array_gamma,squ0_gamma,'r--o')
# plt.plot(array_gamma,squ1_gamma,'b--o')
# ax = plt.gca()
# ax.set_ylim(bottom=0)
# ax.set_ylim([0.0,1.0])
# plt.savefig(f'squeezing_xi.png')

# array_N, squ0_N, squ1_N, invQFI_N = repeataiton_forN(41,0.0,4.0,5,10,50,0.9)
# fig, ax = plt.subplots(1, 1, figsize=(4,4))
# plt.plot(array_N,invQFI_N,'k-o')
# plt.plot(array_N,squ0_N,'r--o')
# plt.plot(array_N,squ1_N,'b--o')
# ax = plt.gca()
# ax.set_ylim(bottom=0)
# ax.set_ylim([0.0,1.0])
# plt.savefig(f'squeezing_N.png')

# timescale, invQFI_t, squ0_t, squ1_t, normx_t = time_evolution(20, 0.1, 41, 0.0, 4.0)

def time_evolution_m0_wigner(N, gamma, Nt, ti, tf):

    base,dim = generate_base_m0(N)
    [N0,Y,Np,Nm,Jx_2,Qzx_2,Dxy_2,N0_2,Jz_2,Y_2,Qxy_2,Jy_2,Qyz_2,Np_2,Nm_2,op,op_conj] = Operators.operators_m0(base, N)
    
    Sx_spinhalf_2 = Jx_2/4
    Sy_spinhalf_2 = Qyz_2/4
    Sz_spinhalf = (-np.sqrt(3)*Y)/4 #(-np.sqrt(3)*Y - Dxy)/4
    Sz_spinhalf_2 = (3*Y_2 + Dxy_2)/16

    mat_toxy_y0_m0 = qt.Qobj(transform_toxy_frompm_modex0_m0(dim,N))

    # time
    timescale = np.linspace(ti, tf, Nt)

    #
    squ0_t = np.zeros(Nt)
    squ1_t = np.zeros(Nt)
    invQFI_t = np.zeros(Nt)
    QFI_phase_t = np.zeros(Nt)

    #
    Nx = 200
    xvec = np.linspace(-5,5,Nx)
    normx_t = np.zeros(Nt)
    
    # initial state 
    # 0N0
    init = qt.Qobj(np.array([1 if base[i][1]==N else 0 for i in range(dim)]))
    # bended
    # r0 = bend_degree(gamma)
    # init = qt.Qobj(coherent(np.sqrt(r0), np.sqrt(0.0), dim, base, N))
    # init = qt.Qobj(coherent(np.sqrt(1.0), np.sqrt(0.0), dim, base, N))
    # bended initial state cannot be used for m=0
    
    # Hamiltonian for evolution
    # 0.2 for ciritical gamma
    H = -(1-gamma)*N0 + gamma/N*(Jx_2 + Jy_2 + Jz_2)
    # H = -(1-gamma)/gamma*N0 + 1/N*(Jx_2 + Jy_2 + Jz_2)

    # time evolution operator
    # evals, ekets = qt.Qobj.eigenstates(H)
    # evals, evecs = sp_eigs(H.data, H.isherm) # sparse=sparse,sort=sort, eigvals=eigvals, tol=tol, maxiter=maxiter)
    # evecs = evecs.T
    evals, evecs = eig(H.full())
    evals = evals.real
    evecs = evecs.T

    for m in tqdm(range(0, Nt)):

        # time evolution
        t = timescale[m]
        # u_evo = (-1j * H * t).expm()
        u_evo = qt.Qobj(evecs@(np.diag(np.exp(-1j*evals*t))@evecs.conj().T))
        state = u_evo * init

        # squeezing and QFI
        lambda_minus, lambda_plus = covariance_matrix_spinhalf_m0(state,N,Sx_spinhalf_2,Sy_spinhalf_2,op,op_conj)
        squ0_t[m] = (qt.expect(Sx_spinhalf_2, state))/qt.expect(Sz_spinhalf, state)**2*N
        squ1_t[m] = lambda_minus/qt.expect(Sz_spinhalf, state)**2*(N/2)**2
        invQFI_t[m] = 1/lambda_plus

        # # one mode
        # rho_reduced = transform_to1mode(base,dim,N,qt.ket2dm(state),0)
        # QFI_phase_t[m] = QFI_onemode(rho_reduced)

        #
        psi = qt.Qobj(np.flipud(qt.Qobj.full(mat_toxy_y0_m0*state)))
        normx_t[m] = qt.Qobj.norm(psi)
        psi_renorm = psi/qt.Qobj.norm(psi)

        # wigner function on phase space
        W_x_fun = qt.wigner(psi_renorm, xvec, xvec,g=2)

        # plot
        mpl.rcParams.update({'font.size': 16})

        fig, ax = plt.subplots(1, 2, figsize=(8,4))
        wmap = qt.wigner_cmap(W_x_fun)
        cs = ax[0].contourf(xvec, xvec, W_x_fun, 100, cmap=wmap)
        fig.colorbar(cs, ax=ax[0])
        ax[0].set_aspect('equal')
        plt.savefig(f'wigner_phase_N{round(N)}_{round(m)}.png')

    # fig, ax = plt.subplots(1, 1, figsize=(4,4))
    # plt.plot(timescale,normx_t,'k-o')
    # ax = plt.gca()
    # # ax.set_ylim([0.0,1.0])
    # ax.set_ylim(bottom=0)
    # plt.savefig(f'norm_psix_N{round(N)}_xi0{round(gamma*10)}.png')
    
    fig, ax = plt.subplots(1, 1, figsize=(5,4))
    plt.plot(timescale,squ0_t,'r--o')
    plt.plot(timescale,squ1_t,'b--o')
    plt.plot(timescale,invQFI_t,'k-o')
    ax = plt.gca()
    # ax.set_ylim([0.0,1.0])
    ax.set_ylim(bottom=0)
    plt.savefig(f'squeezing_time_m0_N{round(N)}_xi0{round(gamma*10)}.png')
    # print(invQFI_t[Nt-1]-invQFI_t[Nt-2])
    
    # fig, ax = plt.subplots(1, 1, figsize=(5,4))
    # plt.plot(timescale[1:Nt-1],np.log10(squ1_t[1:Nt-1]-invQFI_t[1:Nt-1]),'k-o')
    # ax = plt.gca()
    # # ax.set_ylim([0.0,1.0])
    # # ax.set_ylim(bottom=0)
    # plt.savefig(f'squeezing_diff_time_m0_N{round(N)}_xi0{round(gamma*10)}.png')

    vec0 = squ1_t[1:Nt]-invQFI_t[1:Nt]
    # vec1 = moving_average(vec0,Nt-1,int(np.floor(Nt/10)))
    # size_vec1 = np.size(vec1)

    # fig, ax = plt.subplots(1, 1, figsize=(5,4))
    # plt.plot(vec1,'k-o')
    # ax = plt.gca()
    # # ax.set_ylim([0.0,1.0])
    # # ax.set_ylim(bottom=0)
    # plt.savefig(f'squeezing_diff_moveave_m0_N{round(N)}_xi0{round(gamma*10)}.png')

    # fig, ax = plt.subplots(1, 1, figsize=(5,4))
    # plt.plot(timescale,QFI_phase_t,'k-o')
    # ax = plt.gca()
    # # ax.set_ylim([0.0,1.0])
    # # ax.set_ylim(bottom=0)
    # plt.savefig(f'QFI_phase_time_m0_N{round(N)}_xi0{round(gamma*10)}.png')

    # return Nt, timescale, squ1_t, invQFI_t
    return vec0

# time_evolution_m0_wigner(10, 0.3, 11, 0.0, 2*np.pi)

# time_evolution_m0_wigner(50, 0.3, 11, 0.0, 2*np.pi)

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
