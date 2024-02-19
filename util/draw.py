import numpy as np
from matplotlib import pyplot as plt
import functions as funs
import parameters as pars
import constants as cnt
import pdb

def myplot_v2(Parr, y0, ncod, ngas, save=0):
    '''plot the condensate all together and mark the dominant species'''
    import matplotlib.colors as mcolors
    colorarr = list(mcolors.TABLEAU_COLORS.values())

    xc = y0[:ncod]
    xv = y0[ncod:(ncod+ngas)]
    xn = y0[-1]
    xctot = np.sum(xc, axis=0)

    s = funs.cal_ap(xc, xn)

    ax = plt.gca()
    axs = plt.twiny()

    axs.loglog(s*1e4, Parr/1e6, label='size', color=[0.7,0.7,0.7, 0.7], linewidth=3)

    xcdom = np.argmax(xc, axis=0)    # dominant species
    spedom = np.unique(xcdom)    # all the species that once dominate the xc
    change = np.where(np.diff(xcdom)!=0)[0]+1    # where the dominate species changes
    change = np.insert(change, 0, 0)
    change = np.append(change, len(xcdom)-1)

    labeled = []
    lnarr = []
    for i in range(len(change)-1):
        coloridx = np.where(spedom==xcdom[change[i]])[0][0]%10
        # pdb.set_trace()
        if xcdom[change[i]] in labeled:
            ax.loglog(xctot[change[i]:change[i+1]], Parr[change[i]:change[i+1]]/1e6, color=colorarr[coloridx])
        else:
            label = pars.solid[xcdom[change[i]]]+'(s)'
            labeled.append(xcdom[change[i]])
            l, = ax.loglog(xctot[change[i]:(change[i+1]+1)], Parr[change[i]:(change[i+1]+1)]/1e6, color=colorarr[coloridx], label=label)
            lnarr.append(l)

    for i in range(ngas):
        l, = ax.loglog(xv[i], Parr/1e6, label=pars.gas[i]+'(v)', linestyle='--')
        lnarr.append(l)

    l, = ax.loglog(xn, Parr/1e6, label='nuclei')
    lnarr.append(l)

    labs = [x.get_label() for x in lnarr]

    ax.set_xlabel('Mass concentration')
    ax.set_ylabel('Pressure (bar)')
    axs.set_xlabel(r'Particle size ($\mu$m)', color='grey')

    ax.set_xlim([1e-9, 1e-2])
    # ax.set_ylim([1e-4, 3e-2])
    axs.set_xlim([1.1e-4, 1.])
    ax.invert_yaxis()
    axs.tick_params(colors='grey', axis='x', which='both')
    axs.spines['top'].set_color('grey')
    plt.legend(lnarr, labs, loc=2, bbox_to_anchor=(1.03, 1.0), borderaxespad=0)

    plt.subplots_adjust(left=0.1, right=0.77, bottom=0.1, top=0.9, hspace=0, wspace=0)

    if save==1:
        plt.savefig('result_newstyle.png', dpi=288)
    plt.show()

    return

def myplot(Parr, y0, rhop, ncod, ngas, plotmode=0, **kwargs):
    '''plot the solid and gas concentrations'''
    s = funs.cal_ap(y0[:ncod], y0[-1], rhop)

    ax = plt.gca()
    axs = ax.twiny()

    xmax = np.nanmax(y0)
    xmin = xmax/1e7

    #only plot where we have a cloud 
    ii = y0[:ncod].sum(0)>xmin

    axs.loglog(s[ii]*1e4, Parr[ii]/1e6, label='size', color=[0.7,0.7,0.7, 0.7], linewidth=3)    # divide by 1e6 to convert cgs to bar

    # select certain species
    if 'gasplot' in kwargs.keys():
        gasplot = kwargs['gasplot']
    else:
        gasplot = pars.gas
    if 'solidplot' in kwargs.keys():
        solidplot = kwargs['solidplot']
    else:
        solidplot = pars.solid

    lnarr = []

    for i in range(ncod):
        if pars.solid[i] in solidplot:
            l, = ax.loglog(y0[i], Parr/1e6, label=pars.solid[i]+'(s)')
            lnarr.append(l)
    for i in range(ngas):
        if pars.gas[i] in gasplot:
            l, = ax.loglog(y0[ncod+i], Parr/1e6, label=pars.gas[i]+'(v)', linestyle='--')
            lnarr.append(l)

    l, = ax.loglog(y0[-1], Parr/1e6, label='nuclei', color='k')
    lnarr.append(l)


    labs = [x.get_label() for x in lnarr]

    ax.set_xlabel('Mass concentration')
    ax.set_ylabel('Pressure (bar)')
    axs.set_xlabel(r'Particle size ($\mu$m)', color='grey')

    apmax = np.nanmax(s*1e4)
    ax.set_xlim([xmin, xmax*1.5])
    axs.set_xlim([pars.an*1e4, apmax*1.5])
    ax.invert_yaxis()
    axs.tick_params(colors='grey', axis='x', which='both')
    axs.spines['top'].set_color('grey')
    plt.legend(lnarr, labs)
    plt.legend(lnarr, labs, loc=2, bbox_to_anchor=(1.03, 1.0), borderaxespad=0)

    plt.subplots_adjust(left=0.1, right=0.77, bottom=0.1, top=0.9, hspace=0, wspace=0)

    if plotmode=='all' or plotmode=='save':
        if 'savedir' not in pars.__dict__:
            savedir = './'
        else:
            savedir = pars.savedir
        plt.savefig(savedir + 'result.png', dpi=288)
    if plotmode=='all' or plotmode=='popup':
        plt.show()

    plt.clf()

    return

def plotflux(Parr, y0, ncod, ngas, save=0):
    '''
    plot the flux of each species and super saturation ratio
    '''
    logP = np.log(Parr)
    dx = logP[1]-logP[0]
    xc = np.atleast_2d(y0[:ncod])
    xv = np.atleast_2d(y0[ncod:(ncod+ngas)])
    xn = y0[-1]
    
    cache = funs.init_cache(Parr, pars.reactionfile)

    # calculate flux
    solidfluxes = []
    gasfluxes = []

    Tarr = cache.cachegrid.T_grid
    pref_dif = pars.Kzz*pars.mgas*pars.g / (cnt.kb*Tarr)

    ap = funs.cal_ap(xc, xn)
    vsed = funs.cal_vsed(ap, cache.cachegrid)
    for i in range(ncod):
        fluxsed = xc[i] * vsed
        fluxdif = np.gradient(xc[i], logP) * pref_dif
        solidfluxes.append(fluxsed+fluxdif)
    for i in range(ngas):
        fluxdif = np.gradient(xv[i], logP) * pref_dif
        gasfluxes.append(fluxdif)

    fig, axs = plt.subplots(1, 2, sharey=True)
    fig.subplots_adjust(wspace=0)

    lnarr = []
    for i in range(ncod):
        # plot flux
        flux = solidfluxes[i]
        l, = axs[0].loglog(-flux, Parr/1e6, label=pars.solid[i]+'(s)')
        axs[1].loglog(flux, Parr/1e6, color=l.get_color())
        lnarr.append(l)
    
    for i in range(ngas):
        flux = gasfluxes[i]
        l, = axs[0].loglog(-flux, Parr/1e6, label=pars.gas[i]+'(v)', linestyle='--')
        axs[1].loglog(flux, Parr/1e6, color=l.get_color(), linestyle='--')
        lnarr.append(l)

    labs = [x.get_label() for x in lnarr]
    axs[0].set_xlabel(r'-Mc/$\rho_{gas}$ ($cm\ s^{-1}$)')
    axs[0].set_ylabel('Pressure (bar)')
    axs[1].set_xlabel(r'Mc/$\rho_{gas}$ ($cm\ s^{-1}$)')

    axs[0].set_xlim([1e-12, axs[0].get_xlim()[1]])
    axs[1].set_xlim([1e-12, axs[1].get_xlim()[1]])
    axs[0].invert_yaxis()
    axs[0].invert_xaxis()
    axs[0].legend(lnarr, labs)

    if save==1:
        plt.savefig('flux.png', dpi=288)
    plt.show()

    return

def plotS(Parr, y0, ncod, ngas, save=0):
    logP = np.log(Parr)
    dx = logP[1]-logP[0]
    xc = np.atleast_2d(y0[:ncod])
    xv = np.atleast_2d(y0[ncod:(ncod+ngas)])
    xn = y0[-1]
    xcpos = np.maximum(xc, 0)

    cache = funs.init_cache(Parr, pars.reactionfile)

    bs = xcpos / (xcpos.sum(axis=0) + xn)
    # calculate supersaturation ratio
    S = np.zeros((ncod, len(Parr)))
    Sbase = cache.cachegrid.Sbase_grid
    for i, reaction in enumerate(cache.reactions):
        sidx = reaction.solidindex
        gasst = np.atleast_2d(reaction.gasst).T
        S[sidx] += Sbase[i] * np.prod(xv**gasst, axis=0)

    for i in range(ncod):
        # plot supersaturation ratio
        l, = plt.loglog(S[i], Parr/1e6, label=pars.solid[i]+'(s)')
        plt.loglog(bs[i], Parr/1e6, color=l.get_color(), linestyle='--', alpha=0.5, linewidth=4)
        # plt.loglog(S[i]/bs[i], Parr/1e6, label=pars.solid[i]+'(s)')

    plt.vlines(1., Parr[0]/1e6, Parr[-1]/1e6, linestyle=':', color='k')

    ax = plt.gca()
    ax.set_xlabel('S')
    ax.invert_yaxis()
    ax.set_xlim([1e-4, ax.get_xlim()[1]])
    ax.legend()

    if save==1:
        plt.savefig('supersaturation.png', dpi=288)
    plt.show()

    return

if __name__ == '__main__':

    ncod = len(pars.solid)
    ngas = len(pars.gas)

    Parr = np.load('Parr.npy')
    y0 = np.load('y0.npy')

    myplot(Parr, y0, ncod, ngas, save=1)
    # plotflux(Parr, y0, ncod, ngas, save=1)
    # plotS(Parr, y0, ncod, ngas, save=0)
    # myplot_v2(Parr, y0, ncod, ngas, save=1)
