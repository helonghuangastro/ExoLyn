'''
Rename the functions file to functions.py, compatible with relaxation_v6
Calculating the numerical difference by adding different contribution, to avoid big number + small number
question left: Did not use different density for condensate with different composition
'''
import numpy as np
import parameters as pars
import constants as cnt
from scipy.special import erf
import pdb
from chemistry import cal_gibbs

class sim_cache_single():
    def __init__(self):
        return

class sim_cache():
    def __init__(self):
        self.name = []
        return

    def setvalue(self, attr, value):
        setattr(self, attr, value)
        self.name.append(attr)
        return

    def have(self, name):
        if hasattr(self, name):
            return True
        else:
            return False
    def get(self, name):
        if self.have(name):
            return getattr(self, name)
        else:
            print('Did not find ' + name + ' in cache!')
            return None

    def __getitem__(self, i):
        cache_single = sim_cache_single()
        for attr in self.name:
            if self.get(attr).ndim==1:
                setattr(cache_single, attr, self.get(attr)[i])
            elif self.get(attr).ndim==2:
                setattr(cache_single, attr, self.get(attr)[:, i])
            else:
                print('Not a numpy.ndarray object')
        return cache_single


def E(atmosphere, **kwargs):
    '''
    return the error value of the equations
    logP: grid points
    y: [xc, xv, xn]
    cache_grid: cache value for attributes only depend on grid, on the grids
    cache_mid:  cache value for attributes only depend on grid, on middle points of grids
    **kwargs: controlling parameters, fsed, fdif, fcog, etc...
    '''

    dx = atmosphere.dx
    reactions = atmosphere.chem.reactions
    mugas = atmosphere.chem.mugas
    mucond = atmosphere.chem.musolid

    Kzz = pars.Kzz
    fdif = kwargs['fdif']
    fsed = kwargs['fsed']

    xc = atmosphere.xc
    xv = atmosphere.xv
    xn = atmosphere.xn

    rhop = atmosphere.rho

    Tmid   = atmosphere.cachemid.T_grid
    rhomid = atmosphere.cachemid.rho_grid
    Tarr   = atmosphere.cachegrid.T_grid
    rhoarr = atmosphere.cachegrid.rho_grid

    # calculate the shared perfactor for diffusion term
    pref_dif = pars.Kzz*rhomid*pars.mgas*pars.g / (cnt.kb*Tmid)
    # the prefactor for advection term (sedimentation velocity)
    v_sedarr = atmosphere.v_sed
    pref_adv = rhoarr * v_sedarr * fsed
    # the prefactor for source term
    pref_src = cnt.kb * Tarr / (pars.mgas * pars.g)

    # error for xc, except source
    excdif = edif(pref_dif, atmosphere.xc, dx) * fdif
    excadv = eadv(pref_adv, atmosphere.xc, dx)

    # error for xv
    exvdif = edif(pref_dif, atmosphere.xv, dx) * fdif

    # error for xn
    ###CWO: I see this line also later... consider making it a function
    deltv = -0.5 * atmosphere.v_sed * fsed    # collision velocity due to sedimentation
    t_coag_inv = cal_t_coag_inv(atmosphere.ap, rhop, atmosphere.np, atmosphere.cachegrid, deltv)
    exndif = edif(pref_dif, atmosphere.xn, dx) * fdif
    exnadv = eadv(pref_adv, atmosphere.xn, dx)
    exnsrc = get_exnsrc(pref_src, atmosphere.xn, atmosphere.cachegrid, t_coag_inv)
    exnsrc[0] = 0

    # error for source terms
    econ = (atmosphere.Sc * pref_src)    # no source term for the boundary condition
    econ[:, 0] = 0
    excsrc = np.zeros((atmosphere.ncond, atmosphere.N-1))
    exvsrc = np.zeros((atmosphere.ngas, atmosphere.N-1))
    for i, reaction in enumerate(reactions):
        solidindex = reaction.solidindex
        excsrc[solidindex] += econ[i, :-1]
        exvsrc -= np.atleast_2d(reaction.gasst * mugas).T / mucond[solidindex] * econ[i, :-1]

    # error for boundary condition
    excdif = np.hstack((pars.Kzz/pref_src[0] * rhoarr[0] * fdif * np.diff(xc[:, :2])/dx**2, excdif))
    excadv = np.hstack((rhoarr[0] * np.atleast_2d(xc[:, 0]).T*v_sedarr[0]*fsed / dx, excadv))
    exvdif = np.hstack((pars.Kzz/pref_src[0] * rhoarr[0] * fdif * np.diff(xv[:, :2])/dx**2, exvdif))
    exndif = np.hstack((pars.Kzz/pref_src[0] * rhoarr[0] * fdif * np.diff(xn[:2])/dx**2, exndif))
    exnadv = np.hstack((rhoarr[0] * xn[0]*v_sedarr[0]*fsed / dx, exnadv))

    exc = excdif + excadv + excsrc
    exv = exvdif + exvsrc
    exn = exndif + exnadv + exnsrc[:-1]

    return np.vstack((exc, exv, exn))

def dEdy(atmosphere, **kwargs):
    '''
    calculate Jacobian matrix, should return a (N-1, nvar, 3*nvar) tensor.
    fixxn functionality is disabled in this version of code.
    '''
    fdif = kwargs['fdif']
    fsed = kwargs['fsed']
    ncond = atmosphere.ncond
    ngas = atmosphere.ngas
    dx = atmosphere.dx
    N = atmosphere.N
    nvar = ncond + ngas + 1
    J = np.zeros((N-1, nvar, 3*nvar))

    cache_mid = atmosphere.cachemid
    cache_grid = atmosphere.cachegrid
    Tmid   = cache_mid.T_grid
    rhomid = cache_mid.rho_grid

    # calculate diffusion terms
    pref_dif = pars.Kzz*rhomid*pars.mgas*pars.g / (cnt.kb*Tmid) * fdif
    pref_dif_b = pars.Kzz*pars.mgas*pars.g*cache_grid.rho_grid[0] / (cnt.kb*cache_grid.T_grid[0]) * fdif    # prefactor for diffusion at the boundary
    for i in range(1, N-1):    # diffusion involves 3 nabouring terms
        J[i, np.arange(nvar), nvar+np.arange(nvar)] = - (pref_dif[i] + pref_dif[i-1])/dx**2
        J[i, np.arange(nvar), np.arange(nvar)] = pref_dif[i-1]/dx**2
        J[i, np.arange(nvar), 2*nvar + np.arange(nvar)] = pref_dif[i]/dx**2
    J[0, np.arange(nvar), nvar+np.arange(nvar)] = - pref_dif_b / dx**2
    J[0, np.arange(nvar), 2*nvar+np.arange(nvar)] = pref_dif_b / dx**2

    # calculate advection terms
    v_sed0 = atmosphere.v_sed
    y0 = atmosphere.y
    dy = np.maximum(1e-15, np.abs(y0)*1e-3)
    sedidx = np.append(np.arange(ncond), ncond+ngas)    # index that has an influence on the sedimentation (xc and xn)
    for i in sedidx:
        ynew = y0.copy()
        ynew[i] += dy[i]
        dvseddxc = cal_dvsed(ynew, cache_grid, v_sed0, atmosphere.chem.rhosolid)/dy[i]
        none_diag = dvseddxc * y0[sedidx] * cache_grid.rho_grid / dx * fsed    # non-diagonal terms
        J[1:, sedidx, nvar+i] += none_diag[:, 1:-1].T
        J[1:, sedidx, i] -= none_diag[:, :-2].T
        J[0, sedidx, nvar+i] += none_diag[:, 0]
        J[1:, i, nvar+i] += cache_grid.rho_grid[1:-1] * v_sed0[1:-1] / dx * fsed
        J[1:, i, i] -= cache_grid.rho_grid[:-2] * v_sed0[:-2] / dx * fsed
        J[0, i, nvar+i] += v_sed0[0] * cache_grid.rho_grid[0] / dx * fsed

    # source terms
    Tarr = cache_grid.T_grid
    pref_src = cnt.kb * Tarr / (pars.mgas * pars.g)
    econ = (atmosphere.Sc * pref_src)    # no source term for the boundary condition
    econ[:, 0] = 0
    deltv = -0.5 * atmosphere.v_sed * fsed    # collision velocity due to sedimentation
    t_coag_inv = cal_t_coag_inv(atmosphere.ap, atmosphere.rho, atmosphere.np, cache_grid, deltv)
    exnsrc = get_exnsrc(pref_src, atmosphere.xn, cache_grid, t_coag_inv)
    exnsrc[0] = 0

    desrcdy = np.zeros((nvar, nvar, N))
    for i in range(nvar):
        ynew = y0.copy()
        ynew[i] += dy[i]
        desrc = dEdysrc(atmosphere, ynew, pref_src, econ, exnsrc, **kwargs)
        desrcdy[i] = desrc/dy[i]

    # Analytically calculate dexnsrc/dxn
    ynew = y0.copy()
    ynew[-1] += dy[-1]
    xc = np.atleast_2d(ynew[:ncond])
    xn = ynew[-1]
    rhopnew = (xc.sum(axis=0) + xn) / ((xc/np.atleast_2d(atmosphere.chem.rhosolid).T).sum(axis=0) + xn/pars.rho_int)
    apnew = cal_ap(xc, xn, rhopnew)
    npnew = cal_np(xn, cache_grid)
    deltvnew = -0.5 * cal_vsed(apnew, rhopnew, cache_grid) * fsed    # to be changed
    t_coag_invnew = cal_t_coag_inv(apnew, rhopnew, npnew, cache_grid, deltvnew)
    dt_coag_invdxn = (t_coag_invnew-t_coag_inv)/(dy[-1])
    dexnsrcdxn = t_coag_inv + y0[-1]*dt_coag_invdxn
    dexnsrcdxn *= -pref_src * cache_grid.rho_grid
    dexnsrcdxn[0] = 0
    desrcdy[-1, -1] = dexnsrcdxn

    J[:, :, nvar:(2*nvar)] += np.swapaxes(desrcdy[:,:,:-1], 0, 2)

    return J

def cal_dvsed(y1, cache, v_sed0, rhosolid):
    ncond = len(pars.solid)

    xc = np.atleast_2d(y1[:ncond])
    xn = y1[-1]
    xcpos = np.maximum(xc, 0)
    rhop = (xcpos.sum(axis=0) + xn) / ((xcpos/np.atleast_2d(rhosolid).T).sum(axis=0) + xn/pars.rho_int)
    aparr = cal_ap(xcpos, xn, rhop)
    v_sed1 = cal_vsed(aparr, rhop, cache)
    dv_sed = v_sed1-v_sed0
    return dv_sed

def dEdysrc(atmosphere, y1, pref_src, econ0, exnsrc0, **kwargs):
    reactions = atmosphere.chem.reactions
    rhosolid = atmosphere.chem.rhosolid

    # calculate new exnsrc
    ncond = atmosphere.ncond
    ngas = atmosphere.ngas

    xc = np.atleast_2d(y1[:ncond])
    xv = np.atleast_2d(y1[ncond:(ncond+ngas)])
    xn = y1[-1]
    xcpos = np.maximum(xc, 0)    # physical (none-negative) xc to be used in certain places
    xvpos = np.maximum(xv, 0)
    xnpos = np.maximum(xn, 0)

    cache_grid = atmosphere.cachegrid
    Tarr   = cache_grid.T_grid
    rhoarr = cache_grid.rho_grid
    Snarr  = cache_grid.Sn_grid

    rhop = (xcpos.sum(axis=0) + xn) / ((xcpos/np.atleast_2d(rhosolid).T).sum(axis=0) + xn/pars.rho_int)
    aparr = cal_ap(xcpos, xn, rhop)
    n_parr = cal_np(xn, cache_grid)    # could be <0
    bs = xcpos / (xcpos.sum(axis=0) + xn) * rhop/np.atleast_2d(rhosolid).T
    v_sedarr = cal_vsed(aparr, rhop, cache_grid)
    deltvarr = -0.5 * v_sedarr * kwargs['fsed']    # collision velocity due to sedimentation
    t_coag_invarr = cal_t_coag_inv(aparr, rhop, n_parr, cache_grid, deltvarr)
    exnsrc = get_exnsrc(pref_src, xn, cache_grid, t_coag_invarr)
    exnsrc[0] = 0

    # calculate new econ
    mugas = np.atleast_2d(atmosphere.chem.mugas).T
    mucond = atmosphere.chem.musolid
    Sc2 = cal_Sc_all(xv, aparr, n_parr, bs, atmosphere.chem, cache_grid)
    econ = (Sc2 * pref_src)    # no source term for the boundary condition
    econ[:, 0] = 0

    dexcsrc = np.zeros((ncond, len(xn)))
    dexvsrc = np.zeros((ngas, len(xn)))
    dexnsrc = exnsrc - exnsrc0

    decon = econ-econ0
    for i, reaction in enumerate(reactions):
        solidindex = reaction.solidindex
        dexcsrc[solidindex] += decon[i]
        dexvsrc -= np.atleast_2d(reaction.gasst * mugas.T).T / mucond[solidindex] * decon[i]

    return np.vstack((dexcsrc, dexvsrc, dexnsrc))

def edif(pref_dif, xc, dx):
    return np.diff(pref_dif * np.diff(xc))/dx**2

def eadv(pref_adv, xc, dx):
    dim = np.ndim(xc)
    if dim>1:
        advterm = np.diff(pref_adv[:-1]*xc[:, :-1])/dx
    else:
        advterm = np.diff(pref_adv[:-1]*xc[:-1])/dx
    return advterm

def get_exnsrc(pref_src, xn, cache, t_coag_inv):
    Snarr = cache.Sn_grid
    rhoarr = cache.rho_grid
    return pref_src * (Snarr - xn*rhoarr*t_coag_inv)

def cal_Mn(P):
    return -pars.nuc_pro/2*(1-erf(-np.log(P/pars.P_star)/(np.sqrt(2)*pars.sigma_nuc)))

def bcb():
    return np.hstack((np.zeros(len(pars.solid)), pars.xvb, 0))

def TP(Parr):
    # Guillot 2010 expression for T-P profile
    if not hasattr(pars, 'TPmode') or pars.TPmode=='Guillot2010':
        opa_IR = pars.opa_IR    # opacity in IR
        g = pars.g    # gravitational acceleration
        opa_vis_IR = pars.opa_vis_IR    # opacity ratio between visual and IR

        T_int = pars.T_int    # interior temperature
        T_irr = pars.T_star * np.sqrt(pars.R_star/pars.rp)    # irradiation temperature

        opt_dep = opa_IR * Parr / g    # optical depth
        # Eq (1) in OrmelMin2019
        T = (3/4*T_int**4 * (2/3+opt_dep) + 3/4*T_irr**4*pars.firr * (2/3 + 1/(np.sqrt(3)*opa_vis_IR) + (opa_vis_IR-1/opa_vis_IR)*np.exp(-opa_vis_IR*opt_dep*np.sqrt(3))/np.sqrt(3)))**0.25
    # Interpolate from given T-P profile
    elif pars.TPmode=='interp':
        TPdata = np.genfromtxt(pars.TPfile, names=True, deletechars='', comments='#')
        logP = np.log10(Parr)
        logP_ref = np.log10(TPdata['P_ref'])

        T = np.interp(logP, logP_ref, TPdata['T_ref'])

    return T

def rho(Parr, Tarr):
    mgas = pars.mgas
    rhoarr = mgas * Parr / (cnt.kb * Tarr)
    return rhoarr

def v_th(Parr, Tarr):
    '''return thermal velocity'''
    return np.sqrt(8*cnt.kb*Tarr/(np.pi*pars.mgas))

def diffusivity(Parr, Tarr, v_tharr):
    '''Diffusivity, Eq (10) in OrmelMin2019'''
    return cnt.kb * Tarr * v_tharr / (3 * Parr * pars.cs_com)

def lmfp(Parr, rho_gasarr):
    '''mean free path'''
    return pars.mgas / (np.sqrt(2) * rho_gasarr * pars.cs_mol)

def Sn(Parr, rhoarr):
    '''source term for nuleation (particle numbers)'''
    Snarr = rhoarr * pars.g * pars.nuc_pro / (pars.sigma_nuc*Parr*np.sqrt(2*np.pi)) * np.exp(-np.log(Parr/pars.P_star)**2/(2*pars.sigma_nuc**2))
    Snarr = np.maximum(Snarr, 1e-200)    # 20230626: add lower limit to the Sn array
    return Snarr

def lnSn(Parr, rhoarr):
    '''return ln(Sn) for nuleation (particle numbers)'''
    lnSnarr = np.log(rhoarr * pars.g * pars.nuc_pro / (pars.sigma_nuc*Parr*np.sqrt(2*np.pi))) - np.log(Parr/pars.P_star)**2/(2*pars.sigma_nuc**2)
    lnSnarr = np.maximum(lnSnarr, np.log(1e-200))    # 20230626: add lower limit to the lnSn array
    return lnSnarr

def cal_np(xn, cache):
    '''particle number density'''
    rho_gas = cache.rho_grid
    n_parr = xn * rho_gas / pars.mn0
    n_parr = np.maximum(n_parr, 0)
    return n_parr

def cal_ap(xc, xn, rho):
    '''particle radius
    may need to think about when to set particle size to default size'''
    xctot = xc.sum(axis=0)
    xctot = np.maximum(xctot, 0)
    mp = (xctot+xn) * pars.mn0 / xn
    # mp = np.array([mp])    # to be compatible with scalar input
    # mp[np.where((np.array([xn])<=0)|(np.array([xc1])+np.array([xc2])<=0))] = pars.mn0
    # mp = mp[0]
    return np.cbrt(3*mp/(4*np.pi*rho))

def cal_vsed(ap, rhop, cache):
    '''sedimentation velocity'''
    v_tharr = cache.v_th_grid
    rhoarr = cache.rho_grid
    return -pars.g * ap * rhop / (v_tharr * rhoarr) * np.sqrt(1 + (4*ap/(9*cache.lmfp_grid))**2)    # the last term accounts for Stokes regime, smoothed the transition

def cal_t_coag_inv(ap, rhop, n_p, cache, deltv=0):
    '''
    coagulation time scale, Eq. (12) in OrmelMin2019
    Passing rhop to this function may not be the best choice. 
    A better way is to pass mp, which will benefit the Jacobian calculation, allowing analytical evaluation.
    '''
    Tarr = cache.T_grid
    lmfparr = cache.lmfp_grid
    v_tharr = cache.v_th_grid
    rhoarr = cache.rho_grid

    mp = 4*np.pi/3 * rhop * ap**3
    vBM = np.sqrt(16*cnt.kb*Tarr/(np.pi*mp))
    eta = 0.5 * lmfparr * v_tharr * rhoarr    # dynamic viscosity at p.5 of OrmelMin2019
    Dp = cnt.kb * Tarr / (6*np.pi * eta * ap)

    #following OM19
    t_coag_inv = 2*np.pi * n_p * ap**2 * deltv + 2*np.pi * np.minimum(vBM*ap, Dp) * ap * n_p
    t_coag_inv *= pars.f_coag
    return t_coag_inv

def cal_Sc_all(xv, aparr, n_parr, bs, chem, cache):
    ''' To calculate the condensation rate for all of the reaction. 
        A more pythonic way to calculate it. '''
    # chemistry data
    gasst = chem.gasst
    mugas = np.atleast_2d(chem.mugas).T
    mucond = np.atleast_2d(chem.musolid[chem.solidindex]).T

    # xv-independent data
    rhoarr = cache.rho_grid
    v_tharr = cache.v_th_grid
    Di = cache.diffusivity_grid
    Sbase = cache.Sbase_grid
    N = len(xv[0])                # number of grids
    # nvapor = len(xv)              # number of vapor
    nrec = len(chem.reactions)    # number of reactions

    # xv-dependent quantity
    xv3d = np.atleast_3d(xv).repeat(nrec, axis=-1)         # These 3D tensors are (nvapor, N, nreactions)-shaped
    gasst3d = np.moveaxis(np.atleast_3d(gasst).repeat(N, axis=-1), 0, -1)
    S = Sbase * np.prod(xv3d**gasst3d, axis=0).T    # TBD: Maybe calculating log is easier because can call numpy matrix multiply
    rv = np.minimum(4*Di/(aparr*v_tharr), np.sqrt(pars.mgas/(mugas*cnt.mu)))    # vapor molecules impinging velocity, (nvapor, N)

    # how much solid will be formed assuming inpinging from each vapor
    inpingmol = xv3d * np.atleast_3d(rv) / (np.atleast_3d(mugas) * gasst3d)
    inpingkey = np.min(inpingmol, axis=0).T                # key molecule that limit the inpinging rate

    bsrec = np.array([bs[reaction.solidindex] for reaction in chem.reactions])
    Sc_term = pars.f_stick * rhoarr * (1 - bsrec/S) * np.pi* n_parr * aparr**2 * v_tharr * mucond * inpingkey

    return Sc_term

def cal_Sc(xv, aparr, n_parr, bs, gasst, solidindex, i, cache, mugas, mucond):
    ''' To calculate the condensation rate for each reaction. '''
    gasst = np.atleast_2d(gasst).T

    rhoarr = cache.rho_grid
    v_tharr = cache.v_th_grid
    Di = cache.diffusivity_grid
    Sbase = cache.Sbase_grid[i]

    S = Sbase * np.prod(xv**gasst, axis=0)    # should be careful: when two species becomes negative, S would be positive
    rv = np.minimum(4*Di/(aparr*v_tharr), np.sqrt(pars.mgas/(mugas*cnt.mu)))
    # relidx = np.where(gasst[:, 0]!=0)[0]    # gas species relevant to this reaction
    # argkey = np.argmin((xv/mugas/gasst*rv)[relidx], axis=0)    # find the critical species
    # argkey = relidx[argkey]
    argkey = np.argmin((xv/mugas/gasst*rv), axis=0)

    # if xv<0, make Sc=0
    # negxvidx = np.any(xv[relidx]<=0, axis=0)
    # negapidx = np.where(aparr<=0)[0]

    nu = np.choose(argkey, gasst)    # nu = gasst[argkey]?
    mu = np.choose(argkey, mugas)
    xvkey = np.choose(argkey, xv)
    rvkey = np.choose(argkey, rv)

    Sc_term = pars.f_stick * xvkey*rhoarr/nu * (1 - bs/S) * np.pi* n_parr * aparr**2 * v_tharr * rvkey * mucond[solidindex]/mu
    # Sc_term[negxvidx] = 0    # if xv<0, make Sc=0
    # Sc_term[negapidx] = 0    # if ap<0, make Sc=0
    return Sc_term

def cal_Sbase(P, T, chem):
    '''
    TBD: should check the netnu here
    calculate the base of supersaturation rate.
    S = x_v^nu Sbase, this is specific to one reaction
    '''
    Sbase = []
    for reaction in chem.reactions:
        # interpolate the gibbs energy difference. 
        # only use valid gibbs energy
        gibbsref = reaction.delG
        idxvalid = ~np.isnan(gibbsref)
        gibbsref = gibbsref[idxvalid]
        Tref     = chem.gibbsTref[idxvalid]
        delG = np.interp(T, Tref, gibbsref)    # gibbs energy diffrence

        # using fitting formular of the gibbs energy if necessary
        fitidx = (T<np.min(Tref)) | (T>np.max(Tref))
        if (fitidx==True).any():
            print('[funs.cal_Sbase]WARNING: No Gibbs energy data given at some temperatures. Extrapolating the Gibbs energy or fitting it.')
            delGfit = 0
            for molename, st in reaction.product.items():
                delGfit += cal_gibbs(chem, molename, T[fitidx]) * st
            for molename, st in reaction.reactant.items():
                delGfit -= cal_gibbs(chem, molename, T[fitidx]) * st
            delG[fitidx] = delGfit

        netnu = reaction.netnu
        munu = reaction.munu
        Sx = (P/pars.Pref)**netnu*(pars.mgas/cnt.mu)**netnu/munu*np.exp(-delG/(cnt.R*T))
        Sbase.append(Sx)
    Sbase = np.array(Sbase)
    return Sbase

def add_cache(cache, Parr, chem):
    cache.setvalue('T_grid', TP(Parr))
    cache.setvalue('rho_grid', rho(Parr, cache.T_grid))
    cache.setvalue('v_th_grid', v_th(Parr, cache.T_grid))
    cache.setvalue('diffusivity_grid', diffusivity(Parr, cache.T_grid, cache.v_th_grid))
    cache.setvalue('lmfp_grid', lmfp(Parr, cache.rho_grid))
    cache.setvalue('Sn_grid', Sn(Parr, cache.rho_grid))
    cache.setvalue('Sbase_grid', cal_Sbase(Parr, cache.T_grid, chem))
    return

def init_cache(Parr, chem):
    '''
    initialize cache, storing values that do not change with simulation
    '''
    Pmid = np.sqrt(Parr[1:]*Parr[:-1])
    cache = sim_cache()
    cache_grid = sim_cache()
    add_cache(cache_grid, Parr, chem)
    cache_mid = sim_cache()
    add_cache(cache_mid, Pmid, chem)
    cache.setvalue('cachegrid', cache_grid)
    cache.setvalue('cachemid', cache_mid)
    cache.setvalue('chem', chem)

    return cache
