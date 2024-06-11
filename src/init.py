'''
Force the residual F to decrease over two steps
If you want to only solve the equation for xc, rather than xn, you can find the condnewtonold here.
'''
import numpy as np
import parameters as pars
import constants as cnt
import functions as funs
from scipy.integrate import quad
from scipy.optimize import root
import pdb
import sys


def set_params():
    ''' set more parameters for pars '''
    # special treatment for verbose
    if pars.verbose == 'silent' or pars.verbose == -2:
        pars.verboselevel = -2
    elif pars.verbose == 'quiet' or pars.verbose == -1:
        pars.verboselevel = -1
    elif pars.verbose == 'default' or pars.verbose == 0:
        pars.verboselevel = 0
    elif pars.verbose == 'verbose' or pars.verbose == 1:
        pars.verboselevel = 1

def check_input_errors ():

    if pars.xvb.shape[0]!=len(pars.gas):
        print('[init]ERROR:dimensions of >>xvb<< and >>gas<< in parameters.txt are inconsistent!')
        sys.exit()

    if pars.verbose not in ['silent','quiet','default','verbose']:
        print('[init]ERROR:invalid pars.verbose parameter!')
        sys.exit()


def condnewtonold(Parr, reactions, cache, nu, muv, murc, xn):
    '''
    Using Newton-Raphson method to calculate equilibrium concentration
    This is for condensates only, xn is not solved in this method
    '''
    def get_F(n, Sbase, xn, returnjac=False):
        '''get residual function (maybe also Jacobian) of ln(S/bs)'''
        F = np.log(Sbase)
        xvleft = xvb - np.matmul(n, nu)*muv    # a vector with n_v length
        nulogxv = np.sum(nu * np.log(xvleft), axis=1) # a vector with xr length
        xtot = xn + np.sum(n * murc)    # a scalar
        logbs = np.log(n * murc / xtot)    # a scalar with xn length (now xn=xr)
        F = np.log(Sbase) + nulogxv - logbs

        if returnjac==True:
            # calculate Jacobian matrix
            jac1 = -np.matmul(nu, (nu*muv/xvleft).T)
            jac2 = -np.eye(len(n))/n + murc/xtot
            jac = jac1 + jac2

        if returnjac:
            return F, jac
        else:
            return F

    def flag(n):
        ''' check whether the solution is acceptable'''
        xvleft = xvb - np.matmul(n, nu)*muv
        if (n<=0.).any():
            xcflag = False
        else:
            xcflag = True
        if (xvleft<=0.).any():
            xvflag = False
        else:
            xvflag = True
        return xcflag, xvflag

    def newton(Sbase, xn, iniguess):
        ''' Use Newton method to get the supersaturation ratio at one location'''
        while True:    # Reduce initial guess if it is not proper
            xvleft = xvb - np.matmul(iniguess, nu)*muv
            if (xvleft>0).all():
                break
            else:
                iniguess /= 2

        n = iniguess
        F = get_F(n, Sbase, xn)
        while(np.max(np.abs(F))>0.01):
            i = 0
            F, jac = get_F(n, Sbase, xn, returnjac=True)
            step = np.linalg.solve(jac, -F)
            while True:
                xcflag, xvflag = flag(n+step/2**i)
                if xcflag and xvflag:
                    n = n+step/2**i
                    break
                if i>=10:
                    print('[init]error in finding an appropraite solution')
                    pdb.set_trace()
                i=i+1

        return n

    xvb = pars.xvb
    nsolid = np.zeros((len(reactions), len(Parr)))
    Sbase = cache.cachegrid.Sbase_grid
    iniguess = 1e-6*np.ones(len(reactions))
    for i in range(len(Parr)-1):
        result = newton(Sbase[:, i], xn[i], iniguess)
        iniguess = result
        nsolid[:, i] = result
    for i in range(len(Parr)-3, -1, -1):
        result = newton(Sbase[:, i], xn[i], iniguess)
        iniguess = result
        nsolid[:, i] = result

    return nsolid

def condnewton(xv, Parr, reactions, cachegrid, nu, muv, murc, rhosolid):
    '''
    Using Newton-Raphson method to calculate equilibrium concentration
    '''
    def get_F(n, xv, Sbase, cache, returnjac=False):
        '''
        get residual function (maybe also Jacobian) of ln(S/bs)
        the last element is the residual for equation of nuclei ln(Sn*tcoag/xn/rhogas)
        '''
        xvleft = xv
        # xvleft = xvi - np.matmul(n, nu)*muv    # a vector with n_v length
        nulogxv = np.sum(nu * np.log(xvleft), axis=1) # a vector with xr length
        # xtot = xn + np.sum(n * murc/rhorel)    # a scalar
        xtot = np.sum(n * murc/rhorel)    # a scalar
        logbs = np.log(n * murc/rhorel / xtot)    # a scalar with xn length (now xn=xr)
        F = np.log(Sbase) + nulogxv - logbs

        # calculate F (residual) for nuclei
        # xc = murc * n
        # xcpos = np.maximum(xc, 0)
        # rhop = (np.sum(xc)+xn)/(np.sum(xc/rhorel)+xn) * pars.rho_int
        # ap = funs.cal_ap(xcpos, xn, rhop)
        # n_p = funs.cal_np(xn, cache)
        # t_coag = 1/funs.cal_t_coag_inv(ap, rhop, n_p, cache)
        # F = np.append(F, lnSn + np.log(t_coag) - np.log(xn*cache.rho_grid))

        if returnjac:
            # calculate Jacobian matrix
            jac1 = -np.matmul(nu, (nu*muv/xvleft).T)
            jac2 = -np.eye(len(n))/n + murc/rhorel/xtot
            jac = jac1 + jac2
            # add the last row and column (how change of the xn change the equations) to the Jacobian
            # jac = np.hstack((jac, -np.ones((len(n),1))/xtot))
            # jac = np.vstack((jac, np.zeros(len(n)+1)))

            # calculate dt_coag/dxi numerically
            # for i in range(len(n)):
            #     xcnew = xcpos.copy()
            #     xcnew[i] *= 1.001
            #     #rhopnew = (np.sum(xcnew)+xn)/(np.sum(xcnew/rhorel)+xn) * pars.rho_int
            #     rhopnew = (xcnew.sum()+xn)/((xcnew/rhorel).sum()+xn) * pars.rho_int
            #     apnew = funs.cal_ap(xcnew, xn, rhopnew)
            #     t_coag_new = 1/funs.cal_t_coag_inv(apnew, rhopnew, n_p, cache)
            #     jac[-1, i] = (np.log(t_coag_new) - np.log(t_coag)) / (0.001*xcpos[i])*murc[i]

            # calculate dt_coag/da numerically, then calculate dF[-1]/dn analytically
            # xnnew = xn*1.001
            # rhopnew = ((xcpos).sum()+xnnew)/((xcpos/rhorel).sum()+xnnew) * pars.rho_int
            # ap = funs.cal_ap(xcpos, xnnew, rhopnew)
            # n_p = funs.cal_np(xnnew, cache)
            # t_coag_new = 1/funs.cal_t_coag_inv(ap, rhopnew, n_p, cache)
            # jac[-1, -1] = (np.log(t_coag_new) - np.log(t_coag)) / (0.001*xn) - 1/xn

        if returnjac:
            return F, jac
        else:
            return F

    def flag(n, xvi):
        ''' check whether the solution is acceptable'''
        xvleft = xvi - np.matmul(n, nu)*muv
        #cwo: this is per definition satisfied now
        if (n<=0.).any():
            xcflag = False
        else:
            xcflag = True
        if (xvleft<=0.).any():#cwo: why "<="?
            xvflag = False
        else:
            xvflag = True
        return xcflag, xvflag

    def newn(step, n, xvi):
        ''' From the step and n and xn, find the new n '''

        #[24.02.01]cwo:calculate initial i
        #-not 100% sure of this
        #-I also realize that the real issue is of numerical precision
        ii = step<0
        if sum(ii)==0:
            imin = 0
        else:
            imin = max(0, np.ceil(max(-np.log(-n[ii]/step[ii])/np.log(2))))
        dum1 = xvi /muv -(n@nu)
        dum2 = step@nu
        karr = np.where(dum1/dum2>0,-np.log(dum1/dum2)/np.log(2),0)
        kmin = max(0, np.ceil(max(karr)))

        i = max(imin,kmin)
        # nnew = n + step/2**i

        # i = 0 #the standard
        while(True):
            nnew = n+step/2**i
            xvleft = xvi - np.matmul(nnew, nu)*muv
            if (nnew>0.).all():
                xcflag, xvflag= flag(nnew, xvi)
                if xcflag and xvflag:
                    #here we can check my assertion if we start w/ "i=0"
                    #assert(i>=imin)
                    #assert(i>=kmin)
                    #n = n+step[:-1]/2**i
                    n = nnew
                    break
            i = i+1
        
        xvnew = xvi - np.matmul(n, nu) * muv

        return nnew, xvnew

    def newntest(step, n, xv):
        ''' From the step and n and xn, find the new n '''
        ii = step<0
        if sum(ii)==0:
            imin = 0
        else:
            imin = max(0, np.ceil(max(-np.log(-n[ii]/step[ii])/np.log(2))))

        dum1 = xv /muv
        dum2 = step@nu
        karr = np.where(dum1/dum2>0,-np.log(dum1/dum2)/np.log(2),0)
        kmin = max(0, np.ceil(max(karr)))

        i = max(imin,kmin)
        newstep = step/2**i

        nnew = n + newstep
        xvnew = xv - np.matmul(newstep, nu) * muv

        return nnew, xvnew

    def newton(xvi, Sbase, iniguess, cache):
        ''' Use Newton method to get the supersaturation ratio at one location'''
        inin = iniguess
        while True:    # Reduce initial guess if it is not proper
            xvleft = xvi - np.matmul(inin, nu)*muv
            if (xvleft>0).all():
                break
            else:
                inin /= 2    # changed 20230904

        n = inin
        F = get_F(n, xvi, Sbase, cache)
        xv = xvi - np.matmul(inin, nu) * muv
        j = 0
        flagsingle = 0    # flag for the status of finding initial state: 0 for success; -1 for exceed maximum iteration; -2 for step becomes too samll
        Farr = []
        # pdb.set_trace()
        while(np.max(np.abs(F))>0.01):
            F, jac = get_F(n, xv, Sbase, cache, returnjac=True)
            Farr.append(np.max(np.abs(F)))
            step = np.linalg.solve(jac, -F)

            # if the step is too small (iteration stucks), break
            if np.max(np.abs(step/n))<1e-13:
                flagsingle = -2
                break

            # pdb.set_trace()
            # n, xv = newn(step, n, xvi)
            n, xv = newntest(step, n, xv)
            
            # when iterate for too long, break it
            j += 1
            if j>=500:
                flagsingle = -1
                break

            # when nan is encountered
            if np.any(np.isnan(F)):
                flagsingle = -3
                break

        return flagsingle, n, xv

    def interpfail(mat, failidx):
        ''' Interpolate failed grid points '''
        if failidx[0]==0:
            endi = 0
        else:
            endi = -1
        idx = np.where(np.diff(failidx)!=1)[0]+1    #idx is where the failidx is not continuous
        idx = np.append(idx, len(failidx))
        idx = np.insert(idx, 0, 0)
        for i in range(len(idx)-2, endi, -1):
            nfail = failidx[idx[i+1]-1]-failidx[idx[i]]+1    # number of failed grid points in each sector
            interplever = (np.arange(nfail)+1)/(nfail+1)     # leverage to interpolate the failed points from the neighbouring successful cases
            rightcontrib = np.atleast_2d(mat[:, failidx[idx[i+1]-1]+1]).T * interplever
            leftcontrib = np.atleast_2d(mat[:, failidx[idx[i]]-1]).T * (1-interplever)
            mat[:, failidx[idx[i]:idx[i+1]]] = rightcontrib + leftcontrib

        if failidx[0]==0:
            # process the first sector
            mat[:, :(failidx[idx[1]-1]+1)] = np.atleast_2d(mat[:, failidx[idx[1]-1]+1]).T

        return mat


    rhorel = rhosolid/pars.rho_int

    # xvb = pars.xvb
    nsolid = np.zeros((len(reactions), len(Parr)))
    Sbase = cachegrid.Sbase_grid
    iniguess = 1e-6*np.ones(len(reactions))

    nulogxv = np.matmul(nu, np.log(xv)) # a matrix of (xr, ngas) length
    SR = Sbase * np.exp(nulogxv)    # super saturation ratio for cloud base
    status = np.ones_like(Parr) * 100    # save the status of finding initial condition: 0 for success; -1 and -2 for fail; 100 for not done now
    Sfail = np.inf
    clearidx = np.where(SR.sum(axis=0)<1)[0]    # where the atmosphere is clear, without condensation

    # At this step preserve topidx and bottomidx, which is not as ugly as oversupersat parameter
    # loop over the cloud from upper down and try
    for i in range(len(Parr)):
        # set initial state
        # if last grid point success, use it as initial guess
        if status[i-1]==0:
            iniguess = nsolid[:, i-1]

        # solve for the problem
        # if S is too large, skip newton iteration
        # TBD: This is not directly solving the problem, need to think of a way to solve it
        try:
            flagsingle, result, resultxv = newton(xv[:, i], Sbase[:, i], iniguess, cachegrid[i])
        except (TypeError, np.linalg.LinAlgError):
            flagsingle = -1

        status[i] = flagsingle
        if flagsingle == 0:
            nsolid[:, i] = result
            xv[:, i] = resultxv
        elif flagsingle == -1 and SR[:, i].sum()>10:    # update Sfail if necessary
            Sfail = np.minimum(Sfail, SR[:, i].sum())

        print(f'\r[init]grid point {i}/{len(Parr)} evaluates to status {flagsingle}   ', end="")
    print()

    # insert under-saturated values
    idx = np.where((status!=0) & (SR.sum(axis=0)<10.))[0]    # where the atmosphere is undersaturated
    idxdiff = np.diff(idx)    # only do this for the last sector, because previous sectors can be inserted later.
    if np.any(idxdiff!=1):
        idxlastsector = np.where(idxdiff!=1)[0][-1]
        idx = idx[(idxlastsector+1):]
    # substitute the failed solid concentration
    if len(idx)!=0:
        for i in range(len(reactions)):
            nsolid[i, idx] = nsolid[i, idx[0]-1]
        for i in range(len(xv)):
            xv[i, idx] = xv[i, idx[0]-1]
        status[idx] = 1

    # insert super saturated values
    if (status<0).any():
        print('[init]WARNING: Failure to find initial guess for some grid points. Extrapolate successful grids')
        failidx = np.where(status<0)[0]
        nsolid = interpfail(nsolid, failidx)
        xv = interpfail(xv, failidx)

    xn = np.sum(np.atleast_2d(murc / rhorel).T * nsolid, axis=0)

    # set a proper lower boundary condition for xn and xc, if the SR at lower boundary is smaller than 1
    if SR[:, -1].sum()<10:
        if pars.f_coag!=0:
            ap = np.cbrt(3*pars.mn0/(4*np.pi*pars.rho_int))
            Sn = cachegrid.Sn_grid[-1]
            xn_btm = np.sqrt(pars.mn0 * Sn / funs.cal_t_coag_inv(ap, pars.rho_int, 1, cachegrid[-1])) / cachegrid.rho_grid[-1]
            nsolid[:, -1] = xn_btm * SR[:, -1] / murc
            xn[-1] = xn_btm
        else:
            nsolid[:, -1] = 0
            xn[-1] = 0

    return nsolid, xv, xn

def init (atmosphere, method):
    '''
    Change by now: use atmosphere rather than Parr; don't use cache
    '''
    Parr = atmosphere.Parr
    N = atmosphere.N

    ncod = atmosphere.ncond
    ngas = atmosphere.ngas
    chem = atmosphere.chem
    reactions = chem.reactions
    rhosolid = chem.rhosolid

    # prepare chemistry things
    nu = chem.gasst
    muv = chem.mugas
    murc = np.array([chem.molecules[reaction.solid+'(s)'].mu for reaction in reactions])

    y0 = np.zeros((ncod+ngas+1, N))
    # initial concentration without any condensation etc ...
    for i in range(ngas):
        y0[ncod+i] = pars.xvb[i]

    # calculate solid concentration 
    if method=='Newton':
        # My Newton method to calculate the equilibrium concentration of solids.
        nsolid, xv, xn = condnewton(y0[ncod:(ncod+ngas)], Parr, reactions, atmosphere.cachegrid, nu, muv, murc, rhosolid)
    else:
        raise Exception('Unknown method to calculate initial solid concentration')

    for i in range(len(reactions)):
        solidindex = reactions[i].solidindex
        y0[solidindex] += murc[i] * nsolid[i]
    y0[ncod:(ncod+ngas)] = xv

    # post process the gas to be larger than 0. Previous step may introduce numerical error, leading to negative or 0 where it should not.
    # TBD: is this necesssary?
    for i in range(ngas):
        if (y0[ncod+i]<=0).any():
            pdb.set_trace()
            ispos = y0[ncod+i]>0
            smallestpos = np.min(y0[ncod+i][ispos])
            y0[ncod+i, ~ispos] = smallestpos

    print('[init]SUCCESS: Find initial condition')

    # If in the future the code do not work, maybe this could save it.
    # is this really necessary? I tested default, Kzz=1e6, T=3000K, T=8000K, this is not needed.
    # for i in range(ngas):
    #     y0[ncod+i] = np.maximum(1e-16, y0[ncod+i])
    # is this really necessary? I tested default, Kzz=1e6, T=3000K, T=8000K, this is not needed.
    # y0[-1] = np.maximum(xn, y0[-1])
    y0[-1] = xn

    return y0

def findbound(Pa, Pb, N, chem):
    '''
    find an appropriate simulation domain and set up the cache data.
    The initial Pb is, by default, 2 times deeper than the sublimation front
    '''
    def getS(cache):
        '''
        Get the super saturation ratio of all the species.
        '''
        xvb = pars.xvb
        nu = chem.gasst
        nulogxv = np.sum(nu * np.log(xvb), axis=1) # a vector with xr length
        Sbase = cache.cachegrid.Sbase_grid
        SR = Sbase * np.atleast_2d(np.exp(nulogxv)).T    # super saturation ratio for cloud base
        return SR

    Parr = np.logspace(np.log10(Pa), np.log10(Pb), N)
    cache = funs.init_cache(Parr, chem)
    SR = getS(cache)
    if pars.autoboundary:
        while(SR.sum(axis=0)[-1]>1):
        # while(SR.sum(axis=0)[-1]>1 and cache.cachegrid.T_grid[-1]<=chem.gibbsTref[-1]):
            Pb *= 10
            Parr = np.logspace(np.log10(Pa), np.log10(Pb), pars.N)
            cache = funs.init_cache(Parr, chem)
            SR = getS(cache)

        # find the bottom of cloud where everything evaporates
        if np.all(SR.sum(axis=0)<1.):
            # if the super saturation ratio is too low, there will be no cloud
            print('[init.findbound]WARNING: total super saturation ratio smaller than 1 everywhere, the atmosphere is very clean.')
            bottomidx = N-1
        elif np.all(SR.sum(axis=0)>1.):
            print('[init.findbound]WARNING: total super saturation ratio larger than 1 everywhere, may artificially truncate the cloud.')
            bottomidx = N-1
        else:
            bottomidx = np.where((SR.sum(axis=0)[1:]<1) & (SR.sum(axis=0)[:-1]>1))[0][0]+1
        
        # set the final boundary of the atmosphere and initialize cache
        Pb = Parr[bottomidx] * 2
        Parr = np.logspace(np.log10(Pa), np.log10(Pb), pars.N)
        cache = funs.init_cache(Parr, chem)
        if cache.cachegrid.T_grid[-1] > chem.gibbsTref[-1]:
            print('[init.findbound]WARNING: No Gibbs energy data given at the highest temperature. Extrapolating the Gibbs energy.')
        SR = getS(cache)

    # check unimportant species whose super saturation ratio is always smaller than 1. User can remove that in the next run to speed up
    for i, reaction in enumerate(chem.reactions):
        if np.all(SR[i]<1.):
            print('[init.findbound]INFO: ' + reaction.solid + ' is an unimportant species with super saturation ratio < 1 everywhere.')
    
    print('[init]SUCCESS: finding domain for the problem.')

    return Parr, cache
