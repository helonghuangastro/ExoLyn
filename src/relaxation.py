'''
Save the case that results in minimum error, so that we can retrieve it later
In ynew function, require the Error to at most increase by a factor of 10 in consecutive steps
TBD:
1. multiple reactions leading to one condensate
'''
import sys
import numpy as np
# read parameters
import parameters as pars
if len(sys.argv)>1:
    parafilename = sys.argv[1]
else:
    parafilename = 'parameters.txt'
pars.readparsfile(parafilename)

import functions as funs
import constants as cnt
from fmatrixsol import matrixsol
import init
import pdb
import atmosphere_class
import chemistry
import output
import time

init.set_params()

#[24.02.01]cwo: put it outside the if loop
sys.path.append(pars.rdir + '../util/')

# import the plot module
if pars.verboselevel>-2 and pars.plotmode!='none':
    from draw import myplot

class control():
    '''
    return: 100 -- try next iteration
              0 -- converged, try next parameter
             -1 -- exceed maximum iteration
             -2 -- did not converge to the desired error tolerance
    '''
    def __init__(self, mode='E', **kwargs):
        self.mode = mode    # which criterior to use to judge the status of the convergence
        self.status = 100
        self.count = 0
        # save the best y to retrieve it
        self.savebest = False
        if self.savebest == True:
            self.ybest = None

        # based on error array, judge the status of integration
        if mode == 'E':
            self.elast = 1    # last error
            self.Earr = np.array([])
            if 'elast' in kwargs.keys():
                self.elast = kwargs['elast']
            self.dummy = {'wait': 5, 'maxitr': 100}    # waiting before convergence and maximum iteration numbers
            for key in kwargs.keys():
                if key in self.dummy.keys():
                    self.dummy[key] = kwargs[key]
        
        # based on the change in the solution
        elif mode== 'y':
            self.ylast = None
            self.abserr = kwargs['abserr']    # absolute error
            self.relerr = kwargs['relerr']    # relative error
            self.dummy = {'wait': 10, 'maxitr': 100}
            for key in kwargs.keys():
                if key in self.dummy.keys():
                    self.dummy[key] = kwargs[key]
        return
    
    def update(self, **kwargs):
        ''' Add a new data to judge whether can end the convergence '''
        self.count += 1
        if self.mode == 'E':
            E = kwargs['E']
            if self.savebest == True and (E<self.Earr).all():
                # pdb.set_trace()
                y = kwargs['y']
                self.ybest = y.copy()
            self.Earr = np.append(self.Earr, E)
            if np.isnan(E):
                print('[relaxation]did not converge to the desired error tolerance.')
                self.status = -2
            elif self.count<self.dummy['wait']:
                self.status = 100
            else:
                if E>np.mean(self.Earr[-5:])*0.9999:
                    if np.min(self.Earr)<self.elast * 1.5:
                        self.status = 0
                    else:
                        print(f'[relaxation]did not converge to desired error tolerance. Error: {E}')
                        self.status = -2
                else:
                    if self.count>self.dummy['maxitr']:
                        if E<self.elast*1.5:
                            self.status = 0
                        else:
                            print(f'[relaxation]exceed maximum iteration {self.maxitr}')
                            self.status = -1
                    else:
                        self.status = 100

        elif self.mode == 'y':
            ynew = kwargs['ynew']
            if np.isnan(ynew).any():
                print(f'[relaxation]:NaN encountered.')
                self.status = -1
            elif self.count<self.dummy['wait']:
                self.status = 100
            else:
                abschange = np.abs(ynew-self.ylast)
                absflag = abschange<=self.abserr
                relflag = abschange<=np.abs(self.ylast)*self.relerr
                if (absflag | relflag).all():
                    self.status = 0
                elif self.count>self.dummy['maxitr']:
                    print('[relaxation]:exceed maximum iteration')
                    # pdb.set_trace()
                    self.status = -2
                else:
                    self.status = 100
            if self.status != -2:
                self.ylast = ynew.copy()
        return

    def clear(self, **kwargs):
        ''' clear the status '''
        self.status = 100
        self.count = 0
        if self.savebest == True:
            self.ybest = None
        if self.mode == 'E':
            self.Earr = np.array([])
            if 'elast' in kwargs.keys():
                self.elast = kwargs['elast']

        return

# If you want to use the old way to calculate dEdy (change y every 3 grid points), can refer to v7 edition of the code

def sol_f(X, B):
    '''
    Using fortran to solve the solution
    X: (N-1) * (3*9) list
    B: 3 * (N-1) matrix
    '''
    Bfor = np.asfortranarray(B)
    matrixsol.sol(np.asfortranarray(X), Bfor)
    return Bfor

def sol_py(X, B):
    '''
    solve for solutions in figure Gaussionelimination.odg
    X: (N-1) * (3*9) list
    B: 3 * (N-1) matrix
    '''
    N = len(X)
    nvar = len(B)
    for i in range(N):
        Xi = X[i]
        # eliminate until p2
        for j in range(nvar):
            if Xi[j, j+nvar]==0:
                pdb.set_trace()
            B[j, i] /= Xi[j, j+nvar]
            Xi[j, (j+nvar):] /= Xi[j, j+nvar]
            for k in range(j+1, nvar):
                B[k, i] -= Xi[k, j+nvar] * B[j, i]
                Xi[k, (j+nvar):] -= Xi[k, j+nvar] * Xi[j, (j+nvar):]

        # eliminate until p3
        for j in range(1, nvar):
            for k in range(j):
                B[k, i] -= B[j, i] * Xi[k, j+nvar]
                Xi[k, (j+nvar):] -= Xi[j, (j+nvar):] * Xi[k, j+nvar]

        # stop before the last block
        if i==N-1:
            X[i] = Xi
            break

        # eliminate until p4
        Xn = X[i+1]
        for j in range(nvar):
            for k in range(nvar):
                B[k, i+1] -= B[j, i] * Xn[k, j]
                Xn[k, :(2*nvar)] -= Xn[k, j] * Xi[j, nvar:]

        X[i] = Xi
        X[i+1] = Xn

    X[N-1][:, (2*nvar):] = 0
    # eliminate everything
    for i in range(N-1, 0, -1):
        Xi = X[i-1]

def newy(matsol, y0, atmosphere, alpha=1, **kwargs):
    '''
    solve for the guess for next iteration
    '''
    nvar = len(matsol)
    Emat = funs.E(atmosphere, **kwargs)
    Eold = np.mean(np.abs(Emat))

    crel = 1.
    cabs = 1e-10
    maxincrease = crel * np.abs(y0[:nvar, :-1]) + cabs
    maxdecrease = maxincrease / 2
    # maxincrease = np.minimum(maxincrease, 99*y0[:nvar, :-1])    # This line seems to be useless
    maxdecrease = np.minimum(maxdecrease, 0.99*y0[:nvar, :-1])
    dy = np.minimum(matsol,  maxincrease)
    dy = np.maximum(dy, -maxdecrease)

    i = 0
    while True:
        ynew = y0.copy()
        # alpha is a parameter to limit the step. When the iteration becomes stuck, 
        # be more careful on the iteration.  // CWO: unclear to me
        ynew[:nvar, :-1] += dy/2**i * alpha    

        # This line is necessary to limit the nuclei concentration, so that the particles will 
        # not be too large // CWO: you mean the 1e-50? Unclear
        ynew[-1, :-1] = np.maximum(ynew[-1, :-1], 1e-50) 
        atmosphere.update(ynew)
        Emat = funs.E(atmosphere, **kwargs)
        Enew = np.mean(np.abs(Emat))

        #CWO: why again do we have this condition?
        if Enew<Eold*1.2:
            break

        if i>=5:
            print(f'[relaxation.newy]WARNING: In the non-linear regime, no improvement after step reduction by factor {2**i}')
            
            if False:
                # this is a trial that try to find solution from the opposite direction ......
                for j in range(6):
                    # pdb.set_trace()
                    ytry = y0.copy()
                    ytry[:nvar, :-1] -= dy/2**j * alpha
                    ytry[-1, :-1] = np.maximum(ynew[-1, :-1], 1e-50)
                    atmosphere.update(ytry)
                    Emat = funs.E(atmosphere, **kwargs)
                    Enew = np.mean(np.abs(Emat))
                    # if we can find a better solution from the opposite direction ......
                    if Enew<=Eold:
                        # pdb.set_trace()
                        print('find a better solution when ascending gradient direction...')
                        break
                if Enew>Eold:
                    atmosphere.update(ynew)

            break
        i += 1
    
    return ynew

def postprocess(yn, ncond, ngas):
    ''' make xc and xv positive after one iteration step '''
    N = len(yn[0])
    for i in range(ncond):
        if (yn[i, :-1]<=0).any():
            bottomidx = np.where(yn[i]>0)[0][-1]+1
            ispos = yn[i, :bottomidx]>0
            smallestpos = np.min(yn[i, :bottomidx][ispos])
            isnonpos = np.append(~ispos, np.zeros(N-bottomidx, dtype=bool))
            # if isnonpos.any():
            #     pdb.set_trace()
            yn[i, isnonpos] = smallestpos
    for i in range(ngas):
        if (yn[ncond+i, :-1]<=0).any():
            ispos = yn[ncond+i, :-1]>0
            smallestpos = np.min(yn[ncond+i, :-1][ispos])
            isnonpos = np.append(~ispos, False)
            yn[ncond+i, isnonpos] = smallestpos

    # ispos = yn[-1, :-1]>0
    # smallestpos = np.min(yn[-1, :-1][ispos])
    # isnonpos = np.append(~ispos, False)
    # yn[-1, isnonpos] = smallestpos

    return yn

def adjust_upper(atmosphere, atmospheren, **kwargs):
    ''' automatically change the upper atmosphere '''
    Parr = np.exp(atmosphere.grid)
    xc_tot = np.sum(atmosphere.xc, axis=0)    # total solid concentration
    xc_tot_max = np.max(xc_tot)
    N = atmosphere.N
    chem = atmosphere.chem

    # extend the upper boundary when xc at the upper boundary is too large
    while(xc_tot[0]>xc_tot_max/1e10):
        Parr = np.logspace(np.log10(Parr[0])-1, np.log10(Parr[-1]), N)
        cache = funs.init_cache(Parr, chem)
        logP = np.log(Parr)
        ynew = np.empty_like(atmosphere.y)
        for i in range(len(ynew)):
            ynew[i] = np.interp(logP, atmosphere.grid, atmosphere.y[i])    # interpolate to get new y
        atmosphere.update_grid(logP, cache)
        atmosphere.update(ynew)
        for i in range(10):
            relaxation(funs.E, funs.dEdy, atmosphere, **kwargs)
        xc_tot = np.sum(atmosphere.xc, axis=0)
        xc_tot_max = np.max(xc_tot)

    # shrink the upper boundary when xc at the upper boundary is too small
    idx = np.where(xc_tot>=xc_tot_max/1e10)[0][0]
    Parr = np.logspace(np.log10(Parr[idx]), np.log10(Parr[-1]), N)
    cache = funs.init_cache(Parr, chem)

    # update the atmosphere class
    logP = np.log(Parr)
    ynew = np.empty_like(atmosphere.y)
    for i in range(len(ynew)):
        ynew[i] = np.interp(logP, atmosphere.grid, atmosphere.y[i])    # interpolate to get new y
    atmosphere.update_grid(logP, cache)
    atmospheren.update_grid(logP, cache)
    atmosphere.update(ynew)

    # Need more careful treatment. This '10' is arbitrary
    for i in range(10):
        relaxation(funs.E, funs.dEdy, atmosphere, **kwargs)

    return

def saveplot(atmosphere, N, alpha, **kwargs):
    ''' save a series of plot to debug '''
    import shutil
    for i in range(N):
        yn = relaxation(funs.E, funs.dEdy, atmosphere, alpha, **kwargs)
        myplot(atmosphere.Parr, yn, ncond, ngas, plotmode='save')
        shutil.move('result.png', './saveplot/result' + str(i) + '.png')

def relaxation(efun, dedy, atmosphere, alpha=1, fixxn=False, **kwargs):    
    """
    Calculate residual and Jacobi matrix. Solve the matrix equations and get next guess solution
    """
    # calculate E, B matrix
    Emat = efun(atmosphere, **kwargs)
    if fixxn:
        Emat = Emat[:-1]
    # calculate partial derivative matrix
    # The following matrix is shown in Figure 17.3.1, numerical recipe
    dEdymat = dedy(atmosphere, **kwargs)

    # Bmat is shown in Figure 17.3.1, numerical recipe
    Bmat = -Emat

    # solve for solution of y change
    matsol = sol_f(dEdymat, Bmat)

    # solve for new y in next setp
    ynew = newy(matsol, atmosphere.y, atmosphere, alpha, **kwargs)

    return ynew

def iterate(atmosphere, atmospheren, fparas, ctrl):
    '''
    A function to converge over a range of control parameters "fparas".
    Each fpara starts from an off state (fpara=0) whose solution converges. 
    A solution with fpara=1 is sought for.
    If the solution fails, fpara is reduced; if it is successful, fpara increases
    until the solution converges at fpara=1

    Parameters:
        atmosphere: atmosphere class for successfully converged atmosphere
        atmospheren: dummy atmosphere class used during iteration
        fparas: a list of fudge parameters to be converged over. The order matters
        ctrl: controlling class for convergence
        isplot: whether to plot the atmosphere profile after converging on each parameter
    '''
    t1 = time.time()
    kwargs = {}
    for fpara_name in fparas:
        kwargs[fpara_name] = 0
    for fpara_name in fparas:
        fsucc = 0
        ffail = np.array([1.])
        while(fsucc<1.):
            fpara = ffail[-1]
            alpha = 1 - 0.1*(-np.log10((fpara-fsucc)/fpara))    # alpha is a parameter to limit the step. When the iteration becomes stuck, i.e. fpara becomes close to fsucc, be more careful on the iteration.
            kwargs[fpara_name] = fpara
            print('[relaxation.iterate]:converging on ' + fpara_name + ' at ' + str(fpara))
            atmospheren.update(atmosphere.y.copy())
            niter = 0 
            while(ctrl.status==100):
                yn = relaxation(funs.E, funs.dEdy, atmospheren, alpha, **kwargs)
                ctrl.update(ynew=yn)
                niter += 1
            print('[relaxation.iterate]:conducted ', niter, ' iterations; exit status = ', ctrl.status)
            # plot if verbose='verbose'
            if pars.verboselevel >= 1:
                myplot(atmosphere.Parr, atmospheren.y, atmosphere.rho, ncond, ngas, plotmode=pars.plotmode)
            # successful case
            if ctrl.status==0:
                atmosphere.update(yn)
                ctrl.clear()
                ffail = np.delete(ffail, -1)
                fsucc = fpara
                if len(ffail)==0:
                    ffail = np.array([np.minimum(1., 10*fpara)])
                if fpara_name == 'fsed' and True:
                    adjust_upper(atmosphere, atmospheren, **kwargs)
            # failed case
            else:
                ctrl.clear()
                if fsucc==0:
                    ffail = np.append(ffail, ffail[-1]/10)
                else:
                    ffail = np.append(ffail, np.sqrt(fsucc*fpara))
                # when the change in the fudging parameter becomes too small, quit the program and label it as fail
                if fpara <= fsucc*1.0001 or fpara < 1e-10:
                    print('ABORTION: stuck at ' + fpara_name + ' = ' + str(fpara))
                    sys.exit(1)

        print('[relaxation]SUCCESS: converged on >> ' + fpara_name +' <<')
        if pars.verboselevel==0:
            myplot(atmosphere.Parr, atmosphere.y, atmosphere.rho, ncond, ngas, plotmode=pars.plotmode)

    t2 = time.time()
    return t2-t1

if __name__ == '__main__':
    t0 = time.time()
    init.check_input_errors ()

    #suppress warnings
    if pars.suppresswarnings=='all':
        import warnings
        warnings.filterwarnings('ignore')

    # chemistry in all the chemistry data
    chem = chemistry.chemdata(pars.gibbsfile)

    # find the boundary of the domain
    Parr, cache = init.findbound(pars.Pa, pars.Pb, pars.N, chem)

    logP = np.log(Parr)

    atmosphere = atmosphere_class.atmosphere(logP, pars.solid, pars.gas, cache)    # Atmosphere that has alchemistryy been converged
    atmospheren = atmosphere_class.atmosphere(logP, pars.solid, pars.gas, cache)    # Atmosphere class used in each iteration
    ncond = atmosphere.ncond
    ngas = atmosphere.ngas

    y0 = init.init(atmosphere, method='Newton')
    atmosphere.update(y0)
    telap = time.time() -t0
    print(f'[relaxation]:initialization finished in {telap:.2f} seconds')
    # pdb.set_trace()
    # plot the initial state
    if pars.verboselevel >= 0:
        myplot(atmosphere.Parr, atmosphere.y, atmosphere.rho, ncond, ngas, plotmode=pars.plotmode)

    ctrl = control(mode='y', abserr=1e-10, relerr=1e-3)    # This value matters, when relerr=1e-4, T=8000 case cannot converge

    telap = iterate(atmosphere, atmospheren, ['fdif', 'fsed'], ctrl)
    print(f'[relaxation]:iteration finished in {telap:.2f} seconds')

    if pars.verboselevel == -1:
        myplot(atmosphere.Parr, atmosphere.y, atmosphere.rho, ncond, ngas, plotmode=pars.plotmode)

    if pars.writeoutputfile:
        output.writeatm(atmosphere)


    #calculation of optical constants (optional)
    #first we check if we can do it
    calcoptical = pars.calcoptical
    if pars.calcoptical:
        import optical
        calcoptical, doptical = optical.prepare_optical (**pars.doptical)


    #we are all set
    if calcoptical:
        import calmeff, calkappa
        print('[relaxation]:now continue with calculating the effective medium indices...')
        mmat = calmeff.cal_eff_m_all (atmosphere.bs, pars.solid, doptical['wavelengthgrid'])
        calmeff.writelnk(mmat, doptical['wavelengthgrid'], atmosphere.rho, folder=doptical['dirmeff'])

        print('[relaxation]:using optool to calculate the opacities...')
        calkappa.cal_opa_all (atmosphere.ap, write=True, **doptical)
        print('[relaxation]:opacity data stored in ', doptical['dirkappa'])

