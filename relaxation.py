'''
Save the case that results in minimum error, so that we can retrieve it later
In ynew function, require the Error to at most increase by a factor of 10 in consecutive steps
TBD:
1. multiple reactions leading to one condensate
'''
import numpy as np
import parameters as pars
import functions as funs
import constants as cnt
from draw import myplot
from fmatrixsol import matrixsol
import init
import pdb
import atmosphere_class
import read
import sys

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
                print('did not converge to the desired error tolerance.')
                self.status = -2
            elif self.count<self.dummy['wait']:
                self.status = 100
            else:
                if E>np.mean(self.Earr[-5:])*0.9999:
                    if np.min(self.Earr)<self.elast * 1.5:
                        self.status = 0
                    else:
                        print(f'did not converge to desired error tolerance. Error: {E}')
                        self.status = -2
                else:
                    if self.count>self.dummy['maxitr']:
                        if E<self.elast*1.5:
                            self.status = 0
                        else:
                            print(f'exceed maximum iteration {self.maxitr}')
                            self.status = -1
                    else:
                        self.status = 100

        elif self.mode == 'y':
            ynew = kwargs['ynew']
            if np.isnan(ynew).any():
                print(f'NaN encountered.')
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
                    print('exceed maximum iteration')
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

def newy(matsol, y0, bcb, atmosphere, alpha=1, **kwargs):
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
    dy = np.minimum(matsol, maxincrease)
    dy = np.maximum(matsol, -maxdecrease)

    i = 0
    while(True):
        ynew = y0.copy()
        ynew[:nvar, :-1] += dy/2**i * alpha    # alpha is a parameter to limit the step. When the iteration becomes stuck, be more careful on the iteration.
        ynew[:nvar, -1] = bcb()[:nvar]
        ynew[-1, :-1] = np.maximum(ynew[-1, :-1], 1e-18)    # It seems that this line is necessary. I forget why it is.
        atmosphere.update(ynew)
        Emat = funs.E(atmosphere, **kwargs)
        Enew = np.mean(np.abs(Emat))
        if Enew<Eold*1.5:
            break
        if i>=5:
            print(f'WARNING: In the non-linear regime, reduce the step by {2**i}')
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

def relaxation(efun, dedy, bcb, atmosphere, alpha=1, fixxn=False, **kwargs):    
    # calculate E, B matrix
    Emat = efun(atmosphere, **kwargs)
    if fixxn:
        Emat = Emat[:-1]
    # calculate partial derivative matrix
    # The following matrix is shown in Figure 17.3.1, numerical recipe
    dEdymat = dedy(atmosphere, **kwargs)
    # pdb.set_trace()

    # Bmat is shown in Figure 17.3.1, numerical recipe
    Bmat = -Emat

    # solve for solution of y change
    matsol = sol_f(dEdymat, Bmat)

    # solve for new y in next setp
    ynew = newy(matsol, atmosphere.y, bcb, atmosphere, alpha, **kwargs)

    return ynew

def iterate(atmosphere, atmospheren, fparas, ctrl):
    '''
    A function for converging any fudging parameters.
    Parameters:
        atmosphere: atmosphere class for successfully converged atmosphere
        atmospheren: atmosphere class to be used during converging
        fparas: a list of fudging parameters to be converged, the order matters
        ctrl: controlling class for convergence
        isplot: whether plot the atmosphere structure after converging each parameter
    '''
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
            print('Converging ' + fpara_name + ' ' + str(fpara))
            atmospheren.update(atmosphere.y.copy())
            while(ctrl.status==100):
                yn = relaxation(funs.E, funs.dEdy, funs.bcb, atmospheren, alpha, **kwargs)
                ctrl.update(ynew=yn)
            # plot if verbose='verbose'
            if pars.verboselevel >= 1:
                myplot(Parr, atmospheren.y, ncond, ngas)
            if ctrl.status==0:
                atmosphere.update(yn)
                ctrl.clear()
                ffail = np.delete(ffail, -1)
                fsucc = fpara
                if len(ffail)==0:
                    ffail = np.array([np.minimum(1., 10*fpara)])
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

        print('SUCCESS: converging ' + fpara_name)
        if pars.verboselevel==0:
            myplot(Parr, atmosphere.y, ncond, ngas)

if __name__ == '__main__':
    # read in all the chemistry data
    chem = read.chemdata(pars.reactionfile, pars.gibbsfile)

    # find the boundary of the domain
    Parr, cache = init.findbound(pars.Pa, pars.Pb, pars.N, chem)

    logP = np.log(Parr)
    dx = logP[1]-logP[0]

    atmosphere = atmosphere_class.atmosphere(logP, pars.solid, pars.gas, cache)    # Atmosphere that has already been converged
    atmospheren = atmosphere_class.atmosphere(logP, pars.solid, pars.gas, cache)    # Atmosphere class used in each iteration
    ncond = atmosphere.ncond
    ngas = atmosphere.ngas

    y0 = init.init(atmosphere, method='Newton')
    atmosphere.update(y0)
    # plot the initial state
    if pars.verboselevel >= 0:
        myplot(Parr, atmosphere.y, ncond, ngas)

    ctrl = control(mode='y', abserr=1e-10, relerr=1e-3)    # This value matters, when relerr=1e-4, T=8000 case cannot converge

    iterate(atmosphere, atmospheren, ['fdif', 'fsed'], ctrl)

    if pars.verboselevel == -1:
        myplot(Parr, atmosphere.y, ncond, ngas)
