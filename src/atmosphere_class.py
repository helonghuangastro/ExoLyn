'''
The class for an atmosphere
'''
import numpy as np
import parameters as pars
import constants as cnt
import functions as funs

class atmosphere():
    def __init__(self, grid, solid, gas, cache):
        self.grid = grid              # grid
        self.dx = grid[1]-grid[0]     # grid space
        self.N = len(grid)            # number of grid
        self.Parr = np.exp(self.grid)
        self.solid = solid            # list of solid
        self.gas = gas                # list of gas
        self.ncond = len(solid)       # number of solid
        self.ngas = len(gas)          # number of gas
        self.chem = cache.chem    # reactions
        self.y = np.zeros((self.ncond+self.ngas+1, self.N))
        self.cachegrid = cache.cachegrid
        self.cachemid = cache.cachemid
        return

    def update(self, y, do_update_property=True):
        self.y = y
        if do_update_property:
            self.update_property()
        return

    def update_property(self):
        '''
        update derived properties of the atmosphere
        TBD: add mp here
        '''
        ncond = self.ncond
        ngas = self.ngas
        self.Parr = np.exp(self.grid)
        y = self.y

        self.xc = np.atleast_2d(y[:ncond])
        self.xv = np.atleast_2d(y[ncond:(ncond+ngas)])
        self.xn = y[-1]
        self.xcpos = np.maximum(self.xc, 0)    # physical (none-negative) xc to be used in certain places
        self.xvpos = np.maximum(self.xv, 0)    # physical (none-negative) xc to be used in certain places
        self.xnpos = np.maximum(self.xn, 0)

        self.rho = (self.xcpos.sum(axis=0) + self.xn) / ((self.xcpos/np.atleast_2d(self.chem.rhosolid).T).sum(axis=0) + self.xn/pars.rho_int)    # internal density of each particle
        self.ap = funs.cal_ap(self.xcpos, self.xn, self.rho)    # 20230706: xcpos -> xc
        self.np = funs.cal_np(self.xnpos, self.cachegrid)    # could be <0
        self.bs = self.xcpos / (self.xcpos.sum(axis=0) + self.xn) * self.rho/np.atleast_2d(self.chem.rhosolid).T    # volume ratio of each species
        self.v_sed = funs.cal_vsed(self.ap, self.rho, self.cachegrid)

        Sc2 = funs.cal_Sc_all(self.xv, self.ap, self.np, self.bs, self.chem, self.cachegrid)
        self.Sc = Sc2
        
        return

    def update_grid(self, grid, cache):
        ''' Change the grid '''
        self.grid = grid
        self.dx = grid[1]-grid[0]
        self.cachegrid = cache.cachegrid
        self.cachemid = cache.cachemid

        return
