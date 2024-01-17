''' This is a file to read in the output data and reconstruct the atmosphere class '''
import numpy as np
from matplotlib import pyplot as plt

def reconstruct(gridfilename):
    '''
    This function is used to recover the status from a grid file: including atmosphere object, chem object and cache object.
    However, I haven't implemented the right way to construct cache.mid, so user cannot resume simulation from here.
    '''

    # read the parameters, reactions and cols of the grid file
    with open(gridfilename, 'r') as ipt:
        ipt.readline()
        parameterline = ipt.readline()
        reactionline  = ipt.readline()
        headerline    = ipt.readline()

    # process the lines
    parameterline = parameterline.strip('# ').strip().strip(';')
    reactionline  = reactionline[2:].strip().strip(';')
    headerline    = headerline.strip()[7:]

    # write a temperate parameter file
    parameterlist = parameterline.split('; ')
    reactionlist  = reactionline.split('; ')
    with open('_tmpparameters.txt', 'w') as opt:
        for paraitem in parameterlist:
            opt.write(paraitem + '\n')
        opt.write('============ put reactions here =============\n')
        for reacitem in reactionlist:
            opt.write(reacitem + '\n')

    import parameters as pars
    pars.readparsfile('_tmpparameters.txt')

    # read gas part of the parameter file
    colsname = headerline.split(' ')
    solid = [colname for colname in colsname if colname.endswith('(s)')]
    gas = colsname[colsname.index(solid[-1])+1:colsname.index('nuclei')]
    pars.gas = gas


    # read chemistry files
    import chemistry
    chem = chemistry.chemdata(pars.gibbsfile)


    # reconstruct cache object
    # read data from grid file
    data = np.genfromtxt(gridfilename)
    data = data.T
    grid = data[0]
    Parr = np.exp(grid)
    Tarr = data[1]
    Snarr = data[4]

    # create cache from midpoints and gridpoints
    import functions as funs
    cache = funs.sim_cache()
    cachegrid = funs.sim_cache()
    cachemid = funs.sim_cache()

    # add items to cachegrid
    cachegrid.setvalue('T_grid',           Tarr)
    cachegrid.setvalue('rho_grid',         funs.rho(Parr, Tarr))
    cachegrid.setvalue('v_th_grid',        funs.v_th(Parr, Tarr))
    cachegrid.setvalue('diffusivity_grid', funs.diffusivity(Parr, Tarr, cachegrid.v_th_grid))
    cachegrid.setvalue('lmfp_grid',        funs.lmfp(Parr, cachegrid.rho_grid))
    cachegrid.setvalue('Sn_grid',          Snarr)
    cachegrid.setvalue('Sbase_grid',       funs.cal_Sbase(Parr, Tarr, chem))

    # add items to cachemid
    def geomean(x):
        return np.sqrt(x[1:]*x[:-1])
    cachemid.setvalue('T_grid', geomean(Tarr))
    cachemid.setvalue('rho_grid', geomean(cachegrid.rho_grid))
    # cachemid.setvalue('v_th_grid', geomean(cachegrid.v_th_grid))
    # cachemid.setvalue('diffusivity_grid', geomean(cachegrid.diffusivity_grid))
    # cachemid.setvalue('lmfp_grid', geomean(cachegrid.lmfp_grid))
    # cachemid.setvalue('Sn_grid', geomean(cachegrid.Sn_grid))
    # cachemid.setvalue('Sbase_grid', geomean(cachegrid.Sbase_grid.T).T)

    # set cache object
    # Note: cachemid is empty by now. So it's not yet possible to restart simulation from a certain point
    cache.setvalue('cachegrid', cachegrid)
    cache.setvalue('cachemid',  cachemid)
    cache.setvalue('chem',      chem)


    # reconstruct atmosphere object
    solidnos = [solidname.strip('(s)') for solidname in solid]    # solid name without 's'
    import atmosphere_class
    atmosphere = atmosphere_class.atmosphere(grid, solidnos, gas, cache)
    y = data[colsname.index(solid[0]):colsname.index('nuclei')+1]
    atmosphere.update(y)

    return atmosphere, pars

if __name__ == '__main__':
    gridfilename = '../examples/hotjupiter/grid.txt'
    atmosphere, pars = reconstruct(gridfilename)
