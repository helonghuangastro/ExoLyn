import numpy as np
from matplotlib import pyplot as plt
import os
import sys
sys.path.append('../')

import atmosphere_class
import functions as funs
import parameters as pars
import constants as cnt
import read

sys.path.append('/home/helong/software/optool')
import optool

def cal_opa(filename, ap, write=False, savefile=None):
    ''' ap is the particle size in cm '''
    p = optool.particle('~/software/optool/optool ' + filename  + f' -p 0.25 -a {ap*1e4} -l 1 20 100 -o coeff')
    if write:
        with open('coeff/'+savefile, 'w') as opt:
            opt.write('wavelength(micron) kappa_abs kappa_sca kappa_ext asymmetry_parameter\n')
            for j in range(100):
                opt.write(f'{p.lam[j]} {p.kabs[0,j]} {p.ksca[0,j]} {p.kext[0,j]} {p.gsca[0,j]}\n')
    return p

def cal_opa_all(folder, aparr, write=False):
    ''' calculate the opacity for all the wavelengths at all pressure '''
    N = len(aparr)
    wlen = np.logspace(0, np.log10(20), 100)

    kabsmat = np.empty((100, N))
    kscamat = np.empty((100, N))
    kextmat = np.empty((100, N))
    gscamat = np.empty((100, N))

    if write:
        if not os.path.exists('coeff'):
            os.mkdir('coeff')
    for i in range(N):
        p = cal_opa(folder+f'/{i}.lnk', aparr[i], write=write, savefile=f'{i}.txt')
        kabsmat[:, i] = p.kabs[0]
        kscamat[:, i] = p.ksca[0]
        kextmat[:, i] = p.kext[0]
        gscamat[:, i] = p.gsca[0]

    kappadata = {'wlen':wlen, 'kabs':kabsmat, 'ksca':kscamat, 'kext':kextmat, 'gsca':gscamat}
    return kappadata

if __name__ == '__main__':
    # read the data and calculate the particle size
    ngas = len(pars.gas)
    data = np.genfromtxt('../grid.txt', skip_header=1)
    data = data.T
    Parr = np.exp(data[0])
    Parrbar = Parr/1e6
    logP = np.log(Parr)

    chem = read.chemdata('../' + pars.gibbsfile)
    cache = funs.init_cache(Parr, chem)
    atmosphere = atmosphere_class.atmosphere(logP, pars.solid, pars.gas, cache)

    atmosphere.update(data[1:])

    for i in range(atmosphere.N):
        p = optool.particle(f'~/software/optool/optool meff/{i}.lnk -p 0.25 -a {atmosphere.ap[i]*1e4} -l 1 20 100 -o coeff')
        with open(f'coeff/{i}.txt', 'w') as opt:
            opt.write('wavelength(micron) kappa_abs kappa_sca kappa_ext asymmetry_parameter\n')
            for j in range(100):
                opt.write(f'{p.lam[j]} {p.kabs[0,j]} {p.ksca[0,j]} {p.kext[0,j]} {p.gsca[0,j]}\n')
