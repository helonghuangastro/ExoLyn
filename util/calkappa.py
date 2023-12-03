import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('../')

import atmosphere_class
import functions as funs
import parameters as pars
import constants as cnt
import read

sys.path.append('/home/helong/software/optool')
import optool

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
