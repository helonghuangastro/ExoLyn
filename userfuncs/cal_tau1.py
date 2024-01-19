'''
This script is user-defined to calculate equilibrium gas phase chemistry and then spectrum.
FastChem and petitRADTRANS is required in this script. Please download and install them before running:
FastChem: https://exoclime.github.io/FastChem/
petitRADTRANS: https://petitradtrans.readthedocs.io/en/latest/
'''
import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('../src')
import constants as cnt
import pdb
sys.path.append('../util')
import os

gridfile = '../test/test_cloudspecies/gridFe.txt'
from read import reconstruct
atmosphere, pars = reconstruct(gridfile)

from calmeff import cal_eff_m_all, writelnk
from calkappa import cal_opa_all

logP = atmosphere.grid
Parr = np.exp(logP)
Parrbar = Parr/1e6
cachegrid = atmosphere.cachegrid
Tarr = cachegrid.T_grid
y = atmosphere.y
rhop = atmosphere.rho

ncod = len(pars.solid)
ngas = len(pars.gas)

ap = atmosphere.ap
n_p = atmosphere.np
rhog = cachegrid.rho_grid

# calculate effective refractory index
bs = atmosphere.bs
wlenkappa = np.logspace(0, np.log10(20), 100)
mmat = cal_eff_m_all(bs, pars.solid, wlenkappa)
writelnk(mmat, wlenkappa, rhop, folder='meff')

# calculate kappa
opobj = cal_opa_all(ap, write=False, Nlam=len(wlenkappa), optooldir='/home/helong/software/optool')
kappaext = opobj['kext']

# calculate opacity
kappadata = kappaext/rhog*n_p*4*np.pi/3*ap**3

# calculate the optical depth
dx   = logP[1] - logP[0]
dtau = kappadata * rhog * cnt.kb * Tarr / (pars.mgas*pars.g) * dx    # optical depth at each grid point
taucum = np.cumsum(dtau, axis=1)    # cumulative opacity from atmosphere top to bottom
Ptau1 = np.empty_like(wlenkappa)         # tau_cloud = 1 surface at each wavelength
for i in range(len(wlenkappa)):
    idx = np.where(taucum[i]>1.)[0]
    if len(idx)>0:
        Ptau1[i] = Parrbar[idx[0]]
    else:
        Ptau1[i] = Parrbar[-1]

print('maximum P_tau1: ', np.max(Ptau1))
print('minimum P_tau1: ', np.min(Ptau1))

for root, dirs, files in os.walk('meff', topdown=False):
    for name in files:
        os.remove(os.path.join(root, name))
    for name in dirs:
        os.rmdir(os.path.join(root, name))
os.rmdir('meff')
os.remove('_tmpparameters.txt')
