'''
This script is user-defined to calculate equilibrium gas phase chemistry and then spectrum.
FastChem and petitRADTRANS is required in this script. Please download and install them before running:
FastChem: https://exoclime.github.io/FastChem/
petitRADTRANS: https://petitradtrans.readthedocs.io/en/latest/
'''
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import RectBivariateSpline
from petitRADTRANS import Radtrans
from petitRADTRANS import nat_cst as nc
import sys
sys.path.append('/home/helong/software/miniforge3/lib/python3.10/site-packages')
import pyfastchem
sys.path.append('../../src')
import read
import constants as cnt
import parameters as pars
import functions as funs
import itertools
import pdb
sys.path.append('../../util')
from draw import myplot
from calmeff import cal_eff_m_all, writelnk
from calkappa import cal_opa_all

def write_element(elementgrid):
    ''' Write the elemental abundance to input.dat '''
    with open('input.dat', 'w') as opt:
        opt.write('# Cloud model\n')
        for element, abundance in elementgrid.items():
            opt.write(element + ' ' + str(abundance) + '\n')

# gibbsfolder = '../../'
chem = read.chemdata(pars.gibbsfile)
# other interesting species and their concentration
extragas = ['CO', 'H2', 'He', 'CH4', 'CO2']    # extra gas that may contribute to spectrum
extragascon = np.array([8e-3, 1, 0.33, 0, 0])
# all of the gas molecules to be considered in the spectrum calculation
gasmols = []
for gasname in extragas + pars.gas:
    gasmols.append(read.molecule(gasname))
# all of the elements that appear in the molecules
elementlist = {}    # elemental abundance due to extra species
for gasmol in gasmols:
    for element in gasmol.element.keys():
        if element not in elementlist:
            elementlist[element] = 0.
# the elemental abundance of the elements, only considering extra gas
for i in range(len(extragas)):
    gasmol = gasmols[i]
    num = extragascon[i] / gasmol.mu
    for element in gasmol.element.keys():
        elementlist[element] += num * gasmol.element[element]

ncod = len(pars.solid)
ngas = len(pars.gas)

data = np.genfromtxt('./grid.txt', skip_header=1)
data = data.T
Parr = np.exp(data[0])
Parrbar = Parr/1e6    # Parr should be in bar, rather than in cgs unit
y0 = data[1:]
cache = funs.init_cache(Parr, chem)
# prepare for the radiation transfer
ap = funs.cal_ap(y0[:ncod], y0[-1])
ap[-1] = ap[-2]
xc = y0[:ncod]
# xc = xc[[0, 1, 7, 9]]    # only choose cloud species that's included in petitRADTRANS
xn = y0[-1]
n_p = funs.cal_np(xn, cache.cachegrid)
bs = xc / (xc.sum(axis=0) + xn)
Tarr = cache.cachegrid.T_grid

ygasnew = np.empty((len(gasmols), len(Parr)))

# calculate elemental abundances
elementgrid = elementlist.copy()
# transfer to the log version
for element, abundance in elementgrid.items():
    elementgrid[element] = 0

# write the element file
write_element(elementgrid)

fastchem = pyfastchem.FastChem('./input.dat',
                               '/home/helong/software/FastChem/fastchem/input/logK/logK.dat',
                               1)

# find the index of the interesting species in the output file
gasindex = []
for i, gasmol in enumerate(gasmols):
    allelement = list(gasmol.element.keys())
    if len(allelement) == 1 and list(gasmol.element.values())[0]==1:
        index = fastchem.getGasSpeciesIndex(allelement[0])
        gasindex.append(index)
        continue
    for perm in itertools.permutations(np.arange(len(allelement))):
        Hill = ''
        for j in range(len(allelement)):
            element = allelement[perm[j]]
            Hill += element
            Hill += str(gasmol.element[element])
        index = fastchem.getGasSpeciesIndex(Hill)
        if index != pyfastchem.FASTCHEM_UNKNOWN_SPECIES:
            gasindex.append(index)
            break

input_data = pyfastchem.FastChemInput()
output_data = pyfastchem.FastChemOutput()

for j in range(len(Parr)):
    elementgrid = elementlist.copy()
    for i in range(ngas):
        gasmol = gasmols[i+len(extragas)]
        num = y0[i+ncod, j] / gasmol.mu
        for element in gasmol.element.keys():
            elementgrid[element] += num * gasmol.element[element]
    # transfer to the log version
    for element, abundance in elementgrid.items():
        elementgrid[element] = np.log10(abundance)+12

    write_element(elementgrid)
    fastchem = pyfastchem.FastChem('./input.dat',
                                '/home/helong/software/FastChem/fastchem/input/logK/logK.dat',
                                1)

    # Read input file
    input_data.temperature = np.array([cache.cachegrid.T_grid[j]])
    input_data.pressure = np.array([Parrbar[j]])

    # Calculate the chemical equilibrium
    fastchem_flag = fastchem.calcDensities(input_data, output_data)

    for i in range(len(gasmols)):
        # print(gasmols[i].name + ' ', output_data.number_densities[0][gasindex[i]])
        ygasnew[i, j] = gasmols[i].mu * cnt.mu * output_data.number_densities[0][gasindex[i]] / cache.cachegrid.rho_grid[j]

# plot the atmosphere after equilibrium chemistry calculation
ynew = np.vstack((y0[:ncod], ygasnew[len(extragas):], y0[-1]))
# myplot(Parr, ynew, ncod, ngas, plotmode='all')
# pdb.set_trace()

####### calculate effective refractory index #######
wlen = np.logspace(0, np.log10(20), 100)
mmat = cal_eff_m_all(bs, pars.solid, wlen)
writelnk(mmat, wlen, pars.rho_int, folder='meff')
# pdb.set_trace()

####### calculate kappa #######
opobj = cal_opa_all('meff', ap, write=False)
kappadata = opobj['kext']/cache.cachegrid.rho_grid*n_p*4*np.pi/3*ap**3

# calculate the optical depth
logP = data[0]
dx   = logP[1] - logP[0]
dtau = kappadata * cache.cachegrid.rho_grid * cnt.kb * cache.cachegrid.T_grid / (pars.mgas*pars.g) * dx    # optical depth at each grid point
taucum = np.cumsum(dtau, axis=1)    # cumulative opacity from atmosphere top to bottom
Ptau1 = np.empty_like(wlen)         # tau_cloud = 1 surface at each wavelength
for i in range(len(wlen)):
    idx = np.where(taucum[i]>1.)[0]
    if len(idx)>0:
        Ptau1[i] = Parrbar[idx[0]]
    else:
        Ptau1[i] = Parrbar[-1]

# plot the tau_cloud = 1 surface
plt.loglog(wlen, Ptau1, color='k')
ax = plt.gca()
ax.invert_yaxis()
ax.set_ylabel('Pressure (bar)')
ax.set_xlabel(r'wavelength $\mu$m')
ax.set_ylim([Parrbar[0], Parrbar[-1]])
plt.savefig('tau1.png', dpi=288)
plt.show()

import os
os.remove('input.dat')
