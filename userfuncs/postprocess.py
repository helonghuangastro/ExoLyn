'''
This script is user-defined to calculate equilibrium gas phase chemistry and then spectrum.
FastChem and petitRADTRANS is required in this script. Please download and install them before running:
FastChem: https://exoclime.github.io/FastChem/
petitRADTRANS: https://petitradtrans.readthedocs.io/en/latest/
'''
import numpy as np
from matplotlib import pyplot as plt
from petitRADTRANS import Radtrans
from petitRADTRANS import nat_cst as nc
import sys
sys.path.append('/home/helong/software/miniforge3/lib/python3.10/site-packages')
import pyfastchem
sys.path.append('../')
import read
import constants as cnt
import parameters as pars
import functions as funs
import itertools
import pdb
from draw import myplot

def write_element(elementgrid):
    ''' Write the elemental abundance to input.dat '''
    with open('input.dat', 'w') as opt:
        opt.write('# Cloud model\n')
        for element, abundance in elementgrid.items():
            opt.write(element + ' ' + str(abundance) + '\n')

chem = read.chemdata(pars.rdir + pars.gibbsfile)
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

data = np.genfromtxt('../grid.txt', skip_header=1)
data = data.T
Parr = np.exp(data[0])
Parrbar = Parr/1e6    # Parr should be in bar, rather than in cgs unit
y0 = data[1:]
cache = funs.init_cache(Parr, chem)
# prepare for the radiation transfer
ap = funs.cal_ap(y0[:ncod], y0[-1])
ap[-1] = ap[-2]
xc = y0[:ncod]
xc = xc[[0, 1, 7, 9]]    # only choose cloud species that's included in petitRADTRANS
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
myplot(Parr, ynew, ncod, ngas, plotmode='all')
pdb.set_trace()

####### Radiation transfer part #######
linespecies = ['H2O_HITEMP', 'CO_all_iso_HITEMP', 'H2S', 'Mg', 'SiO', 'Fe', 'CO2', 'CH4', 'TiO_all_Exomol', 'Al']
cloud_species = ['Mg2SiO4(c)_cm', 'MgSiO3(c)_cm', 'Fe(c)_cm', 'Al2O3(c)_cm']
atmosphere = Radtrans(line_species=linespecies, cloud_species=cloud_species, rayleigh_species=['H2', 'He'], continuum_opacities = ['H2-H2', 'H2-He'], wlen_bords_micron = [1, 20])
atmosphere.setup_opa_structure(Parrbar)

mass_fractions = {}
mass_fractions['H2'] = ygasnew[1]
mass_fractions['He'] = ygasnew[2]
mass_fractions['H2O_HITEMP'] = ygasnew[7]
mass_fractions['CO_all_iso_HITEMP'] = ygasnew[0]
mass_fractions['H2S'] = ygasnew[9]
mass_fractions['Mg'] = ygasnew[5]
mass_fractions['SiO'] = ygasnew[6]
mass_fractions['Fe'] = ygasnew[8]
mass_fractions['CO2'] = ygasnew[4]
mass_fractions['CH4'] = ygasnew[3]
mass_fractions['TiO_all_Exomol'] = ygasnew[10]
mass_fractions['Al'] = ygasnew[11]

# set the mass fraction for cloud
xcdom = np.argmax(xc, axis=0)    # dominant species
xc[0, np.where(xcdom!=0)[0]] = 0
xc[1, np.where(xcdom!=1)[0]] = 0
xc[2, np.where(xcdom!=2)[0]] = 0
xc[3, np.where(xcdom!=3)[0]] = 0
mass_fractions['MgSiO3(c)'] = xc[0]
mass_fractions['Mg2SiO4(c)'] = xc[1]
mass_fractions['Fe(c)'] = xc[2]
mass_fractions['Al2O3(c)'] = xc[3]

# set the particle radius for the cloud
radius = {}
radius['MgSiO3(c)'] = ap
radius['Mg2SiO4(c)'] = ap
radius['Fe(c)'] = ap
radius['Al2O3(c)'] = ap
sigma_lnorm = 1.05

MMW = 2.34 * np.ones_like(Parr)

R_pl = nc.r_jup_mean
gravity = pars.g
P0 = 1e-4
R_star = nc.r_sun    # in Jupiter radius

plt.figure(figsize=(12, 6))
# calculate atmosphere without CO2
mass_fractions['CO2'] = np.zeros_like(Parr)

atmosphere.calc_transm(Tarr, mass_fractions, gravity, MMW, R_pl=R_pl, P0_bar=P0, radius=radius, sigma_lnorm=sigma_lnorm)

plt.plot(nc.c/atmosphere.freq/1e-4, (atmosphere.transm_rad/R_star)**2*100, label='no CO2', alpha=0.5)

# calculate atmosphere without CO
mass_fractions['CO2'] = ygasnew[4]
mass_fractions['CO_all_iso_HITEMP'] = np.zeros_like(Parr)

atmosphere.calc_transm(Tarr, mass_fractions, gravity, MMW, R_pl=R_pl, P0_bar=P0, radius=radius, sigma_lnorm=sigma_lnorm)

plt.plot(nc.c/atmosphere.freq/1e-4, (atmosphere.transm_rad/R_star)**2*100, label='no CO', alpha=0.5)

# calculate atmosphere without H2O
mass_fractions['CO_all_iso_HITEMP'] = ygasnew[0]
mass_fractions['H2O_HITEMP'] = np.zeros_like(Parr)

atmosphere.calc_transm(Tarr, mass_fractions, gravity, MMW, R_pl=R_pl, P0_bar=P0, radius=radius, sigma_lnorm=sigma_lnorm)

plt.plot(nc.c/atmosphere.freq/1e-4, (atmosphere.transm_rad/R_star)**2*100, label='no H2O', alpha=0.5)

# calculate atmosphere without CH4
mass_fractions['H2O_HITEMP'] = ygasnew[7]
mass_fractions['CH4'] = np.zeros_like(Parr)

atmosphere.calc_transm(Tarr, mass_fractions, gravity, MMW, R_pl=R_pl, P0_bar=P0, radius=radius, sigma_lnorm=sigma_lnorm)

plt.plot(nc.c/atmosphere.freq/1e-4, (atmosphere.transm_rad/R_star)**2*100, label='no CH4', alpha=0.5)

# calculate atmosphere without H2S
mass_fractions['CH4'] = ygasnew[3]
mass_fractions['H2S'] = np.zeros_like(Parr)

atmosphere.calc_transm(Tarr, mass_fractions, gravity, MMW, R_pl=R_pl, P0_bar=P0, radius=radius, sigma_lnorm=sigma_lnorm)

plt.plot(nc.c/atmosphere.freq/1e-4, (atmosphere.transm_rad/R_star)**2*100, label='no H2S', alpha=0.5)

# calculate clear atmosphere
mass_fractions['H2S'] = ygasnew[9]
mass_fractions['MgSiO3(c)'] = np.zeros_like(Parr)
mass_fractions['Mg2SiO4(c)'] = np.zeros_like(Parr)
mass_fractions['Fe(c)'] = np.zeros_like(Parr)
mass_fractions['Al2O3(c)'] = np.zeros_like(Parr)

atmosphere.calc_transm(Tarr, mass_fractions, gravity, MMW, R_pl=R_pl, P0_bar=P0, radius=radius, sigma_lnorm=sigma_lnorm)

plt.plot(nc.c/atmosphere.freq/1e-4, (atmosphere.transm_rad/R_star)**2*100, label='clear', linestyle='--', alpha=0.5)

# calculate cloudy atmosphere
mass_fractions['MgSiO3(c)'] = xc[0]
mass_fractions['Mg2SiO4(c)'] = xc[1]
mass_fractions['Fe(c)'] = xc[2]
mass_fractions['Al2O3(c)'] = xc[3]
atmosphere.calc_transm(Tarr, mass_fractions, gravity, MMW, R_pl=R_pl, P0_bar=P0, radius=radius, sigma_lnorm=sigma_lnorm)

plt.plot(nc.c/atmosphere.freq/1e-4, (atmosphere.transm_rad/R_star)**2*100, label='cloudy', color='k')

plt.legend()
plt.xscale('log')
plt.xlabel('Wavelength (microns)')
plt.ylabel(r'Transit depth (\%)')
ax = plt.gca()
ax.set_xlim([1, 20])
plt.savefig('spectrumold.png', dpi=288)
plt.show()

import os
os.remove('input.dat')
