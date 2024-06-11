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
import chemistry
import constants as cnt
import itertools
import pdb
sys.path.append('../../util')
from draw import myplot

def write_element(elementgrid):
    ''' Write the elemental abundance to input.dat '''
    with open('input.dat', 'w') as opt:
        opt.write('# Cloud model\n')
        for element, abundance in elementgrid.items():
            opt.write(element + ' ' + str(abundance) + '\n')

########### reconstruct the atmosphere object ###########
gridfile = 'gridCO09.txt'
kappafolder = 'coeff/'
# kappafolder = None
from read import reconstruct
atmosphere, pars = reconstruct(gridfile)

ncod = len(pars.solid)
ngas = len(pars.gas)

Parr = np.exp(atmosphere.grid)
Parrbar = Parr/1e6
cachegrid = atmosphere.cachegrid
Tarr = cachegrid.T_grid
y = atmosphere.y
rhop = atmosphere.rho

########### equilibrium chemistry calculations ##########
chem = atmosphere.chem
# other interesting species and their concentration
extragas = ['CO', 'H2', 'He', 'CH4', 'CO2']    # extra gas that may contribute to spectrum
# extragascon = np.array([1.3e-3, 1, 0.33, 0, 0])    # for CtoO /= 3
# extragascon = np.array([4.1e-3, 0.75, 0.25, 0, 0])    # for CtoO = 0.8
extragascon = np.array([4.4e-3, 0.75, 0.25, 0, 0])    # for CtoO = 0.9
# extragascon = np.array([3.2e-2, 1, 0.33, 0, 0])    # for metallicity *= 10
# extragascon = np.array([3.2e-3, 1, 0.33, 0, 0])
# all of the gas molecules to be considered in the spectrum calculation
gasmols = []
for gasname in extragas + pars.gas:
    gasmols.append(chemistry.molecule(gasname))
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
        num = y[i+ncod, j] / gasmol.mu
        for element in gasmol.element.keys():
            elementgrid[element] += num * gasmol.element[element]

    # normalize the elemental abundance
    totalnumber = np.sum(list(elementgrid.values()))
    # transfer to the log version
    for element, abundance in elementgrid.items():
        elementgrid[element] = np.log10(abundance/totalnumber)+12

    write_element(elementgrid)
    fastchem = pyfastchem.FastChem('./input.dat',
                                '/home/helong/software/FastChem/fastchem/input/logK/logK.dat',
                                1)

    # Read input file
    input_data.temperature = np.array([cachegrid.T_grid[j]])
    input_data.pressure = np.array([Parrbar[j]])

    # Calculate the chemical equilibrium
    fastchem_flag = fastchem.calcDensities(input_data, output_data)

    for i in range(len(gasmols)):
        ygasnew[i, j] = gasmols[i].mu * cnt.mu * output_data.number_densities[0][gasindex[i]] / cachegrid.rho_grid[j]

# plot the atmosphere after equilibrium chemistry calculation
ynew = np.vstack((y[:ncod], ygasnew[len(extragas):], y[-1]))
myplot(Parr, ynew, rhop, ncod, ngas, plotmode='popup')
# pdb.set_trace()
# sys.exit()

########### calculate or read oopacity #########
ap = atmosphere.ap
n_p = atmosphere.np
rhog = cachegrid.rho_grid
if kappafolder == None:
    from calmeff import cal_eff_m_all, writelnk
    from calkappa import cal_opa_all
    # calculate effective refractory index
    bs = atmosphere.bs
    wlenkappa = np.logspace(np.log10(0.5), np.log10(20), 100)
    mmat = cal_eff_m_all(bs, pars.solid, wlenkappa)
    writelnk(mmat, wlenkappa, rhop, folder='meff')

    # calculate kappa
    opobj = cal_opa_all(ap, write=False, Nlam=len(wlenkappa), optooldir='/home/helong/software/optool')
    kappaext = opobj['kext']
else:
    data = np.genfromtxt(kappafolder + '0.txt')
    wlenkappa = data[:, 0]
    kappaext = np.empty((len(wlenkappa), len(Parr)))
    for i in range(len(Parr)):
        kappafilename = kappafolder + f'{i}.txt'
        data = np.genfromtxt(kappafilename)
        data = data.T
        kappaext[:, i] = data[3]
kappadata = kappaext/rhog*n_p*4*np.pi/3*ap**3
# calculate spline object
spobj = RectBivariateSpline(wlenkappa, Parrbar, kappadata)
# sys.exit()

####### Radiation transfer part #######
wlenrange = [0.5, 20]
# cloud opacity function
def cloud_opas(spobj):
    ''' return cloud opacity (cm^2/g)
        the parameter is a spline object from scipy.interpolate.RectBivariateSpline
    '''
    def give_opacity(wlen, press):
        ''' wavelength in micron and pressure in bar '''
        pressmat, wlenmat = np.meshgrid(press, wlen)
        retVal = spobj.ev(wlenmat, pressmat)
        return retVal

    return give_opacity

linespecies = ['H2O_HITEMP', 'CO_all_iso_HITEMP', 'H2S', 'Mg', 'SiO', 'Fe', 'CO2', 'CH4', 'TiO_all_Exomol', 'Al']
# linespecies = ['CO_all_iso_HITEMP', 'CH4', 'CO2', 'Na_allard', 'K_allard', 'H2S']
atmosphere = Radtrans(line_species=linespecies, rayleigh_species=['H2', 'He'], continuum_opacities = ['H2-H2', 'H2-He'], wlen_bords_micron = [wlenrange[0], wlenrange[-1]])
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

MMW = pars.mgas / cnt.mu * np.ones_like(Parr)

R_pl = pars.Rp
gravity = pars.g
P0 = 1e-4
R_star = pars.R_star    # in Jupiter radius

transitdepth = nc.c/atmosphere.freq/1e-4
labels = ['wavelength']

plt.figure(figsize=(12, 4))
# calculate atmosphere without CO2
mass_fractions['CO2'] = np.zeros_like(Parr)

atmosphere.calc_transm(Tarr, mass_fractions, gravity, MMW, R_pl=R_pl, P0_bar=P0, give_absorption_opacity=cloud_opas(spobj))

l, = plt.plot(nc.c/atmosphere.freq/1e-4, (atmosphere.transm_rad/R_star)**2*100, label='no CO2', alpha=0.5, linewidth=1.)
transitdepth = np.vstack((transitdepth, (atmosphere.transm_rad/R_star)**2*100))
labels.append(l.get_label().replace(' ', '_'))

# calculate atmosphere without CO
mass_fractions['CO2'] = ygasnew[4]
mass_fractions['CO_all_iso_HITEMP'] = np.zeros_like(Parr)

atmosphere.calc_transm(Tarr, mass_fractions, gravity, MMW, R_pl=R_pl, P0_bar=P0, give_absorption_opacity=cloud_opas(spobj))

l, = plt.plot(nc.c/atmosphere.freq/1e-4, (atmosphere.transm_rad/R_star)**2*100, label='no CO', alpha=0.5, linewidth=1.)
transitdepth = np.vstack((transitdepth, (atmosphere.transm_rad/R_star)**2*100))
labels.append(l.get_label().replace(' ', '_'))

# calculate atmosphere without H2O
mass_fractions['CO_all_iso_HITEMP'] = ygasnew[0]
mass_fractions['H2O_HITEMP'] = np.zeros_like(Parr)

atmosphere.calc_transm(Tarr, mass_fractions, gravity, MMW, R_pl=R_pl, P0_bar=P0, give_absorption_opacity=cloud_opas(spobj))

l, = plt.plot(nc.c/atmosphere.freq/1e-4, (atmosphere.transm_rad/R_star)**2*100, label='no H2O', alpha=0.5, linewidth=1.)
transitdepth = np.vstack((transitdepth, (atmosphere.transm_rad/R_star)**2*100))
labels.append(l.get_label().replace(' ', '_'))

# calculate atmosphere without CH4
mass_fractions['H2O_HITEMP'] = ygasnew[7]
mass_fractions['CH4'] = np.zeros_like(Parr)

atmosphere.calc_transm(Tarr, mass_fractions, gravity, MMW, R_pl=R_pl, P0_bar=P0, give_absorption_opacity=cloud_opas(spobj))

l, = plt.plot(nc.c/atmosphere.freq/1e-4, (atmosphere.transm_rad/R_star)**2*100, label='no CH4', alpha=0.5, linewidth=1.)
transitdepth = np.vstack((transitdepth, (atmosphere.transm_rad/R_star)**2*100))
labels.append(l.get_label().replace(' ', '_'))

# calculate atmosphere without H2S
mass_fractions['CH4'] = ygasnew[3]
mass_fractions['H2S'] = np.zeros_like(Parr)

atmosphere.calc_transm(Tarr, mass_fractions, gravity, MMW, R_pl=R_pl, P0_bar=P0, give_absorption_opacity=cloud_opas(spobj))

l, = plt.plot(nc.c/atmosphere.freq/1e-4, (atmosphere.transm_rad/R_star)**2*100, label='no H2S', alpha=0.5, linewidth=1.)
transitdepth = np.vstack((transitdepth, (atmosphere.transm_rad/R_star)**2*100))
labels.append(l.get_label().replace(' ', '_'))

# calculate clear atmosphere
mass_fractions['H2S'] = ygasnew[9]

atmosphere.calc_transm(Tarr, mass_fractions, gravity, MMW, R_pl=R_pl, P0_bar=P0)

l, = plt.plot(nc.c/atmosphere.freq/1e-4, (atmosphere.transm_rad/R_star)**2*100, label='clear', color='grey', alpha=0.5, linewidth=1.)
transitdepth = np.vstack((transitdepth, (atmosphere.transm_rad/R_star)**2*100))
labels.append(l.get_label())

# calculate cloudy atmosphere
atmosphere.calc_transm(Tarr, mass_fractions, gravity, MMW, R_pl=R_pl, P0_bar=P0, Pcloud=100, give_absorption_opacity=cloud_opas(spobj))

l, = plt.plot(nc.c/atmosphere.freq/1e-4, (atmosphere.transm_rad/R_star)**2*100, label='cloudy', color='k', linewidth=1.5)
transitdepth = np.vstack((transitdepth, (atmosphere.transm_rad/R_star)**2*100))
labels.append(l.get_label())

plt.legend()
plt.xscale('log')
plt.xlabel('Wavelength (microns)')
plt.ylabel(r'Transit depth (\%)')
ax = plt.gca()
ax.set_xlim([0.5, 20])
#get x and y limits
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
#set aspect ratio
# ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*0.02)
plt.subplots_adjust(left=0.07, right=0.98, bottom=0.11, top=0.98, hspace=0, wspace=0)
# plt.savefig('spectrum' + pars.runname + '.pdf', dpi=288)
plt.show()

with open('spectrum' + pars.runname + '.txt', 'w') as opt:
    for label in labels:
        opt.write(label + ' ')
    opt.write('\n')
    for i in range(len(transitdepth[0])):
        for j in range(len(transitdepth)):
            opt.write(f'{transitdepth[j, i]} ')
        opt.write('\n')

import os
os.remove('input.dat')
