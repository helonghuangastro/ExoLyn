'''
This script is user-defined to calculate equilibrium gas phase chemistry and then spectrum.
FastChem and petitRADTRANS is required in this script. Please download and install them before running:
FastChem: https://exoclime.github.io/FastChem/
petitRADTRANS: https://petitradtrans.readthedocs.io/en/latest/
'''
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import RectBivariateSpline
from petitRADTRANS.radtrans import Radtrans
from petitRADTRANS import physical_constants as cst
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
gridfile = 'grid.txt'
# kappafolder = 'coeff/'
kappafolder = None
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
extragas = ['CO', 'H2', 'He', 'CH4', 'CO2', 'K', 'Na']    # extra gas that may contribute to spectrum
# extragascon = np.array([1.3e-3, 1, 0.33, 0, 0])    # for CtoO /= 3
# extragascon = np.array([4.4e-3, 0.75, 0.25, 0, 0, 1.7e-6, 1.7e-5])    # for CtoO = 0.9
# extragascon = np.array([3.2e-2, 1, 0.33, 0, 0, 1.7e-5, 1.7e-4])    # for metallicity *= 10
extragascon = np.array([3.2e-3, 1, 0.33, 0, 0, 1.7e-6, 1.7e-5])
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
myplot(Parr, ynew, rhop, ncod, ngas, plotmode='none')
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
    kappascat = opobj['ksca']
    gsca = opobj['gsca']
else:
    data = np.genfromtxt(kappafolder + '0.txt')
    wlenkappa = data[:, 0]
    kappaext = np.empty((len(wlenkappa), len(Parr)))
    kappascat = np.empty((len(wlenkappa), len(Parr)))
    gsca = np.empty((len(wlenkappa), len(Parr)))
    for i in range(len(Parr)):
        kappafilename = kappafolder + f'{i}.txt'
        data = np.genfromtxt(kappafilename)
        data = data.T
        kappaext[:, i] = data[3]
        kappascat[:, i] = data[2]
        gsca[:, i] = data[4]
kappadata = kappaext/rhog*n_p*4*np.pi/3*ap**3
kappascat = kappascat/rhog*n_p*4*np.pi/3*ap**3
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

gasnames = [gasmol.name for gasmol in gasmols]
line_species = [gasname for gasname in gasnames if gasname!='H2' and gasname!='He']
atmosphere = Radtrans(pressures=Parrbar, line_species=line_species, rayleigh_species=['H2', 'He'], 
                        gas_continuum_contributors=['H2-H2', 'H2-He'], wavelength_boundaries=[wlenrange[0], wlenrange[-1]])

mass_fractions = {}
for i, gasname in enumerate(gasnames):
    mass_fractions[gasname] = ygasnew[i]

MMW = pars.mgas / cnt.mu * np.ones_like(Parr)

R_pl = pars.Rp
gravity = pars.g
P0 = 1e-4
R_star = pars.R_star    # in Jupiter radius

def remove1test(atmosphere, mass_fractions, removelist, Tarr, gravity, MMW, R_pl, P0, cloud_opas=None, *cloudargs):
    '''
    perform remove one test
    input:
        atmosphere: RADTRANS object
        removelist: the list of things to be removed
        mass_fractions, Tarr, gravity, MMW, R_pl, P0: to be transfered to calculate_transit_radii() method
    '''
    # default atmosphere
    wavelengths, transit_radii, _ = atmosphere.calculate_transit_radii(temperatures=Tarr, mass_fractions=mass_fractions, 
                                                                    reference_gravity=gravity, mean_molar_masses=MMW, 
                                                                    planet_radius=R_pl, reference_pressure=P0, 
                                                                    additional_absorption_opacities_function=cloud_opas(*cloudargs))
    transit_radii_remove = transit_radii.copy()

    # remove one component at a time
    for toremove in removelist:
        print('testing removing ' + toremove)
        # clear
        if toremove == 'cloud':
            wavelengths, transit_radii, _ = atmosphere.calculate_transit_radii(temperatures=Tarr, mass_fractions=mass_fractions, 
                                                                            reference_gravity=gravity, mean_molar_masses=MMW, 
                                                                            planet_radius=R_pl, reference_pressure=P0)
        
        # remove one gas species
        else:
            mass_fractions_new = mass_fractions.copy()
            mass_fractions_new[toremove] = 0    # set the corresponding gas composition to 0
            wavelengths, transit_radii, _ = atmosphere.calculate_transit_radii(temperatures=Tarr, mass_fractions=mass_fractions_new, 
                                                                            reference_gravity=gravity, mean_molar_masses=MMW, 
                                                                            planet_radius=R_pl, reference_pressure=P0,
                                                                            additional_absorption_opacities_function=cloud_opas(*cloudargs))

        transit_radii_remove = np.vstack((transit_radii_remove, transit_radii))

    return transit_radii_remove

def plot_remove1test(transit_depth_remove, removelist, wlen):
    ''' Plot the result of the remove 1 test '''
    plt.figure(figsize=(12, 4))

    for i, toremove in enumerate(removelist):
        plt.plot(wlen, transit_depth_remove[i+1]*100, alpha=0.5, linewidth=1., label='no'+toremove)    # "100" accounts for the unit "%"

    plt.plot(wlen, transit_depth_remove[0]*100, label='cloudy', color='k', linewidth=1.5)

    # set the plot
    plt.legend()
    plt.xscale('log')
    plt.xlabel('Wavelength (microns)')
    plt.ylabel(r'Transit depth (\%)')
    ax = plt.gca()
    ax.set_xlim([0.5, 20])
    #get x and y limits
    # x_left, x_right = ax.get_xlim()
    # y_low, y_high = ax.get_ylim()
    #set aspect ratio
    # ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*0.02)
    plt.subplots_adjust(left=0.07, right=0.98, bottom=0.11, top=0.98, hspace=0, wspace=0)
    # plt.savefig('spectrum' + pars.runname + '.pdf', dpi=288)
    plt.show()

def save_remove1test(filename, transit_depth_remove, removelist, wlen):
    ''' save the result of remove 1 test to a file '''
    with open(filename, 'w') as opt:
        opt.write('wavelength cloudy ')
        for toremove in removelist:
            opt.write('no_' + toremove + ' ')
        opt.write('\n')
        for i in range(len(wlen)):
            opt.write(f'{wlen[i]} ')
            for j in range(len(transit_depth_remove)):
                opt.write(f'{transit_depth_remove[j, i]} ')
            opt.write('\n')

wavelengths = cst.c/atmosphere.frequencies/1e-4

def get_TP_input(flaggauss=False):
    # opacities, continuum_opacities_scattering, _, _, _, _ = atmosphere._calculate_opacities(temperatures=Tarr, mass_fractions=mass_fractions, 
    #                                                                     mean_molar_masses=MMW, reference_gravity=gravity, 
    #                                                                     opaque_cloud_top_pressure=None,
    #                                                                     cloud_particles_mean_radii=None, cloud_particle_radius_distribution_std=None,
    #                                                                     cloud_particles_radius_distribution="lognormal", cloud_hansen_a=None, cloud_hansen_b=None,
    #                                                                     clouds_particles_porosity_factor=None,
    #                                                                     cloud_f_sed=None, eddy_diffusion_coefficients=None,
    #                                                                     haze_factor=1.0, power_law_opacity_350nm=None, power_law_opacity_coefficient=None,
    #                                                                     gray_opacity=None, cloud_photosphere_median_optical_depth=None,
    #                                                                     return_cloud_contribution=False,
    #                                                                     additional_absorption_opacities_function=cloud_opas(spobj),
    #                                                                     additional_scattering_opacities_function=None)
    opacities, continuum_opacities_scattering, _, _, _, _ = atmosphere._calculate_opacities(temperatures=Tarr, mass_fractions=mass_fractions, 
                                                                        mean_molar_masses=MMW, reference_gravity=gravity, 
                                                                        opaque_cloud_top_pressure=None,
                                                                        cloud_particles_mean_radii=None, cloud_particle_radius_distribution_std=None,
                                                                        cloud_particles_radius_distribution="lognormal", cloud_hansen_a=None, cloud_hansen_b=None,
                                                                        clouds_particles_porosity_factor=None,
                                                                        cloud_f_sed=None, eddy_diffusion_coefficients=None,
                                                                        haze_factor=1.0, power_law_opacity_350nm=None, power_law_opacity_coefficient=None,
                                                                        gray_opacity=None, cloud_photosphere_median_optical_depth=None,
                                                                        return_cloud_contribution=False,
                                                                        additional_absorption_opacities_function=None,
                                                                        additional_scattering_opacities_function=None)

    opacities = np.sum(opacities, axis=2)
    if not flaggauss:
        weights_gauss = np.load('../../../TPiter/shared_data/weightsgauss.npy')
        opacities = np.einsum('ijk,i', opacities, weights_gauss, optimize=True)
    # np.savez('../../../TPiter/cloudy/gasopacities.npz', opacities=opacities, wavelength=wavelengths, pressures=Parr)
    if flaggauss:
        np.savez('../../../TPiter/cloudy/gasopacitiesg.npz', opacities=opacities, rayleigh=continuum_opacities_scattering, wavelength=wavelengths, pressures=Parrbar)
    else:
        np.savez('../../../TPiter/cloudy/gasopacities.npz', opacities=opacities, rayleigh=continuum_opacities_scattering, wavelength=wavelengths, pressures=Parrbar)
    # save cloud opacities
    np.savez('../../../TPiter/cloudy/cloudopacities.npz', opaext=kappadata, opasca=kappascat, gsca=gsca, wavelength=wlenkappa, pressures=Parrbar)

get_TP_input(flaggauss=True)

# removelist = ['cloud', 'CO2', 'CO', 'H2O', 'H2S', 'CH4']
# transit_radii_remove = remove1test(atmosphere, mass_fractions, removelist, Tarr, gravity, MMW, R_pl, P0, cloud_opas, spobj)
# transit_depth_remove = (transit_radii_remove/R_star)**2

# plot_remove1test(transit_depth_remove, removelist, wavelengths)

# save_remove1test('spectrum' + pars.runname + '.txt', transit_depth_remove, removelist, wavelengths)

import os
os.remove('input.dat')
