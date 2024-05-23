import numpy as np
import pdb
import os
import parameters as pars

def preparepoly(marr, i):
    '''
    prepare polynomial for a single species: (m_e^2-m_i^2)*\Multiply_{j!=i}(m_e^2+m_j^2/2)
    marr: complex array of refractory indices
    '''
    N = len(marr)

    rootarr = np.empty(N, dtype=complex)
    rootarr = -marr**2/2
    rootarr[i] = marr[i]**2
    polyidxy = np.poly(rootarr)
    polyidx = np.zeros(2*N+1, dtype=complex)
    polyidx[::2] = polyidxy

    return polyidx

def cal_eff_m (abundance, solid, wavelength, mspecies, polyidxmat):
    """
    CWO: Please say briefly what this function does
    """
    sortidx = np.argsort(-abundance)    # rank from large to small

    marr = np.empty(len(wavelength), dtype=complex)

    for k in range(len(wavelength)):
        # initialize
        mold = mspecies[k][sortidx[0]]
        # print(mold)
        # try to include everything at once. If that succeed, then no need to iterate

        ## CWO: hard to follow what you're doing here... 
        #       preparepoly is slow...
        polyidx = np.matmul(polyidxmat[k], abundance)
        allroots = np.roots(polyidx)
        physicalidx = np.where((allroots.real>0)&(allroots.imag>0))[0]    # physical solution for the refractory index.
        if len(physicalidx)==1:
            marr[k] = allroots[physicalidx]
            continue

        # find the polynomial index of the equation, with a new species
        for n in range(2, len(solid)+1):
            print('Warning!')
            # polynomial index only inluding previous species
            polyidxold = np.zeros(2*n+1, dtype=complex)
            for i in range(n-1):
                polyidxold += preparepoly(mspecies[k][sortidx[:n]], i) * abundance[sortidx[i]]
            # polynomial index of newly-added species
            polyidxnew = preparepoly(mspecies[k][sortidx[:n]], n-1) * abundance[sortidx[n-1]]

            # relax towards the solution, to ensure new solution is close to the old one
            fsucc = 0.
            ffail = np.array([1.])
            while(fsucc<1.):
                frudge = ffail[-1]
                polyidx = polyidxold + polyidxnew * frudge
                allroots = np.roots(polyidx)
                physicalidx = np.where((allroots.real>0)&(allroots.imag>0))[0]    # physical solution for the refractory index.
                if len(physicalidx)==1:
                    ffail = np.delete(ffail, -1)
                    fsucc = frudge
                    mold = allroots[physicalidx[0]]
                else:
                    change = np.abs(allroots-mold)
                    nearidx = np.argmin(change)

                    if np.all(change[nearidx]*10<np.delete(change, nearidx)):
                        ffail = np.delete(ffail, -1)
                        fsucc = frudge
                        mold = allroots[nearidx]
                    else:
                        if fsucc==0:
                            ffail = np.append(ffail, ffail[-1]/10)
                        else:
                            ffail = np.append(ffail, np.sqrt(fsucc*frudge))

            marr[k] = mold

    return marr

def cal_eff_m_all (abundance, solid, wavelengthgrid):
    """
    calculate effective medium optical constants:
    - abundance:        :2d array of solid mass fractions (species-id, particle)     
    - wavelengthgrid    :1d array of wavelength (micron)
    - solid             :1d array giving the name for the solid species

    in our case each particle corresponds to a single grid point
    """

    Nwlen = len(wavelengthgrid)
    Nsolid = len(solid)

    ## CWO: maybe we can do this once
    mdataL = []
    for i, solidname in enumerate(solid):
        filename = datadict[solidname]
        mdata = np.genfromtxt(folder+filename)
        mdataL.append(mdata)

    # n+ik for each species
    mspecies = np.empty((Nwlen, Nsolid), dtype=complex)
    # read the n-k data and interpolate
    for i, solidname in enumerate(solid):
        #filename = datadict[solidname]
        #mdata = np.genfromtxt(folder+filename)
        mdata = mdataL[i]

        ## CWO: what if the interpolation be out-of-bounds?
        n = np.interp(wavelengthgrid, mdata[:, 0], mdata[:, 1])
        k = np.interp(wavelengthgrid, mdata[:, 0], mdata[:, 2])
        mspecies[:, i].real = n
        mspecies[:, i].imag = k

    ##### prepare the polyindex ahead of time #####
    polyidxmat = np.empty([Nwlen, 2*Nsolid+1, Nsolid], dtype=complex)
    for i in range(Nwlen):
        for j in range(Nsolid):
            polyidxmat[i, :, j] = preparepoly(mspecies[i], j)

    mmat = np.empty((abundance.shape[1], Nwlen), dtype=complex)
    for i in range(abundance.shape[1]):
        print(f'\r[calmeff.cal_eff_m_all]:performing effective medium on particle {i}/{abundance.shape[1]}', end="")
        marr = cal_eff_m(abundance[:, i], solid, wavelengthgrid, mspecies, polyidxmat)
        mmat[i] = marr
    print()
    return mmat

def writelnk(mmat, wavelength, rho, folder='util/meff'):

    # create folder is it doesn't exist
    if not os.path.exists(folder):
        os.mkdir(folder)

    Nwlen = len(wavelength)
    for i in range(mmat.shape[0]):
        filename = folder + f'/{i}.lnk'
        with open(filename, 'w') as opt:
            opt.write(f'{Nwlen} {rho[i]}\n')
            for j in range(Nwlen):
                opt.write(f'{wavelength[j]} {mmat[i,j].real} {mmat[i,j].imag}\n')

# files containing the n-k data of species
folder = '/home/helong/ARCiS/code/tables/nk/'
folder = os.path.join(pars.rootdir,'tables/nk/')
datadict = {'MgSiO3':'MgSiO3_amorph_sol-gel.dat', 'Mg2SiO4':'Mg2SiO4_amorph_sol-gel.dat',
            'SiO2':'SiO2_amorph.dat', 'MgO':'MgO.dat', 'FeO':'FeO.dat', 'FeS':'FeS.dat',
            'Fe2O3':'Fe2O3.dat', 'Fe':'Fe.dat', 'TiO2':'TiO2_anatase.dat', 'Al2O3':'Al2O3.dat',
            'KCl':'KCl.dat', 'NaCl':'NaCl.dat', 'ZnS':'ZnS.dat', 'Na2S':'Na2S.dat'}
files = datadict.values()

# wavelength (micron) of interest
# wavelength = np.logspace(np.log10(1), np.log10(20), 100)

# n+ik for each species
# mspecies = np.empty((len(wavelength), len(datadict)), dtype=complex)
# read the n-k data and interpolate
# for i, solidname in enumerate(datadict):
#     filename = datadict[solidname]
#     mdata = np.genfromtxt(folder+filename)
#     n = np.interp(wavelength, mdata[:, 0], mdata[:, 1])
#     k = np.interp(wavelength, mdata[:, 0], mdata[:, 2])
#     mspecies[:, i].real = n
#     mspecies[:, i].imag = k

# frac = np.array([0.2,0.1,0.05,0.24,0.01,0.01,0.01,0.2,0.09,0.09])    # fraction of each species
# frac = np.array([0.12,0.01,0.01,0.01,0.01,0.01,0.01,0.8,0.01,0.01])
# sortidx = np.argsort(-frac)    # rank from large to small

# marr = np.empty(len(wavelength), dtype=complex)

# for k in range(len(wavelength)):
#     # initialize
#     mold = mspecies[k][sortidx[0]]
#     # print(mold)

#     # find the polynomial index of the equation, with a new species
#     for n in range(2, len(datadict)+1):
#         # polynomial index only inluding previous species
#         polyidxold = np.zeros(2*n+1, dtype=complex)
#         for i in range(n-1):
#             polyidxold += preparepoly(mspecies[k][sortidx[:n]], i) * frac[sortidx[i]]
#         # polynomial index of newly-added species
#         polyidxnew = preparepoly(mspecies[k][sortidx[:n]], n-1) * frac[sortidx[n-1]]

#         # relax towards the solution, to ensure new solution is close to the old one
#         fsucc = 0.
#         ffail = np.array([1.])
#         while(fsucc<1.):
#             frudge = ffail[-1]
#             polyidx = polyidxold + polyidxnew * frudge
#             allroots = np.roots(polyidx)
#             change = np.abs(allroots-mold)
#             nearidx = np.argmin(change)

#             if np.all(change[nearidx]*10<np.delete(change, nearidx)):
#                 ffail = np.delete(ffail, -1)
#                 fsucc = frudge
#                 mold = allroots[nearidx]
#             else:
#                 if fsucc==0:
#                     ffail = np.append(ffail, ffail[-1]/10)
#                 else:
#                     ffail = np.append(ffail, np.sqrt(fsucc*frudge))

#         marr[k] = mold
        # print(mold)

# marr = cal_eff_m(frac, list(datadict.keys()), wavelength)

# plot the relation between pure material and mixed material --scatter data
# from matplotlib import pyplot as plt
# for i in range(len(datadict)):
#     plt.plot(mspecies[0, i].real, mspecies[0, i].imag, 'o', markersize=frac[i]*30, label=list(datadict.keys())[i])
# plt.plot(mold.real, mold.imag, 'x', color='r', markersize=10)
# plt.legend()
# plt.show()

# plot read and image part of the effective refractory index
# fig, (ax1, ax2) = plt.subplots(1, 2)
# for i in range(len(datadict)):
#     l, = ax1.loglog(wavelength, mspecies[:, i].real, label=list(datadict.keys())[i], alpha=frac[i]*2)
#     ax2.loglog(wavelength, mspecies[:, i].imag, alpha=frac[i]*2, color=l.get_color())
# ax1.loglog(wavelength, marr.real, color='k')
# ax1.legend()
# ax2.loglog(wavelength, marr.imag, color='k')
# ax1.set_xlabel('wavelength (micron)')
# ax1.set_ylabel('n')
# ax2.set_xlabel('wavelength (micron)')
# ax2.set_ylabel('k')
# plt.show()
