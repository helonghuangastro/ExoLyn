import numpy as np
from matplotlib import pyplot as plt
import os
import sys
sys.path.append('../')

import atmosphere_class
# import parameters as pars
import constants as cnt
# import chemistry

##TBA: /home/helong is not the same on my computer!

def cal_opa (filename, ap, Nlam, optooldir, dirkappa='./coeff', write=False, savefile=None):
    ''' 
    ap is the particle size in cm 
    '''
    sys.path.append(optooldir)
    import optool

    #command = optooldir+'/optool ' + filename  + f' -p 0.25 -a {ap*1e4} -l 1 20 100 -o {dirkappa}'
    #command = optooldir+'/optool ' + filename  + f' -p 0.25 -a {ap*1e4} -l {filename} -o {dirkappa}'
    #p = optool.particle(optooldir+'/optool ' + filename  + f' -p 0.25 -a {ap*1e4} -l 1 20 100 -o coeff')

    ## CWO: I have changed the wavelength grid consistent with the .lnk file, which makes sense the most
    ##      It is also possible to provide the particle file, e.g. "-a particles" it seems...
    ##      Also, why do we have a porosity? "-p 0.25"?
    ##      Can you check w/r the units are correct? (cm**2 /g?) or is it a cross section?!
    ##      The standard output is very annoying -- any way to suppress it?
    command = optooldir+f'/optool {filename} -q -p 0.25 -a {ap*1e4} -l {filename} -o {dirkappa}'
    p = optool.particle(command, silent=True)

    if write:
        with open('coeff/'+savefile, 'w') as opt:
            opt.write('#optical properties for particle ...\n')
            opt.write('#cols::[wavelength,kappa_abs,kappa_sca,kappa_ext,asymmetry_parameter]\n')
            opt.write('#colunits::[micron,cm2/g,cm2/g,cm2/g,]\n')
            for j in range(Nlam):
                sfmt = 5*'{:10.3e} '
                line = sfmt.format(p.lam[j], p.kabs[0,j], p.ksca[0,j], p.kext[0,j], p.gsca[0,j])
                opt.write(line+'\n')
                #opt.write(f'{p.lam[j]} {p.kabs[0,j]} {p.ksca[0,j]} {p.kext[0,j]} {p.gsca[0,j]}\n')
    return p


def cal_opa_all(aparr, write=False, Nlam=100, wavelengthgrid=None, 
                optooldir='./', dirmeff='./meff', dirkappa='./coeff', **kwargs):
    ''' calculate the opacity for all the wavelengths at all pressure '''
    N = len(aparr)

    #wavelength range in micron
    wmin = 1.0 #micron -- should become a parameter
    wmax = 20
    #wlen = np.logspace(0, np.log10(20), Nlam) #This np.logspace is confusing...

    if wavelengthgrid is None:
        wlen = 10**np.linspace(np.log10(wmin), np.log10(wmax), Nlam)
    else:
        wlen = wavelengthgrid
        Nlam = len(wlen)

    kabsmat = np.empty((Nlam, N))
    kscamat = np.empty((Nlam, N))
    kextmat = np.empty((Nlam, N))
    gscamat = np.empty((Nlam, N))

    if write:
        if not os.path.exists(dirkappa):
            os.mkdir(dirkappa)

    for i in range(N):
        print(f'\r[calmeff.cal_opa_all]:running optool on particle {i}/{N}', end="")
        p = cal_opa(dirmeff+f'/{i}.lnk', aparr[i], Nlam, optooldir, dirkappa, write=write, savefile=f'{i}.txt')
        kabsmat[:, i] = p.kabs[0]
        kscamat[:, i] = p.ksca[0]
        kextmat[:, i] = p.kext[0]
        gscamat[:, i] = p.gsca[0]
    print()

    ## CWO: maybe only return when write is False?  
    if write==False:
        kappadata = {'wlen':wlen, 'kabs':kabsmat, 'ksca':kscamat, 'kext':kextmat, 'gsca':gscamat}
        return kappadata


# if __name__ == '__main__':

#     optooldir = sys.path.append('/home/helong/software/optool')

#     # read the data and calculate the particle size
#     ngas = len(pars.gas)
#     data = np.genfromtxt('../grid.txt', skip_header=1)
#     data = data.T
#     Parr = np.exp(data[0])
#     Parrbar = Parr/1e6
#     logP = np.log(Parr)

#     chem = chemistry.chemdata('../' + pars.gibbsfile)
#     cache = funs.init_cache(Parr, chem)
#     atmosphere = atmosphere_class.atmosphere(logP, pars.solid, pars.gas, cache)

#     atmosphere.update(data[1:])

#     for i in range(atmosphere.N):
#         #CWO -- I'm confused by this
#         p = optool.particle(f'~/software/optool/optool meff/{i}.lnk -p 0.25 -a {atmosphere.ap[i]*1e4} -l 1 20 100 -o coeff')
#         with open(f'coeff/{i}.txt', 'w') as opt:
#             opt.write('wavelength(micron) kappa_abs kappa_sca kappa_ext asymmetry_parameter\n')
#             for j in range(100):
#                 opt.write(f'{p.lam[j]} {p.kabs[0,j]} {p.ksca[0,j]} {p.kext[0,j]} {p.gsca[0,j]}\n')
