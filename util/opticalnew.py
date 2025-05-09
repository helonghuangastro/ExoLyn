import os, sys
import numpy as np
# from units import cgs2015 as cgs
# import scipy.integrate as sciint
# import scipy.optimize as sciop
import parameters as pars
# import subprocess as sp
from multiprocessing import Pool
from functools import partial

## CWO put it at the module level...
# sys.path.append(pars.doptical['optooldir'])
sys.path.append('/home/helong/software/optool')
import optool

folder = os.path.join(pars.rootdir,'tables/nk/')
datadict = {'MgSiO3':'MgSiO3_amorph_sol-gel.dat', 'Mg2SiO4':'Mg2SiO4_amorph_sol-gel.dat',
            'SiO2':'SiO2_amorph.dat', 'MgO':'MgO.dat', 'FeO':'FeO.dat', 'FeS':'FeS.dat',
            'Fe2O3':'Fe2O3.dat', 'Fe':'Fe.dat', 'TiO2':'TiO2_anatase.dat', 'Al2O3':'Al2O3.dat',
            'KCl':'KCl.dat', 'NaCl':'NaCl.dat', 'ZnS':'ZnS.dat', 'Na2S':'Na2S.dat'}
files = datadict.values()

def prepare_optical (optooldir=None, wavelengthgrid=None, dirmeff=None, dirkappa=None, multiproc=False, multi_nproc=1):
    """
    does some checks to see w/r we can perform optical constant calculations...
    """
    calcoptical = True

    #check w/r optool is there...
    if 'optooldir' is None:
        print('[optical.py]ERROR:No >> optooldir << provided')
        calcoptical = False
    else:
        try:
            flist = os.listdir(optooldir)
        except:
            print(f'[optical.py]ERROR:No valid dir for >> optooldir << ({optooldir}) provided')
            calcoptical = False

    #search for optool and optool.py
    if calcoptical:
        if 'optool' not in flist or 'optool.py' not in flist:
            print(f'[optical.py]ERROR:"optool" and/or "optool.py" not in {optooldir}')
            calcoptical = False

    #output entries for effective medium
    if calcoptical:
        if dirmeff == None:
            dirmeff = './meff'
        if not os.path.exists(dirmeff):
            try:
                os.mkdir(dirmeff)
            except:
                print(f'[optical.py]ERROR:creating dir >> dirmeff << ({dirmeff})')
                calcoptical = False

    if calcoptical:
        if dirkappa != None and not os.path.exists(dirkappa):
            try:
                os.mkdir(dirkappa)
            except:
                print(f'[optical.py]ERROR:creating dir >> dirkappa << ({dirkappa})')
                calcoptical = False

    if wavelengthgrid==None:
        #default grid 0.5--10 micron
        wavelengthgrid = 10**np.linspace(np.log10(0.5), np.log10(20.0), 100)

    elif type(wavelengthgrid)==str:
        #TBD read the wavelength grid from file 
        wavelengthgrid = np.load(wavelengthgrid)

    #perhaps not the most elegant solution
    doptical = {'optooldir':optooldir, 'wavelengthgrid':wavelengthgrid, 
                'dirmeff':dirmeff, 'dirkappa':dirkappa,
                'multiproc': multiproc, 'multi_nproc': multi_nproc}

    #TBD: we need input from parameters for multi-processing
    # doptical['scipy_root'] = False
    # doptical['rosseland_sampling'] = True
    # doptical['ross_nsample'] = 30
    # doptical['ross_pwl'] = 3

    pars.calcoptical = calcoptical
    pars.doptical = doptical

    return calcoptical, doptical


def f_sample (ross_nsample=30, ross_pwl=3, **kwargs):
    """
    [25.04.09]:CWO
    the sampling function; somewhat arbitrary
    - the exponent "pwl" (choose it >1) controls how the density of the sampling
        with higher pwl giving more weight to the edges of the cumulative weight
        function. So pwl=1 amounts to linear sampling, but then we may go wrong when
        the opacity function peaks outside the domain..
    - ns    :the number of sampling poins
    """
    n1 = ross_nsample//2
    farr = 0.5*(np.arange(1,n1+1)/n1)**ross_pwl
    return np.concatenate((farr, 1-farr[:-1][::-1]))


def w_x (x):
    """
    the weighting fn for Rosseland-mean calculations

    here x = h*nu /kT

    This integrates to unity, \int w_x dx = 1
    """
    return 15*x**4 /(8*np.pi**4 *(np.cosh(x)-1))


def prepare_kRosseland (doptical, agrid, Tgrid=1200):
    """
    for Rosseland mean opacity calculations

    TBD: the temperature along the grid (see Tarr below)
    """
    lgrid = doptical['wavelengthgrid']
    nlam = len(lgrid)
    nagr = len(agrid)

    Isample = np.zeros((nagr,nlam),dtype=bool)

    #TBD completely arbitrary now... CWO !!
    Tarr = 800 + 1000*np.linspace(0,nagr,nagr) /nagr

    fsample = f_sample (**doptical)

    #dimensional frequency matrix
    Xarr = cgs.hP*cgs.cl/(1e-4*lgrid) /(cgs.kB*Tarr[:,np.newaxis])
    Warr = w_x (Xarr)

    #this should be approximately unity
    check = np.trapz(Warr,Xarr,1)

    if doptical['rosseland_sampling']:

        Carr = sciint.cumulative_trapezoid(Warr,Xarr,1)

        for i in range(nagr):
            #TBD: check w/r searchsorted is as intended
            ii = np.searchsorted(-Carr[i], fsample)
            Isample[i,ii] = True

            ## so the contention is that sampling by these points is sufficient
            ## TBD: we could correct the weights Warr by c1
            iun = np.unique(ii)
            c1 = np.trapz(Warr[i,iun],Xarr[i,iun])
            line = f'{i} {len(iun)} {c1:10.4f} {check[i]:10.4f}'
            #print(line)

        print('[optical.prepare_kRosseland]Roseland sampling fraction is:', Isample.sum()/nagr/nlam)
    else:
        Isample[:,:] = True

    doptical['Warr'] = Warr
    doptical['Xarr'] = Xarr
    doptical['Isample'] = Isample


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


def root_eqn (zvec, polyidx):
    """
    [25.04.09] (CWO) -- This is the polynomial equation to solve
    when iterative methods (scipy.roots(...)) are used

    zvec    :is a vector of lenght 2 that are the real and imag 
             components of a complex number
    """
    z = complex(*zvec)
    n1 = len(polyidx)
    narr = np.arange(n1)[::-1]
    out = (polyidx *z**narr).sum() #fastest
    #out1 = np.polyval(polyidx, z)
    #out = np.polynomial.polynomial.polyval(z,polyidx[::-1]) #little faster
    return out.real, out.imag


def root_jac (zvec, polyidx):
    """
    the Jacobean to the above
    """
    z = complex(*zvec)

    #the derivative wrt z
    n1 = len(polyidx)
    narr = np.arange(1,n1)[::-1]
    df_dz = (polyidx[:-1] *narr *z**(narr-1)).sum()

    #dpoly = np.polyder(polyidx)
    #df_dz1 = np.polyval(dpoly, z)
    #df_dz = np.polynomial.polynomial.polyval(z, dpoly[::-1])

    jac = np.array([
        [ df_dz.real, -df_dz.imag],
        [ df_dz.imag,  df_dz.real]
    ])

    return jac

# def cal_eff_m (abundance, solid, wavelength, mspecies, polyidxmat, iw, doptical):
def cal_eff_m (abundance, solid, wavelength, mspecies, polyidxmat, doptical):
    """
    This function calculates the effective refractive index based on the composition
    Input:
        abundance   : abundance of each species
        solid       : a list of solid species
        wavelength  : wavelength on which to calculate the refractive index
        mspecies    : a list of refractive indices of each species
        polyidxmat  : Basically we are solving a complex polynomial equation, this is the polynomial index (see preparepoly)
        iw          : mask of the wavelength grid where calculations are perfomred
        doptical    : utility stuff
    """
    sortidx = np.argsort(-abundance)    # rank from large to small

    marr = np.zeros_like(wavelength, dtype=complex)

    # karr, = iw.nonzero()
    karr = np.arange(wavelength.size)

    for k in list(karr):
        # initialize
        mold = mspecies[k][sortidx[0]]
        # print(mold)
        # try to include everything at once. If that succeed, then no need to iterate

        polyidx = np.matmul(polyidxmat[k], abundance)

        #[25.04.09]CWO possibly factor 2 faster in the optimal case, which is 
        #already rather meagre. 
        #However, not infrequently, it fails to find the right solution
        #in which case it takes time and we need to rely on the 
        #more robust np.roots method anyway...
        #so, I don't recommend using it, except perhaps when the wavelength 
        #grid is dense
        # if doptical['scipy_root'] and k!=karr[0]:
        #     sol = sciop.root(root_eqn, ztry, polyidx, jac=root_jac, options={'maxfev':20})

        #     if sol.success and sol.x[0]>0 and sol.x[1]>0:
        #         marr[k] = complex(*sol.x)
        #         ztry = sol.x
        #         continue

        #TBD: use numpy's polynomial allroots method
        #allroots = np.roots(polyidx)
        allroots = np.polynomial.polynomial.polyroots(polyidx[::-1])

        physicalidx = np.where((allroots.real>0)&(allroots.imag>0))[0]    # physical solution for the refractory index.
        # if there is only one root with both positive real and imaginary part, then we are done
        if len(physicalidx)==1:
            marr[k] = allroots[physicalidx]
            ztry = np.array([allroots[physicalidx].real, allroots[physicalidx].imag])[:,0]
            continue


        # find the polynomial index of the equation, with a new species
        # Methematically I cannot proof that the polynomial equation only have one physical solution
        # If there is more than one solution, then I have to add a species once at a time (from abundance order) to find the physical one.
        # But in practice I have never seen more than one solution.
        # So the readers can ignore all the big loop below.
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


def cal_meff_and_kappa (agrain, abundance, rho, solid, doptical):
    """
    calculate effective medium optical constants:
    - abundance:        :2d array of solid mass fractions (species-id, particle)     
    - wavelengthgrid    :1d array of wavelength (micron)
    - solid             :1d array giving the name for the solid species

    in our case each particle corresponds to a single grid point
    """

    wavelengthgrid = doptical['wavelengthgrid']
    optooldir = doptical['optooldir']

    Nwlen = len(wavelengthgrid)
    Nsolid = len(solid)
    nagr = len(agrain)

    # load the bulk refractive index for the pure material
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


    #"partial" pre-populates a function..
    worker_fun = partial(meff_kappa_single, agrain, abundance, rho, solid, wavelengthgrid, 
                         mspecies, polyidxmat, doptical)  

    if doptical['multiproc']:
        with Pool(processes=doptical['multi_nproc']) as pool:
            koutL = pool.map(worker_fun, np.arange(nagr))

    else:
        koutL = []
        for i in range(nagr): 
            kout = worker_fun (i)
            koutL.append(kout)
    
    # combine the results into one matrix
    kabsmat = np.empty((Nwlen, nagr))
    kscamat = np.empty((Nwlen, nagr))
    kextmat = np.empty((Nwlen, nagr))
    gscamat = np.empty((Nwlen, nagr))

    for i in range(nagr):
        kabsmat[:, i] = koutL[i].kabs[0]
        kscamat[:, i] = koutL[i].ksca[0]
        kextmat[:, i] = koutL[i].kext[0]
        gscamat[:, i] = koutL[i].gsca[0]

    kappadata = {'wlen':koutL[0].lam, 'kabs':kabsmat, 'ksca':kscamat, 'kext':kextmat, 'gsca':gscamat}

    return kappadata
    # for i in range(0,690,30):
    #     line = f'{i:4d} {koutL[i]:9.2f}'
    #     print(line)


def meff_kappa_single (agrain, abundance, rho, solid, wavelengthgrid, 
                       mspecies, polyidxmat, doptical, i):
    '''
    Compute the effective refractive index and opacity for a single layer
    return: the Rosseland mean opacity
    '''

    print(f'\r[calmeff.cal_eff_m_all]:performing effective medium on particle {i}/{abundance.shape[1]}', end="")

    optooldir = doptical['optooldir']

    #this is the wavelength grid to use
    # iw = doptical['Isample'][i]
    # nwact = sum(iw) ## CWO!
    nwact = len(wavelengthgrid)
    iw = np.ones(nwact)
    marr = cal_eff_m(abundance[:, i], solid, wavelengthgrid, mspecies, polyidxmat, doptical)

    # write for particle i
    if doptical['dirmeff'] != None:
        filename = doptical['dirmeff'] + f'/{i}.lnk'
        with open(filename, 'w') as opt:
            opt.write(f'{nwact} {rho[i]}\n')
            karr, = iw.nonzero()
            for j in list(karr):
                opt.write(f'{wavelengthgrid[j]} {marr[j].real} {marr[j].imag}\n')

    # run optool...
    # command = optooldir+f'/optool {filename} -q -p 0.25 -a {agrain[i]*1e4} -l {filename} -o {doptical['dirkappa']}'
    command = optooldir+f'/optool {filename} -q -xlim 1e3 -p 0.25 -a {agrain[i]*1e4} -l {filename}'

    p = optool.particle(command, silent=True)

    # save the opacity for particle i
    if doptical['dirkappa'] != None:
        savefile = f'{i}.txt'
        with open(doptical['dirkappa'] + '/'+savefile, 'w') as opt:
            opt.write('#optical properties for particle ...\n')
            opt.write('#cols::[wavelength,kappa_abs,kappa_sca,kappa_ext,asymmetry_parameter]\n')
            opt.write('#colunits::[micron,cm2/g,cm2/g,cm2/g,]\n')
            for j in range(nwact):
                sfmt = 5*'{:10.3e} '
                line = sfmt.format(p.lam[j], p.kabs[0,j], p.ksca[0,j], p.kext[0,j], p.gsca[0,j])
                opt.write(line+'\n')

    #do the Rosseland mean
    # kros_ext = -np.trapz(doptical['Warr'][i,karr]*p.kext[0], doptical['Xarr'][i,karr])

    return p
