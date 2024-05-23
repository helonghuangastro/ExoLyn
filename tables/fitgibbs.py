import numpy as np
from matplotlib import pyplot as plt
import os

R = 8.314 / 1e3    # ideal gas constant in kJ/mol

def fitfuncs0(T, coeffs):
    '''
    fitting function for the gas phase gibbs energy
    G_f/R = a1 * logT + a2*T + a3*T^2 + a4*T^3
    '''
    a1, a2, a3, a4 = coeffs
    Gf = R*T*(a1*np.log(T) + a2 + a3*T + a4*T**2)
    return Gf    # return Gibbs energy in kJ/mol

def fitfuncs2(T, coeffs):
    '''
    fitting function for the solid phase gibbs energy, expresssion 2 in GGChem paper
    G_f = a0/T + a1 + a2*T + a3*T^2 + a4*T^3
    '''
    a0, a1, a2, a3, a4 = coeffs
    Gf = a0/T + a1 + a2*T + a3*T**2 + a4*T**3
    return Gf

def fitfuncs3(T, coeffs, gibbsgas):
    '''
    fitting function for the solid phase gibbs energy, expresssion 3 in GGChem paper
    G_f(solid)-G_f(gas) = R(c0 + c1*T + c2*T^2 + c3*T^3 + c4*T^4)
    '''
    a0, a1, a2, a3, a4 = coeffs
    Gf = gibbsgas + R * (a0 + a1*T + a2*T**2 + a3*T**3 + a4*T**4)
    return Gf

def fitfuncs5(T, coeffs):
    '''
    fitting function for the solid phase gibbs energy, expresssion 5 in GGChem paper
    G_f/R = a0 + a1 * logT + a2*T + a3*T^2 + a4*T^3
    '''
    a0, a1, a2, a3, a4 = coeffs
    Gf = R*(a0 + a1*np.log(T)*T + a2*T + a3*T**2 + a4*T**3)
    return Gf    # return Gibbs energy in kJ/mol

def fitfuncs6(T, coeffs, gibbsgas):
    '''
    fitting function for the solid phase gibbs energy, expresssion 6 in GGChem paper
    G_f(solid)-G_f(gas) = R(c0 + c1*T + c2*T*logT + c3*T^2 + c4*T^3)
    '''
    a0, a1, a2, a3, a4 = coeffs
    Gf = gibbsgas + R * (a0 + a1*T + a2*T*np.log(T) + a3*T**2 + a4*T**3)
    return Gf

def fitfuncs8(T, coeffs, gibbsgas):
    '''
    fitting function for the solid phase gibbs energy, expresssion 6 in GGChem paper
    G_f(solid)-G_f(gas) = R(c0 + c1*T + c2*T*logT + c3*T^2 + c4*T^3)
    '''
    a0, a1, a2 = coeffs
    Gf = gibbsgas + R * (a0 + a1/T + a2/T**2)
    return

def gibbsfit(gibbsfile, expressiondict, outputfile=None, isplot=False):
    gibbsdata = np.genfromtxt(gibbsfile, names=True, deletechars=" !#$%&'*+, -./:;<=>?@[\\]^{|}~")

    coefflist = []
    # fit every molecule
    for mole, Nexp in expressiondict.items():
        Tarr = gibbsdata['Tref']
        gibbsarr = gibbsdata[mole]

        # remove nan gibbs values
        idxvalid = ~np.isnan(gibbsarr)
        gibbsarr = gibbsarr[idxvalid]
        Tarr     = Tarr[idxvalid]

        if Nexp == 0:
            # fitting expression for gas
            Amat = np.vstack((Tarr*np.log(Tarr), Tarr, Tarr**2, Tarr**3)).T
            Barr = gibbsarr / R
            coeff = np.linalg.lstsq(Amat, Barr, rcond=1)[0]

        elif Nexp == 2:
            # using expression 2 to fit
            Amat = np.vstack((1/Tarr, np.ones_like(Tarr), Tarr, Tarr**2, Tarr**3)).T
            Barr = gibbsarr
            coeff = np.linalg.lstsq(Amat, Barr, rcond=1)[0]

        elif Nexp == 3:
            # using expression 3 to fit
            # check whether gas phase gibbs energy in the database:
            molegas = mole.strip('(s)')
            if molegas not in gibbsdata.dtype.names:
                raise Exception(f'You are fitting gibbs energy of {mole} with formular 3, \
                which requires the corresponding gas phase gibbs energy. However, the gibbs energy of {molegas} is not found in {gibbsfile}.')
            # get gas phase gibbs energy data
            gibbsgas = gibbsdata[molegas]
            gibbsgas = gibbsgas[idxvalid]
            # if gas phase molecules are not available, use the fitted formular to calculate it
            if np.isnan(gibbsgas).any():
                print(f'Warning: using fitted formular for the gibbs energy of {gibbsgas} at some temperatures.')
                idxnodata = np.isnan(gibbsgas)
                coeffgas = gibbsfit(gibbsfile, {molegas: 0})
                gibbsgas[idxnodata] = fitfuncs0(Tarr[idxnodata], coeffgas)
            Amat = np.vstack((np.ones_like(Tarr), Tarr, Tarr**2, Tarr**3, Tarr**4)).T
            Barr = (gibbsarr - gibbsgas) / R
            coeff = np.linalg.lstsq(Amat, Barr, rcond=1)[0]

        elif Nexp == 5:
            # use expression 5 to fit
            Amat = np.vstack((np.ones_like(Tarr), Tarr*np.log(Tarr), Tarr, Tarr**2, Tarr**3)).T
            Barr = gibbsarr / R
            coeff = np.linalg.lstsq(Amat, Barr, rcond=1)[0]

        elif Nexp == 6:
            # using expression 6 to fit
            # check whether gas phase gibbs energy in the database:
            molegas = mole.strip('(s)')
            if molegas not in gibbsdata.dtype.names:
                raise Exception(f'You are fitting gibbs energy of {mole} with formular 3, \
                which requires the corresponding gas phase gibbs energy. However, the gibbs energy of {molegas} is not found in {gibbsfile}.')
            # get gas phase gibbs energy data
            gibbsgas = gibbsdata[molegas]
            gibbsgas = gibbsgas[idxvalid]
            # if gas phase molecules are not available, use the fitted formular to calculate it
            if np.isnan(gibbsgas).any():
                print(f'Warning: using fitted formular for the gibbs energy of {gibbsgas} at some temperatures.')
                idxnodata = np.isnan(gibbsgas)
                coeffgas = gibbsfit(gibbsfile, {molegas: 0})
                gibbsgas[idxnodata] = fitfuncs0(Tarr[idxnodata], coeffgas)
            Amat = np.vstack((np.ones_like(Tarr), Tarr, Tarr*np.log(Tarr), Tarr**2, Tarr**3)).T
            Barr = (gibbsarr - gibbsgas) / R
            coeff = np.linalg.lstsq(Amat, Barr, rcond=1)[0]

        coefflist.append(coeff)

        # plot the fitting result
        if isplot:
            plt.plot(Tarr, gibbsarr, 'o')
            Tref = np.linspace(100, 6000, 100)
            plt.plot(Tref, fitfuncs5(Tref, coeff))
            # plt.plot(Tref, fitfuncs6(Tref, coeff, np.interp(Tref, Tarr, gibbsgas)))
            plt.show()

    # save the fited coefficient
    if outputfile!=None:
        with open(outputfile, 'w') as opt:
            i = 0
            for mole, Nexp in expressiondict.items():
                line = mole + ' ' + str(Nexp) + ' '    # which expression to use when fit
                for coeff in coefflist[i]:                 # write the coefficients
                    line = line + str(coeff) + ' '
                line = line + '\n'
                opt.write(line)
                i = i+1
            for mole, fitcoeff in addexpressiondict.items():
                line = mole + ' '
                for coeff in fitcoeff:
                    line = line + str(coeff) + ' '
                line = line + '\n'
                opt.write(line)

    return coeff

fitdict = {'H2O':0, 'Na':0, 'TiO':0, 'H2':0, 'H2S':0, 'K':0,
            'SiO':0, 'Zn':0, 'Al':0, 'Fe':0, 'Ca':0, 'Mg':0,
            'FeS':0, 'KCl':0, 'TiO2':0, 'SiO2':0, 'MgO':0, 
            'HCl':0, 'NaCl':0, 'FeO':0, 'NH3':0,
            'MgSiO3(s)':2, 'Mg2SiO4(s)':2, 'TiO2(s)':3, 'SiO2(s)':3, 
            'Fe(s)':3, 'FeS(s)':3, 'FeO(s)':3, 'MgO(s)':3, 'TiO(s)':3,
            'NaCl(s)':3, 'KCl(s)':3, 'Al2O3(s)':5, 'Fe2O3(s)':5, 'H2O(s)':6, 
            'Na2S(s)':2}

addexpressiondict = {'NH3(s)': [8, 10.53, -2161, -86594]}

coeff = gibbsfit('gibbs_test.txt', fitdict, outputfile='gibbsfit.txt', isplot=False)
# coeff = gibbsfit('gibbs_test.txt', {'ZnS(s)':5}, outputfile=None, isplot=True)
