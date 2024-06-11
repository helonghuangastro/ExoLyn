import numpy as np
import constants as cnt
import pdb
import parameters as pars
import os

class chemdata():
    def __init__(self, gibbs_file):
        # read the reactions and molecules
        self.read_reaction(pars.parafilename)
        self.addgasmolecule(pars.gas)

        # set the properties for solid species
        solidlist = [molename.strip('(s)') for molename in self.mollist if molename.endswith('(s)')]    # all the solid that appear in the reactions
        self.musolid = np.array([self.molecules[solid+'(s)'].mu for solid in solidlist])    # the molecular weight for each solid
        self.reactionidx = [self.molecules[solid+'(s)'].reactionidx for solid in solidlist]    # the index of reaction that correspond to each solid
        pars.solid = solidlist
        self.rhosolid = self.readrho()

        # set the properties for the gas
        self.mugas = np.array([self.molecules[gas].mu for gas in pars.gas])
        self.gasst = np.array([reaction.gasst for reaction in self.reactions])    # gas stoichemics for all reactions
        
        # set the properties for the reactions
        for reaction in self.reactions:
            reaction.solidindex = solidlist.index(reaction.solid)    # the index of solid in the solidlist
        self.solidindex = np.array([solidlist.index(reaction.solid) for reaction in self.reactions])

        # read gibbs coeffecient data
        gibbsdata = np.genfromtxt(gibbs_file, names=True, deletechars='', comments='#')
        self.gibbsTref = gibbsdata['Tref']
        self.add_gibbs(gibbsdata, self.molecules)
        self.gibbsfitcoeffs = self.readgibbsfit(gibbsfitfile=os.path.join(pars.rootdir,'tables/gibbsfit.txt'))
        for reaction in self.reactions:
            reaction.cal_dG(self.molecules, self.gibbsTref)

        return

    def read_reaction(self, filename):
        mollist = []
        molecules = {}
        reactions = []
        i = 0
        with open(filename, 'r') as ipt:
            parameterline = True
            for line in ipt:
                if line.startswith('====='):
                    parameterline = False
                    continue
                if parameterline:
                    continue
                # skip comments
                if line.startswith('#') or line.isspace():
                    continue

                isreactant = True    # reactant

                line = line.strip()
                words = line.split(' ')
                
                reactant = {}
                product = {}
                solid = None
                for equaterm in words:
                    # skip + and ->
                    if equaterm=='+':
                        continue
                    if equaterm=='->':
                        isreactant = False
                        continue
                    
                    # equaterm is the term in the equation, like 2H2O
                    # moleterm is the molecular term, like H2O or SiO2(s)
                    # seperate st and molecular term
                    st, moleterm = get_mole(equaterm)
                    # add the molecule to the class
                    if moleterm not in mollist:
                        if moleterm.endswith('(s)'):
                            mole = solidmol(moleterm)
                        else:
                            mole = molecule(moleterm)
                        mollist.append(moleterm)
                        molecules[moleterm] = mole
                    else:
                        mole = molecules[moleterm]

                    if mole.solid:
                        solid = mole.name
                        mole.addindex(i)
                    
                    if isreactant:
                        reactant[moleterm] = st
                    else:
                        product[moleterm] = st
                
                # if the stoichiometric number of solid is not 1
                stsolid = product[solid+'(s)']
                for name in reactant.keys():
                    reactant[name] /= stsolid
                for name in product.keys():
                    product[name] /= stsolid

                readreaction = reaction(reactant, product, solid)
                reactions.append(readreaction)
                i += 1

        self.reactions = reactions
        self.mollist = mollist
        self.molecules = molecules

        return

    def addgasmolecule(self, gasmollist):
        ''' Add gas molecules that's not in the reaction, but in the parameter file '''
        for gasname in gasmollist:
            if gasname not in self.molecules:
                mole = molecule(gasname)
                self.molecules[gasname] = mole
        return

    def add_gibbs(self, gibbsdata, molecules):
        ''' Add gibbs data to the molecule class '''
        MoleWithGibbs = gibbsdata.dtype.names
        for molename in molecules:
            if molename in MoleWithGibbs:
                molecules[molename].hasgibbs = True
                molecules[molename].gibbs = gibbsdata[molename]
        return

    def readrho(self):
        rhofilename = pars.rootdir + '/tables/density.txt'
        densitydict = {}
        # read the density file and get all the densities
        with open(rhofilename, 'r') as ipt:
            while(True):
                line = ipt.readline()
                if not line:
                    break
                if line.startswith('#'):
                    continue
                name, density = line.split(' ')
                densitydict[name] = float(density)
        
        # the density of each solid species
        rhosolid = np.array([densitydict[solid] for solid in pars.solid])
        return rhosolid
 
    def readgibbsfit(self, gibbsfitfile):
        ''' read the fitting coefficient for the gibbs energy '''
        gibbsfitcoeffs = {}
        allfitmoles = {}
        # read every line in the gibbs fit file
        with open(gibbsfitfile, 'r') as ipt:
            lines = ipt.readlines()
        
        # read the name of molecules that have fitting results
        for i, line in enumerate(lines):
            allfitmoles[line.strip().split(' ')[0]] = i

        # read the fitting result for the molecules
        for mole in self.molecules.keys():
            if mole in allfitmoles.keys():
                linelist = lines[allfitmoles[mole]].strip().split(' ')    # split the line
                # in some cases, need gasphase Gibbs energy to fit the solid phase ones.
                if linelist[1]=='3' or linelist[1]=='6':
                    gasmole = mole.strip('(s)')
                    if gasmole in allfitmoles.keys():
                        linelistgas = lines[allfitmoles[gasmole]].strip().split(' ')
                        coefflistgas = [int(linelistgas[1])]
                        for i in range(2, len(linelistgas)):
                            coefflistgas.append(float(linelistgas[i]))
                        gibbsfitcoeffs[gasmole] = coefflistgas
                coefflist = [int(linelist[1])]       # fitting coefficient, the first one is the fit fomular
                for i in range(2, len(linelist)):
                    coefflist.append(float(linelist[i]))
                gibbsfitcoeffs[mole] = coefflist

        return gibbsfitcoeffs

class reaction():
    def __init__(self, reactant, product, solid):
        self.reactant = reactant
        self.product = product
        self.solid = solid
        self.labelreaction()
        return

    def labelreaction(self):
        '''
        compare the reaction with the vapor and solid species in parameters.py to get the contribution to the species we considered
        '''
        self.gasst = np.zeros(len(pars.gas))
        for i, gas in enumerate(pars.gas):
            if gas in self.reactant.keys():
                self.gasst[i] = self.reactant[gas]
            # support the cases where vapor is in the product
            elif gas in self.product.keys() and gas!=self.solid:
                self.gasst[i] = -self.product[gas]
        return

    def cal_dG(self, molecules, Tref):
        delG = 0
        netnu = 0
        munu = 1

        nogibbs = False    # any molecule has no gibbs data

        for molename, st in self.product.items():
            if molecules[molename].hasgibbs:
                delG += molecules[molename].gibbs*st
            else:
                nogibbs = True
            if molename.strip('(s)')!=self.solid:
                netnu -= st
                munu /= molecules[molename].mu**st
        for molename, st in self.reactant.items():
            if molecules[molename].hasgibbs:
                delG -= molecules[molename].gibbs*st
            else:
                nogibbs = True
            netnu += st
            munu *= molecules[molename].mu**st    # this is defined as mu**nu, which will be used in the Sbase later

        delG *= 1e3    # Now the unit is J, consistent with unit of R (ideal gas const.)

        if nogibbs:
            delG = np.array(len(Tref) * [np.nan])

        self.delG = delG
        self.netnu = netnu
        self.munu = munu

        return

class molecule():
    def __init__(self, moleterm):
        self.name = moleterm.strip('(s)')
        self.solid = moleterm.endswith('(s)')
        self.element = {}    # the dictionary storing how many atoms each element is in the molecule
        self.hasgibbs = False       # whether this molecule has gibbs data
        self.hasgibbsfit = False    # whether this molecule has gibbs fit coeffecient

        molename = moleterm.strip('(s)')
        while molename!='':
            i = 1
            st = False    # whether the atom in the molecule has stoichiometric number
            while i<len(molename):
                if st==False and molename[i].isdigit():
                    st = i
                if molename[i].isupper():    # next element
                    break
                i+=1

            if st==False:
                self.element[molename[:i]] = 1
            else:
                self.element[molename[:st]] = int(molename[st:i])
            molename = molename[i:]

        # calculate the mean molecular weight
        self.mu = 0
        for ele, st in self.element.items():
            self.mu += element_data[ele]*st

        return

class solidmol(molecule):
    def __init__(self, moleterm):
        super(solidmol, self).__init__(moleterm)
        self.reactionidx = []    # Save which reaction is related to this solid
        return

    def addindex(self, reactionidx):
        self.reactionidx.append(reactionidx)
        return

def cal_gibbs(chem, mole, T):
    '''
    get the gibbs energy of one species by fitting or extrapolating Gibbs energy
    I feel that this function can be combined with the above function
    '''
    doint = False    # whether we interpolate
    dofit = False    # whether we fit
    isext = False    # whether the interpolate extrapolate
    # if T within valid temperature of JANAF table, just interpolate
    if chem.molecules[mole].hasgibbs:
        doint = True
    else:
        dofit = False

    # first interpolate
    if doint:
        gibbsref = chem.molecules[mole].gibbs * 1e3
        idxvalid = ~np.isnan(gibbsref)
        gibbsref = gibbsref[idxvalid]
        Tref = chem.gibbsTref[idxvalid]
        gibbs = np.interp(T, Tref, gibbsref)

        # find the temperature where extrapolate
        idxext = (T<np.min(Tref)) | (T>np.max(Tref))
        if (idxext==True).any():
            isext = True

    # when extrapolate or the gibbs energy not in the table, using fit expression
    if (doint and isext) or (not doint):    # NEED to fit
        if (mole in chem.gibbsfitcoeffs.keys()):    # ABLE to fit
            dofit = True
        else:
            dofit = False

    # if has the fitting formular of mole and extrapolate, use the fitting formular
    if dofit:
        coefflist = chem.gibbsfitcoeffs[mole]
        Nexp = coefflist[0]
        coeff = coefflist[1:]
        # fitting formular 0
        if Nexp == 0:
            a1, a2, a3, a4 = coeff
            gibbsfit = cnt.R*T*(a1*np.log(T) + a2 + a3*T + a4*T**2)
        elif Nexp == 2:
            a0, a1, a2, a3, a4 = coeff
            gibbsfit = a0/T + a1 + a2*T + a3*T**2 + a4*T**3
            gibbsfit *= 1e3    # convert the unit to J
        elif Nexp == 3:
            molegas = mole.strip('(s)')
            try:
                gibbsgas = cal_gibbs(chem, molegas, T)    # in J
            except:
                print(f'[funcs.cal_gibbs]WARNING: You are fitting gibbs energy of {mole} with formular 3, \
                which requires the corresponding gas phase gibbs energy. However, the gibbs energy of {molegas} is not found in the gibbsfile.')
                dofit = False    # cannot get the gas phase gibbs energy
            else:
                a0, a1, a2, a3, a4 = coeff
                gibbsfit = gibbsgas + cnt.R * (a0 + a1*T + a2*T**2 + a3*T**3 + a4*T**4)
        elif Nexp == 5:
            a0, a1, a2, a3, a4 = coeff
            gibbsfit = cnt.R*(a0 + a1*np.log(T)*T + a2*T + a3*T**2 + a4*T**3)
        elif Nexp == 6:
            molegas = mole.strip('(s)')
            try:
                gibbsgas = gibbsfit(chem, molegas, T)    # in J
            except:
                print(f'[funcs.cal_gibbs]WARNING: You are fitting gibbs energy of {mole} with formular 6, \
                which requires the corresponding gas phase gibbs energy. However, the gibbs energy of {molegas} is not found in the gibbsfile.')
                dofit = False    # cannot get the gas phase gibbs energy
            else:
                a0, a1, a2, a3, a4 = coeff
                gibbsfit = gibbsgas + cnt.R * (a0 + a1*T + a2*T*np.log(T) + a3*T**2 + a4*T**3)
        elif Nexp == 8:
            pdb.set_trace()
            molegas = mole.strip('(s)')
            try:
                gibbsgas = cal_gibbs(chem, molegas, T)    # in J
            except:
                print(f'[funcs.cal_gibbs]WARNING: You are fitting gibbs energy of {mole} with formular 8, \
                which requires the corresponding gas phase gibbs energy. However, the gibbs energy of {molegas} is not found in the gibbsfile.')
                dofit = False    # cannot get the gas phase gibbs energy
            else:
                a0, a1, a2 = coeff
                gibbsfit = gibbsgas + cnt.R * (a0*T + a1 + a2/T)

    # cannot find the gibbs energy
    if (not dofit) and (not doint):
        raise Exception('[funcs.gibbsfit] Cannot find the gibbs energy of {mole} because either gibbstable and gibbsfittable do not contain it.')
    # only use fitting formulars
    if (not doint) and dofit:
        gibbs = gibbsfit
    # replace the extrapolated gibbs energy with the fit ones
    if doint and isext and dofit:
        gibbs[idxext] = gibbsfit[idxext]

    return gibbs

def get_mole(moleterm):
    '''from the term in the chemistry equation getting stoichiometric number and molecule name'''
    i = 0
    while moleterm[i].isdigit():
        i = i+1
    molename = moleterm[i:]
    if i==0:
        st = 1
    else:
        st = int(moleterm[:i])
    return st, molename

element_data = {'H':1, 'He':4, 'C':12, 'N':14, 'O':16, 'Na':23, 'Mg':24, 'Al':27, 'Si':28, 'S':32, 'Cl':35.5, 'K':39, 'Ca':40, 'Ti':48, 'Cr':52, 'Mn':55, 'Fe':56, 'Zn':65}    # all element data 
