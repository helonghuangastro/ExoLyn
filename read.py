import numpy as np
import constants as cnt
import pdb
import parameters as pars

class chemdata():
    def __init__(self, gibbs_file):
        self.gibbsdata = np.genfromtxt(gibbs_file, names=True, deletechars='', comments='#')
        self.mollist = []    # name of all molecules
        self.molecules = {}    # store all the molecular data
        self.reactions = self.get_reaction(pars.parafilename)
        ## BUG?? 
        ##I get an error here when gas is in pars.gas but not in self.molecules:
        ##
        ##      self.mugas = np.array([self.molecules[gas].mu for gas in pars.gas])
        ## KeyError: 'TiO'
        ##
        ##I don't see why we should get this error. If a molecule is non-reactant, let it be...
        ##
        self.mugas = np.array([self.molecules[gas].mu for gas in pars.gas])
        for reaction in self.reactions:
            reaction.cal_dG(self.gibbsdata, self.molecules)
        solidlist = [molename.strip('(s)') for molename in self.mollist if molename.endswith('(s)')]    # all the solid that appear in the reactions
        self.reactionidx = [self.molecules[solid+'(s)'].reactionidx for solid in solidlist]    # the index of reaction that correspond to each solid
        self.musolid = np.array([self.molecules[solid+'(s)'].mu for solid in solidlist])    # the molecular weight for each solid
        for reaction in self.reactions:
            reaction.solidindex = solidlist.index(reaction.solid)    # the index of solid in the solidlist
        self.solidindex = np.array([solidlist.index(reaction.solid) for reaction in self.reactions])
        self.gasst = np.array([reaction.gasst for reaction in self.reactions])    # gas stoichemics for all reactions
        pars.solid = solidlist

    def get_reaction(self, filename):
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
                    if moleterm not in self.mollist:
                        if moleterm.endswith('(s)'):
                            mole = solidmol(moleterm)
                        else:
                            mole = molecule(moleterm)
                        self.mollist.append(moleterm)
                        self.molecules[moleterm] = mole
                    else:
                        mole = self.molecules[moleterm]

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

            return reactions

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

    def cal_dG(self, moldata, molecules):
        delG = 0
        netnu = 0
        munu = 1
        for molename, st in self.product.items():
            delG += moldata[molename]*st
            if molename.strip('(s)')!=self.solid:
                netnu -= st
        for molename, st in self.reactant.items():
            delG -= moldata[molename]*st
            netnu += st
            munu *= molecules[molename].mu**st    # this is defined as mu**nu, which will be used in the Sbase later

        delG *= 1e3    # Now the unit is J, consistent with unit of R (ideal gas const.)

        self.delG = delG
        self.netnu = netnu
        self.munu = munu

        return

class molecule():
    def __init__(self, moleterm):
        self.name = moleterm.strip('(s)')
        self.solid = moleterm.endswith('(s)')
        self.element = {}    # the dictionary storing how many atoms each element is in the molecule

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

element_data = {'H':1, 'He':4, 'C':12, 'N':14, 'O':16, 'Na':23, 'Mg':24, 'Al':27, 'Si':28, 'S':32, 'Cl':35.5, 'K':39, 'Ca':40, 'Ti':48, 'Cr':52, 'Mn':55, 'Fe':56}    # all element data 
