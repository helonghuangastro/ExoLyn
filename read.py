import numpy as np
import constants as cnt
import pdb
import parameters as pars

class chemdata():
    def __init__(self, reaction_file, gibbs_file):
        self.gibbsdata = np.genfromtxt(gibbs_file, names=True, deletechars='')
        self.mollist = []    # name of all molecules
        self.molecules = {}    # store all the molecular data
        self.reactions = self.get_reaction(reaction_file)
        self.reactionidx = [self.molecules[solid+'(s)'].reactionidx for solid in pars.solid]
        self.mugas = np.array([self.molecules[gas].mu for gas in pars.gas])
        self.musolid = np.array([self.molecules[solid+'(s)'].mu for solid in pars.solid])
        for reaction in self.reactions:
            reaction.cal_dG(self.gibbsdata, self.molecules)

    def get_reaction(self, filename):
        reactions = []
        i = 0
        with open(filename, 'r') as ipt:
            for line in ipt:
                if line.startswith('#'):
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
        for molecule in self.product.keys():
            if molecule.strip('(s)') in pars.solid:
                self.solidindex = pars.solid.index(molecule.strip('(s)'))    # the index of the product solid in the solid list
        self.gasst = np.zeros(len(pars.gas))
        for i, gas in enumerate(pars.gas):
            if gas in self.reactant.keys():
                self.gasst[i] = self.reactant[gas]
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

element_data = {'H':1, 'He':4, 'O':16, 'Mg':24, 'Al':27, 'Si':28, 'S':32, 'Ca':40, 'Ti':48, 'Fe':56}    # all element data 

class parameters():
    def __init__(self, parafilename):
        ''' read in the parameter file (.txt file) and save the parameters in the class '''
        with open(parafilename, 'r') as ipt:
            i = -1
            for line in ipt:
                i += 1
                # ignore comment and space line
                if line.startswith('#') or line.isspace():
                    continue
                if '#' in line:
                    commentidx = line.find('#')
                    line = line[:commentidx]    # remove comments
                    line = line.strip()    # remove space

                # load name-value pair
                if line.count('=')!=1:
                    raise Exception(parafilename + ' line ' + str(i) + ': Should only have one \'=\' sign .')
                equalsignidx = line.find('=')
                paraname = line[:equalsignidx].strip()
                valuestr = line[(equalsignidx+1):].strip()    # The value string of the line

                # get the type of the valuestring
                valuetype = self.mygettype(valuestr)

                # read list
                if valuetype == 5:
                    valuestr = valuestr.strip('[').strip(']')
                    valuelist = valuestr.split(', ')
                    listtype = []
                    for singlevalue in valuelist:
                        listtype.append(self.mygettype(singlevalue))
                    if listtype.count(listtype[0]) != len(listtype):
                        raise Exception(parafilename + ' line ' + str(i) + ': List should have the same type')
                    # read value of each element
                    value = []
                    for singlevalue in valuelist:
                        if listtype[0] == 0:
                            if singlevalue == 'True':
                                value.append(True)
                            else:
                                value.append(False)
                        elif listtype[0] == 1:
                            value.append(int(singlevalue))
                        elif listtype[0] == 2:
                            value.append(self.getfloat(singlevalue))
                        else:
                            value.append(singlevalue.strip('\''))
                    if listtype[0] != 3:
                        value = np.array(value)

                # evaluate the value
                # Only compatible with *, /, **
                elif valuetype == 4:
                    valuelist = valuestr.split(' ')
                    # transform all the values from strring to float
                    for j, singlevaluestr in enumerate(valuelist):
                        if singlevaluestr in ['*', '/', '**']:
                            continue
                        if singlevaluestr in self.__dict__.keys():
                            singlevalue = getattr(self, singlevaluestr)
                        else:
                            singlevalue = self.getfloat(singlevaluestr)
                        valuelist[j] = singlevalue
                    # operate all the ** in the list
                    while('**' in valuelist):
                        opidx = valuelist.index('**')    # index of the operator
                        resultvalue = valuelist[opidx-1] ** valuelist[opidx+1]
                        del valuelist[opidx:(opidx+2)]
                        valuelist[opidx-1] = resultvalue
                    # operate all the * or / in the list
                    while('*' in valuelist or '/' in valuelist):
                        if valuelist[1] == '*':
                            resultvalue = valuelist[0] * valuelist[2]
                        elif valuelist[1] == '/':
                            resultvalue = valuelist[0] / valuelist[2]
                        del valuelist[1:3]
                        valuelist[0] = resultvalue
                    # after this should already operated all the operators
                    if len(valuelist) != 1:
                        raise Exception(parafilename + ' line ' + str(i) + ': Wrong syntax for operation.')
                    value = valuelist[0]

                # read other datatype
                elif valuetype == 0:
                    if valuestr == 'True':
                        value = True
                    else:
                        value = False
                elif valuetype == 1:
                    value = int(valuestr)
                elif valuetype == 2:
                    value = self.getfloat(valuestr)
                else:
                    value = valuestr.strip('\'')

                # save the attr to pars class
                setattr(self, paraname, value)

        return

    def mygettype(self, valuestr):
        '''
        return the type of the parameter:
        0 bool
        1 int
        2 float
        3 str
        4 exp
        5 list
        '''
        if valuestr.startswith('['):
            return 5
        if valuestr.startswith('\''):
            return 3
        if valuestr == 'True' or valuestr == 'False':
            return 0
        if ' ' in valuestr:
            return 4
        if '.' not in valuestr and 'e' not in valuestr:
            return 1
        else:
            return 2

    def getfloat(self, valuestr):
        '''
        get a float from a string, compatible with constant.py and numpy
        '''
        if 'cnt.' in valuestr:
            value = getattr(cnt, valuestr[4:])
        elif 'np.' in valuestr:
            value = getattr(np, valuestr[3:])
        else:
            value = float(valuestr)

        return value

if __name__ == '__main__':
    pars = parameters('parameters.txt')
