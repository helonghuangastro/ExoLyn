import numpy as np
import constants as cnt
import sys, os

# add path to /AtmCloud dir
# HH: I don't like this line, because sometimes the file I ran is not in the src folder 
# (for example, it may be calmeff.py in util, postprocess.py in userfuncs or paperplot.py in paperplot)
# I suggest that the user should write down rootdir explicitly in parameters.txt
# (not tested thoroughly)
# relcommand = sys.argv[0]
# ix = relcommand.index('src')
# rootdir = relcommand[:ix]

def replace_environment (sss):
    '''
    check w/r "string" contains environment variables ($...)
    and replace accordingly
    '''
    while True:
        i0 = sss.find('$')
        if i0>-1:
            i1 = sss.find(r'/',i0)  #this is a bit tricky...
            envvar = sss[i0:i1]     #including $ sign
            if envvar[1:] in os.environ:
                sss =sss.replace(envvar,os.environ.get(envvar[1:]))
        else:
            break

    return sss


def mygettype(valuestr):
    '''
    return the type of the parameter:
    0 bool
    1 int
    2 float
    3 str
    4 exp
    5 list
    '''
    if valuestr.startswith('{'):
        return 6
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

def getfloat(valuestr):
    '''
    get a float from a string, compatible with constant.py and numpy
    '''
    valuetype = mygettype(valuestr)
    if valuetype == 3:
        value = valuestr.strip('\'')
    elif 'cnt.' in valuestr:
        value = getattr(cnt, valuestr[4:])
    elif 'np.' in valuestr:
        value = getattr(np, valuestr[3:])
    else:
        value = float(valuestr)

    return value

def readparsfile(parafilename):
    dictOpen = False
    with open(parafilename, 'r') as ipt:
        i = -1
        for line in ipt:
            i += 1
            if line.startswith('====='):
                break
            # ignore comment and space line
            if line.startswith('#') or line.isspace():
                continue
            if '#' in line:
                commentidx = line.find('#')
                line = line[:commentidx]    # remove comments
                line = line.strip()    # remove space

            if dictOpen and line[0]=='}':#dictionary will be closed
                valuetype = 7
            else:
                # load name-value pair
                if line.count('=')!=1:
                    raise Exception(parafilename + ' line ' + str(i) + ': Should only have one \'=\' sign .')
                equalsignidx = line.find('=')
                paraname = line[:equalsignidx].strip()
                valuestr = line[(equalsignidx+1):].strip()    # The value string of the line

                # get the type of the valuestring
                valuetype = mygettype(valuestr)

            #dictionary
            if valuetype==6:
                nentries = 0
                dictOpen = True

            # read list
            elif valuetype == 5:
                valuestr = valuestr.strip('[').strip(']')
                valuelist = valuestr.split(', ')
                listtype = []
                for singlevalue in valuelist:
                    listtype.append(mygettype(singlevalue))
                if len(set(listtype)) != 1:
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
                        value.append(getfloat(singlevalue))
                    else:
                        value.append(singlevalue.strip('\''))
                if listtype[0] != 3:
                    value = np.array(value)

            # evaluate the value
            # Only compatible with +, -, *, /, **. No '( )' supported.
            elif valuetype == 4:
                valuelist = valuestr.split(' ')
                # transform all the values from strring to float
                for j, singlevaluestr in enumerate(valuelist):
                    if singlevaluestr in ['+', '-', '*', '/', '**']:
                        continue
                    if singlevaluestr in paralist.keys():
                        singlevalue = paralist[singlevaluestr]
                    else:
                        singlevalue = getfloat(singlevaluestr)
                    valuelist[j] = singlevalue
                # operate all the ** in the list
                while('**' in valuelist):
                    opidx = valuelist.index('**')    # index of the operator
                    resultvalue = valuelist[opidx-1] ** valuelist[opidx+1]
                    del valuelist[opidx:(opidx+2)]
                    valuelist[opidx-1] = resultvalue
                # operate all the * or / in the list
                while('*' in valuelist or '/' in valuelist):
                    mulidx = (valuelist + ['*', '/']).index('*')
                    dividx = (valuelist + ['*', '/']).index('/')
                    opidx = min(mulidx, dividx)
                    if valuelist[opidx] == '*':
                        resultvalue = valuelist[opidx-1] * valuelist[opidx+1]
                    elif valuelist[opidx] == '/':
                        resultvalue = valuelist[opidx-1] / valuelist[opidx+1]
                    del valuelist[opidx:(opidx+2)]
                    valuelist[opidx-1] = resultvalue
                while('+' in valuelist or '-' in valuelist):
                    addidx = (valuelist + ['+', '-']).index('+')
                    subidx = (valuelist + ['+', '-']).index('-')
                    opidx = min(addidx, subidx)
                    if valuelist[opidx] == '+':
                        resultvalue = valuelist[opidx-1] + valuelist[opidx+1]
                    elif valuelist[opidx] == '-':
                        resultvalue = valuelist[opidx-1] - valuelist[opidx+1]
                    del valuelist[opidx:(opidx+2)]
                    valuelist[opidx-1] = resultvalue
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
                value = getfloat(valuestr)

            #[24.02.01]cwo: this is a string?
            else:
                value = valuestr.strip('\'')
                value = replace_environment (value)

            #assign dictionary to paralist and close the dictionary
            if valuetype==7:
                paralist[dictname] = ddum
                dictOpen = False

            elif dictOpen:
                if nentries==0:
                    dictname = paraname
                    ddum = dict()
                else:
                    ddum[paraname] = value
                nentries += 1
            else:
                # save the attr to pars class
                paralist[paraname] = value

    ## special treatment for parameters
    # special treatment for parafilename
    paralist['parafilename'] = parafilename
    return

# get the local variables and parameter files
paralist = locals()

i1 = sys.argv[0].rfind('/')
paralist['rdir'] = sys.argv[0][:i1+1]
