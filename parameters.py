import numpy as np
import constants as cnt
import sys

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
    if 'cnt.' in valuestr:
        value = getattr(cnt, valuestr[4:])
    elif 'np.' in valuestr:
        value = getattr(np, valuestr[3:])
    else:
        value = float(valuestr)

    return value

# get the local variables and parameter files
paralist = locals()
if len(sys.argv)>1:
    parafilename = sys.argv[1]
else:
    parafilename = 'parameters.txt'

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

        # load name-value pair
        if line.count('=')!=1:
            raise Exception(parafilename + ' line ' + str(i) + ': Should only have one \'=\' sign .')
        equalsignidx = line.find('=')
        paraname = line[:equalsignidx].strip()
        valuestr = line[(equalsignidx+1):].strip()    # The value string of the line

        # get the type of the valuestring
        valuetype = mygettype(valuestr)

        # read list
        if valuetype == 5:
            valuestr = valuestr.strip('[').strip(']')
            valuelist = valuestr.split(', ')
            listtype = []
            for singlevalue in valuelist:
                listtype.append(mygettype(singlevalue))
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
                    value.append(getfloat(singlevalue))
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
            value = getfloat(valuestr)
        else:
            value = valuestr.strip('\'')

        # save the attr to pars class
        paralist[paraname] = value

## special treatment for parameters
# special treatment for verbose
if paralist['verbose'] == 'silent' or paralist['verbose'] == -2:
    paralist['verboselevel'] = -2
elif paralist['verbose'] == 'quiet' or paralist['verbose'] == -1:
    paralist['verboselevel'] = -1
elif paralist['verbose'] == 'default' or paralist['verbose'] == 0:
    paralist['verboselevel'] = 0
elif paralist['verbose'] == 'verbose' or paralist['verbose'] == 1:
    paralist['verboselevel'] = 1
