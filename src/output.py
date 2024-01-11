import numpy as np
import pdb
import parameters as pars

def writeatm(atmosphere, comment='', fmt1='{:12.4e}'):
    """
    chris: what is the meaning of "additional"?
    Answer: any additional information the user wants to write
    """

    if 'savedir' not in pars.__dict__:
        savedir = './'
    else:
        savedir = pars.savedir

    filename = 'grid' + pars.runname + '.txt'
    print("[output]:writing output to file: "+filename)
    with open(savedir + filename, 'w') as opt:
        # write additional comments
        opt.write('# ' + comment + '\n')

        # write necesssary parameters
        parsstring = '# '
        # choose parameters to save    # Maybe should make this into pars
        parstosave = ['runname', 'g', 'mgas', 'rho_int', 'an', 'T_star', 'R_star', 'Rp', 'rp', 'gibbsfile']
        for paraname in parstosave:
            paravalue = getattr(pars, paraname)
            paratype = type(paravalue)
            if paratype == int or paratype == float:
                paravalue = '{:.4e}'.format(paravalue)
            parsstring += paraname + ' = ' + paravalue + '; '
        parsstring += '\n'
        opt.write(parsstring)

        # write reactions
        reactions = atmosphere.chem.reactions
        reactionstring = '# '
        with open(pars.parafilename, 'r') as ipt:
            parameterline = True
            for line in ipt:
                if line.startswith('====='):
                    parameterline = False
                    continue
                if parameterline:
                    continue
                line = line.strip()
                reactionstring += line + '; '
        reactionstring = reactionstring[:-2]
        reactionstring += '\n'
        opt.write(reactionstring)

        # write the header
        header = '#cols::logP T(K) rhop(gcm-3) ap(cm)'
        for solidname in pars.solid:
            header += ' '
            header += solidname + '(s)'
        for gasname in pars.gas:
            header += ' '
            header += gasname
        header += ' nuclei\n'
        opt.write(header)


        y = atmosphere.y
        grid = atmosphere.grid
        T = atmosphere.cachegrid.T_grid
        rhop = atmosphere.rho
        ap = atmosphere.ap

        # write each row (different pressure layer)
        Nc, Ngrid = y.shape
        fmt2 = (Nc+4)*fmt1+'\n'
        for i in range(len(grid)):
            line = fmt2.format(*((grid[i], T[i], rhop[i], ap[i])+tuple(y[:,i])))
            opt.write(line)
