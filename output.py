import numpy as np
import pdb
import parameters as pars

def writeatm(y, grid, filename='grid.txt', additional='', fmt1='{:12.4e}'):
    """
    chris: what is the meaning of "additional"?
    """

    print("[output]:writing output to filen: "+filename)
    with open(filename, 'w') as opt:
        # write additional
        opt.write(additional)

        # write the header
        header = '#cols::logP'
        for solidname in pars.solid:
            header += ','
            header += solidname + '(s)'
        for gasname in pars.gas:
            header += ','
            header += gasname
        header += ',nuclei\n'
        opt.write(header)


        # write each row (different pressure layer)
        Nc, Ngrid = y.shape
        fmt2 = (Nc+1)*fmt1+'\n'
        for i in range(len(grid)):
            line = fmt2.format(*((grid[i],)+tuple(y[:,i])))
            opt.write(line)
