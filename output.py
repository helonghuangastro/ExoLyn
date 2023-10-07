import numpy as np
import pdb
import parameters as pars

def writeatm(y, grid, filename='grid.txt', additional=''):
    with open(filename, 'w') as opt:
        # write additional
        opt.write(additional)

        # write the header
        header = 'logP'
        for solidname in pars.solid:
            header += ' '
            header += solidname + '(s)'
        for gasname in pars.gas:
            header += ' '
            header += gasname
        header += ' nuclei\n'
        opt.write(header)

        # write each row (different pressure layer)
        for i in range(len(grid)):
            line = str(grid[i])
            for concentration in y[:, i]:
                line += ' '
                line += str(concentration)
            line += '\n'
            opt.write(line)
