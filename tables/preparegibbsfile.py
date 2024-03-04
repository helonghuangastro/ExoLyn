import numpy as np
import os

def read_last_line(file_name):
    with open(file_name, 'rb') as file:
        file.seek(-2, 2)
        while file.read(1) != b'\n':
            file.seek(-2, 1)
        last_line = file.readline().decode('utf-8')
    return last_line

files = os.listdir('janaf_tables')

# find the maximum temperature
maxT = 0
for file in files:
    lastline = read_last_line('janaf_tables/' + file)
    T = float(lastline.split('\t')[0])
    maxT = np.maximum(T, maxT)

# everyline to write in the file
Tarr = np.arange(100, maxT+1, 100)
lines = [str(T)+' ' for T in Tarr]
title = 'Tref '
credit = '# Gibbs formation energt at 1bar. The unit is kJ/mol. From janaf.nist.gov. ZnS from Robie & Hemingway (1995)'

# iterate the files to read gibbs free energy to the tables
for file in files:
    with open('janaf_tables/'+file, 'r') as ipt:
        # add to the first line
        if file.endswith('s.txt'):
            title += file.strip('s.txt') + '(s) '
        else:
            title += file.strip('.txt') + ' '
        iline = 0
        while True:
            lineinjanaf = ipt.readline()
            if not lineinjanaf:
                break

            # judge whether the line starts with numbers and whether the temperature is the **00
            if not lineinjanaf[0].isdigit():
                continue
            janafdata = lineinjanaf.split()
            if float(janafdata[0])==0. or float(janafdata[0])%100!=0:
                continue

            while float(janafdata[0])/100 - iline > 1:
                lines[iline] += '-- '.ljust(10, ' ')
                iline += 1

            # read the file and fill in the data
            # if a line is missing
            if len(janafdata) == 1:
                lines[iline] += '-- '.ljust(10, ' ')
            # normal data
            else:
                lines[iline] += janafdata[6].ljust(9, ' ')
                lines[iline] += ' '
            iline = iline+1
        
        # add a label to occupy space
        for i in range(iline, len(Tarr)):
            lines[i] += '-- '

# add \n to the end of the strings
for i, line in enumerate(lines):
    lines[i] = line + '\n'

# write the data to one file
with open('gibbs_test.txt', 'w') as opt:
    opt.write(title + '\n')
    opt.write(credit + '\n')
    for line in lines:
        opt.write(line)
