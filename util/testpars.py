import os
import time
import numpy as np
import shutil
import subprocess as sp
# import signal
def writepars():
    with open('parameters.txt', 'w') as opt:
        rn = np.random.rand()
        opt.write('T_star = ' + str(T_star_range[0]*(1-rn)+T_star_range[1]*rn) + '\n')
        opt.write('R_star = cnt.R_sun\n')
        opt.write('\n')

        opt.write('Rp = 1.087 * cnt.Rj\n')
        opt.write('rp = 0.05 * cnt.au\n')    # TBD: This can be changed
        rn = np.random.rand()    # rn stands for random number
        opt.write('g = ' + str(g_range[0]**(1-rn)*g_range[1]**rn) + '\n')
        opt.write('\n')

        opt.write('runname = \'\'\n')
        opt.write('rootdir = \'../\'\n')
        opt.write('gibbsfile = rootdir + \'tables/gibbs_test.txt\'\n')
        opt.write('suppresswarnings = \'all\'')
        opt.write('\n')

        opt.write('verbose = \'silent\'\n')
        opt.write('plotmode = \'none\'\n')
        opt.write('writeoutputfile = False\n')
        opt.write('\n')

        opt.write('N = 100\n')
        opt.write('Pa = 1.\n')
        opt.write('Pb = 1e7\n')
        opt.write('Pref = 1e6\n')
        opt.write('autoboundary = True\n')
        opt.write('\n')

        opt.write('opa_IR = 0.3\n')
        opt.write('firr = 1 / 4\n')
        opt.write('opa_vis_IR = 0.158\n')
        rn = np.random.rand()
        opt.write('T_int = ' + str(T_int_range[0]**(1-rn)*T_int_range[1]**rn) + '\n')
        opt.write('\n')

        opt.write('mgas = 2.34 * cnt.mu\n')
        rn = np.random.rand()
        opt.write('Kzz = ' + str(Kzz_range[0]**(1-rn)*Kzz_range[1]**rn) + '\n')
        opt.write('Kp = ' + str(Kzz_range[0]**(1-rn)*Kzz_range[1]**rn) + '\n')
        opt.write('cs_mol = 2e-15\n')
        opt.write('\n')

        opt.write('f_stick = 1.\n')
        opt.write('f_coag = 1.\n')
        opt.write('cs_com = 8e-15\n')
        opt.write('rho_int = 2.8\n')
        opt.write('an = 1e-7\n')    # TBD: This can also be discussed
        opt.write('mn0 = 4 / 3 * np.pi * rho_int * an ** 3\n')
        rn = np.random.rand()
        opt.write('nuc_pro = ' + str(nuc_pro_range[0]**(1-rn)*nuc_pro_range[1]**rn) + '\n')
        rn = np.random.rand()
        opt.write('sigma_nuc = ' + str(sigma_nuc_range[0]*(1-rn)+sigma_nuc_range[1]*rn) + '\n')
        rn = np.random.rand()
        opt.write('P_star = ' + str(P_star_range[0]**(1-rn)*P_star_range[1]**rn) + '\n')
        opt.write('\n')

        opt.write('gas = ' + str(gas))
        opt.write('\n')
        rnarr = np.random.rand(len(gas))
        xvb = []
        for i in range(len(gas)):
            xvb.append(xvb0[i]*10**(2*rnarr[i]-1))
        opt.write('xvb = ' + str(xvb))
        opt.write('\n')

        opt.write('calcoptical = False')
        opt.write('\n')

        opt.write('============ put reactions here =============\n')
        opt.write('Mg + SiO + 2H2O -> MgSiO3(s) + 2H2\n')
        opt.write('2Mg + SiO + 3H2O -> Mg2SiO4(s) + 3H2\n')
        opt.write('SiO + H2O -> SiO2(s) + H2\n')
        opt.write('Mg + H2O -> MgO(s) + H2\n')
        opt.write('Fe + H2O -> FeO(s) + H2\n')
        opt.write('Fe + H2S -> FeS(s) + H2\n')
        opt.write('2Fe + 3H2O -> Fe2O3(s) + 3H2\n')
        opt.write('Fe -> Fe(s)\n')
        opt.write('TiO + H2O -> TiO2(s) + H2\n')
        opt.write('2Al + 3H2O -> Al2O3(s) + 3H2\n')
        opt.write('\n')

    return

solid = ['MgSiO3', 'Mg2SiO4', 'SiO2', 'MgO', 'FeO', 'Fe2O3', 'FeS', 'Fe', 'TiO2', 'Al2O3']
gas = ['Mg', 'SiO', 'H2O', 'Fe', 'H2S', 'TiO', 'Al']
xvb0 = [4.1e-4, 6.1e-4, 1.4e-3, 7.6e-4, 1.9e-4, 2.4e-6, 3.2e-5]

g_range = [250, 25000]
T_int_range = [300, 1000]
Kzz_range = [1e6, 1e10]
nuc_pro_range = [1e-19, 1e-11]
sigma_nuc_range = [0.1, 0.5]
P_star_range = [6, 600]
T_star_range = [3000, 8000]

# define a signal handler so that I can limit the time of running
# def signal_handler(signum, frame):
#     raise Exception("Time out!")

failcount = 0
totaltime = 0
succtime = 0
N = 100
for i in range(N):
    print(i)

    writepars()

    # Limit the time of running
    # signal.signal(signal.SIGALRM, signal_handler)
    # signal.alarm(120)

    starttime = time.time()
    #isfail = os.system("python3 ../relaxation.py ./parameters.txt")
    out = sp.run(['python3', '../src/relaxation.py', './parameters.txt'])
    isfail = out.returncode

    endtime = time.time()
    totaltime += endtime - starttime
    if isfail == 0:
        succtime += endtime - starttime
    # try:
    #     starttime = time.time()
    #     # Running the cloud model
    #     isfail = os.system("python3 relaxation.py")
    #     endtime = time.time()
    #     succtime += endtime - starttime
    # except Exception:
    #     failcount += 1
    #     isfail = -100
    #     print('ABORTION: Exceed maximum running time 2 min.')

    # Save the failed parameters
    if isfail !=0 :
        failcount += 1
        shutil.move('parameters.txt', 'failpars/parameters' + str(i) + '.txt')

print('Failed cases: ' + str(failcount))
print('Total time: ' + str(totaltime) + ' s')
print('Average successful time: ' + str(succtime/(N-failcount)) + ' s')

os.remove('parameters.txt')
