import os

def prepare_optical (optooldir=None, wavelengthgrid=None, dirmeff=None, dirkappa=None):
    """
    does some checks to see w/r we can perform optical constant calculations...
    """

    calcoptical = True

    #check w/r optool is there...
    if 'optooldir' is None:
        print('[optical.py]ERROR:No >> optooldir << provided')
        calcoptical = False
    else:
        try:
            flist = os.listdir(optooldir)
        except:
            print(f'[optical.py]ERROR:No valid dir for >> optooldir << ({optooldir}) provided')
            calcoptical = False

    #search for optool and optool.py
    if calcoptical:
        if 'optool' not in flist or 'optool.py' not in flist:
            print(f'[optical.py]ERROR:"optool" and/or "optool.py" not in {optooldir}')
            calcoptical = False

    #output entries for effective medium
    if calcoptical:
        if dirmeff is None:
            dirmeff = './meff' #the default dir by default

        if not os.path.exists(dirmeff):
            try:
                os.mkdir(dirmeff)
            except:
                print(f'[optical.py]ERROR:creating dir >> dirmeff << ({dirmeff})')
                calcoptical = False

    if calcoptical:
        if dirkappa is None:
            dirkappa = './meff' #the default dir by default

        if not os.path.exists(dirkappa):
            try:
                os.mkdir(dirkappa)
            except:
                print(f'[optical.py]ERROR:creating dir >> dirkappa << ({dirkappa})')
                calcoptical = False


    if type(wavelengthgrid)==str:
        #read the wavelength grid from file TBD!
        pass


    #perhaps not the most elegant solution
    doptical = {'optooldir':optooldir, 'wavelengthgrid':wavelengthgrid, 'dirmeff':dirmeff}

    return calcoptical, doptical
