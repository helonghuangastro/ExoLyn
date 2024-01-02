import parameters as pars


def prepare_optical (optooldir, wavelengthgrid):
    """
    does some checks to see w/r we can perform optical constant calculations...
    """

    import pdb; pdb.set_trace()

    #check w/r optool is there...
    if 'calcoptical' in paralist and calcoptical:
        if 'optooldir' not in paralist:
            print('[parameters.py]:No >> optooldir << provided')
            calcoptical = False
        else:
            try:
                flist = os.listdir(optooldir)
            except:
                print(f'[parameters.py]:No valid dir for >> optooldir << ({optooldir}) provided')
                calcoptical = False

        if calcoptical:
            if 'optool' not in flist or 'optool.py' not in flist:
                print(f'[parameters.py]:optool tools not in {optooldir}')
                calcoptical = False

