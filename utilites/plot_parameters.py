def adjust_plotsize(size, ratio=0.75):
    #points = 246.
    points = 380.
    width = points*size
    height = points*size*ratio
    return (width/72.27, height/72.27)

def set_latex_font():
    from matplotlib import rc
    from matplotlib import rcParams
    #rcParams['font.family'] = 'serif'
    #rcParams['font.serif'] = 'Palatino'
    rc('text', usetex=True)

fontsize = 10
labelsize = 9