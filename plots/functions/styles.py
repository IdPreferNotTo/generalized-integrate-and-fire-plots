from matplotlib import rc
from matplotlib import rcParams
import warnings

colors = ["#173F5F", "#20639B", "#3CAEA3", "#F6D55C", "#BF6B63", "#D9A384"]

def color_palette():
    return ["#173F5F", "#20639B", "#3CAEA3", "#F6D55C", "#BF6B63", "#D9A384", "#ED553B"]




def figsize(shape, size):
    """
    :param shape: List with number of columns and rows. I.e. [3, 2] would create a plot with 3 columns and 2 rows
    :param size: String that can be "small" or "large"
    :return: width, height the resulting plot size
    """
    if size == "large":
        height = 200
        width = 300
    elif size == "small":
        height = 150
        width = 200
    else:
        return warnings.warn("No valid plot size specified. Options are \"small\" and \"large\". ")
    x = shape[0]
    y = shape[1]
    height = y*height
    width = x*width
    return width, height


def set_default_plot_style():
    rcParams['text.latex.preamble'] = r'\usepackage{lmodern}'
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = 'Latin Modern'
    rcParams.update({'font.size': 11})
    rc('text', usetex=True)


def remove_top_right_axis(axis):
    for ax in axis:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)


def remove_all_axis(axis):
    for ax in axis:
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)