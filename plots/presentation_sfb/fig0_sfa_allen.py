import numpy as np
import matplotlib.pyplot as plt
from pynwb import NWBFile, TimeSeries, NWBHDF5IO
import os

if __name__ == "__main__":
    home = os.path.expanduser("~")
    file_str = home + "/Desktop/human_middle_frontal_gyrus_adap.nwb"

    raw_io = NWBHDF5IO(file_str, 'r')
    nwb_in = raw_io.read()

    nwb_proc = nwb_in.copy()


