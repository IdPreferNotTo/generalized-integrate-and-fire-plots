import numpy as np
import os

home = os.path.expanduser('~')
script_name = home + "/BCCN File system/cLIF.submit"
with open(script_name, "w") as file:
    file.write("universe = vanilla \n "
               "notification = Error \n "
               "Request_memory = 500  \n "
               "initialdir=/home/lukasra/CLionProjects/isi-correlations-phd/cmake-build-release \n "
               "log = /home/lukasra/condor_out/condor.log \n "
               "output = /home/lukasra/condor_out/condor.out \n "
               " error = /home/lukasra/condor_out/condor.err \n")
    for n in range(20):
        file.write("executable = /home/lukasra/CLionProjects/isi-correlations-phd/cmake-build-release/SCH13 \n")
        file.write("arguments = {:d} \n".format(n))
        file.write(" queue \n # \n")