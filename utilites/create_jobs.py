import numpy as np
import os, shutil

def clear_folder(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

def write_script(i):
    home = os.path.expanduser('~')
    script_name = home +"/HU File system/Scripts/cLIF_{:03d}.sh".format(i)
    with open(script_name, "w") as file:
        file.write("#!/bin/bash \n"
                   "#PBS -N cLIF_{0:d} \n"
                   "cd $PBS_O_WORKDIR \n"
                   "#SBATCH -o /users/nph/lukas/Report/output.out # STDOUT \n"
                   "/users/nph/lukas/CLionProjects/isi-correlations-phd/cmake-build-release/SCH13 {0:d}".format(i))

if __name__ == "__main__":
    home = os.path.expanduser('~')
    folder = home + "/HU File system/Scripts"
    clear_folder(folder)
    for i in range(40):
        write_script(i)
