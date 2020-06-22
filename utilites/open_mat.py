import scipy.io
import os

home = os.path.expanduser('~')
mat = scipy.io.loadmat(home + '/Data/Experimental/data/DataSet2.mat')
print(mat.keys())
data = mat["data"]
#print(data)
print(data[0][0])