import numpy as np

arr = np.fromfile("input.tensor", sep='\n').reshape((1,240,320,1))
print(arr)