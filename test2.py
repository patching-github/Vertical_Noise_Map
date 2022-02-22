import numpy as np
import matplotlib.pyplot as plt

x = np.random.rayleigh(50, size=5000)
y = np.random.rayleigh(50, size=5000)


a = np.array([[ 5, 1 ,3, 1], 
                [ 1, 1 ,1, 1], 
                [ 1, 2 ,1, 1]])

b = b = np.array([1, 2, 3, 1])
print(a.dot(b))


#plt.hist2d(x,y, bins=[np.arange(0,400,5),np.arange(0,400,5)])

#plt.show()