import numpy as np
import matplotlib.pyplot as plt

x = np.random.rayleigh(50, size=5000)
y = np.random.rayleigh(50, size=5000)


plt.hist2d(x,y, bins=[np.arange(0,400,5),np.arange(0,400,5)])

plt.show()