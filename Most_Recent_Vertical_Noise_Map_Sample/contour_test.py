from turtle import width
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import pandas as pd

image_coordinates = pd.read_excel('Image.xlsx')            
df = pd.DataFrame(image_coordinates)

npts = len(df)
im = plt.imread('left000862.png')
dimensions = im.shape
ngridy = dimensions[0]
ngridx = dimensions[1]
df['u'] = df['u'].iloc[270:940]
#df['u'] = df['u'] - 0.5*ngridx
x = df['u'].to_numpy()
x = x[~np.isnan(x)]
df['v'] = df['v'].iloc[270:940]
#df['v'] = df['v'] + 0.5*ngridy
y = df['v'].to_numpy()
y = y[~np.isnan(y)]
df['LAeq'] = df['LAeq'].iloc[270:940]
z = df['LAeq'].to_numpy()
z = z[~np.isnan(z)]

#X, Y = np.meshgrid(x, y)

#import scipy.interpolate
#rbf = scipy.interpolate.Rbf(x, y, z, function='cubic')
#Z=rbf(X,Y)
#
#lmin=y.min()
#lmax=y.max()
##progn=(lmax-lmin)/20
##limit=np.arange(lmin,lmax,progn)
#
#print(len(X),len(Y))


fig, ax = plt.subplots(figsize=(6,2))
#ax.contour(X,Y,Z) 
print(z.min(), z.max(), int(z.min()), int(z.max()))

#levels = np.arange(int(z.min()), int(z.max())+1, 0.1)
#print(levels)

#v = np.linspace(60, 80, 10, endpoint=True)
#ax.tricontour(x,y,z, levels =[72.5,75,77.5,80,82.5,85], cmap = "turbo") 
ax.scatter(x = x, y = y,s = 0.5)
ct = ax.tricontourf(x,y,z, levels =[73,74,75,76,77,78,79,80,81],alpha = 0.5, cmap = "jet")
fig.colorbar(ct)
ax.set_title('Niose Map')
ax.imshow(im, interpolation='spline16', cmap='viridis')
plt.show()

#fig, (ax2) = plt.subplots(nrows=1)
#
## -----------------------
## Interpolation on a grid
## -----------------------
## A contour plot of irregularly spaced data coordinates
## via interpolation on a grid.
#
#
#ax2.tricontour(ngridx, ngridy, z, levels=14, linewidths=0.5, colors='k')
#cntr2 = ax2.tricontourf(ngridx, ngridy, z, levels=14, cmap="RdBu_r")
#
#fig.colorbar(cntr2, ax=ax2)
#ax2.plot(x, y, 'ko', ms=3)
##ax1.set(xlim=(-2, 2), ylim=(-2, 2))
##ax1.set_title('grid and contour (%d points, %d grid points)' %
##             (npts, ngridx * ngridy))
#
#plt.show()