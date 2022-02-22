from configparser import Interpolation
from turtle import width
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2



image_coordinates = pd.read_excel('Image.xlsx')            
df = pd.DataFrame(image_coordinates)

#print(df.columns)

# df.plot(x ='u', y='v', kind = 'scatter')
# plt.show()

#df['v'] = -df['v']

im = plt.imread('left000862.png')
# flip = im[::-1,:,:]
# plt.imshow(flip[:,:,:]), plt.title('flip vertical')
dimensions = im.shape
height = dimensions[0]
width = dimensions[1]
implot = plt.imshow(im, interpolation='spline16', cmap='viridis')
plt.scatter(x = df['u'].iloc[270:940], y = df['v'].iloc[270:940],s = 0.5)
plt.show()
