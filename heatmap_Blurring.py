import cv2  # Not actually necessary if you just want to create an image.
import numpy as np
import pandas as pd

image_coordinates = pd.read_excel('Image.xlsx')            
df = pd.DataFrame(image_coordinates)
df['u'] = df['u'].iloc[270:940]
df['v'] = df['v'].iloc[270:940]
df['LAeq'] = df['LAeq'].iloc[270:940]

df.dropna(subset = ['u', 'v', 'LAeq'], inplace=True)

x = df['u'].to_numpy()
y = df['v'].to_numpy()
z = df['LAeq'].to_numpy()

#minL = np.min(z)
#maxL = np.max(z)
#
#imageZ = []
#
#for i in range(0, len(z)):
#    temp = (z[i] - minL)/(maxL-minL)
#    imageZ.append(temp)
#
#points = np.array(imageZ)

img = cv2.imread('left000862.png')

dimensions = img.shape
 
# height, width, number of channels in image
height = img.shape[0]
width = img.shape[1]
channels = img.shape[2]

blank_image = np.zeros((height,width,3), np.uint8)

for i in range(0, len(x)-1):
    tempX = int(x[i])
    tempY = int(y[i])
    tempZ = int(z[i])
    blank_image[tempY, tempX] = [tempZ, tempZ, tempZ]

#dst = cv2.GaussianBlur(blank_image,(9999,9999),cv2.BORDER_DEFAULT)

cv2.imshow('image',blank_image)
cv2.waitKey(0) 
cv2.destroyAllWindows() 