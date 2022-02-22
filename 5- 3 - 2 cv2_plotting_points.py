import pandas as pd
import cv2

image_coordinates = pd.read_excel('Image.xlsx')            
df = pd.DataFrame(image_coordinates)

img = cv2.imread('left000862.png')

dimensions = img.shape
 
# height, width, number of channels in image
height = img.shape[0]
width = img.shape[1]
channels = img.shape[2]
 
print('Image Dimension    : ',dimensions)
print('Image Height       : ',height)
print('Image Width        : ',width)
print('Number of Channels : ',channels)

for i in range(0, len(df)):
    tempU = df['u'][i]
    tempV = df['v'][i]
    cv2.drawMarker(img, (int(tempU), int(tempV)), color=(255,0,0))

#cv2.imwrite('left000862.png',img)

cv2.imshow('image',img)
cv2.waitKey(0) 
cv2.destroyAllWindows() 
