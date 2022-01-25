from turtle import width
import pandas as pd
import matplotlib.pyplot as plt
import cv2



image_coordinates = pd.read_excel('Image.xlsx')            
df = pd.DataFrame(image_coordinates)

#print(df.columns)

df.plot(x ='u', y='v', kind = 'scatter')
plt.show()

#df['v'] = -df['v']

# im = plt.imread('left000862.png')
# dimensions = im.shape
# height = dimensions[0]
# width = dimensions[1]
# implot = plt.imshow(im)
# plt.scatter(x = df['u'], y = df['v'], s = 0.1)
# plt.show()


# img = cv2.imread('left000861.png')
# 
# dimensions = img.shape
#  
# # height, width, number of channels in image
# height = img.shape[0]
# width = img.shape[1]
# channels = img.shape[2]
#  
# print('Image Dimension    : ',dimensions)
# print('Image Height       : ',height)
# print('Image Width        : ',width)
# print('Number of Channels : ',channels)
# 
# for i in range(0, len(df)):
#     tempU = df['u'][i] + width/2
#     tempV = df['v'][i] + height/2
#     cv2.drawMarker(img, (int(tempU), int(tempV)), color=(255,0,0))
# 
# cv2.imwrite('left000861.png',img)
# 
# cv2.imshow('image',img)
# cv2.waitKey(0) 
# cv2.destroyAllWindows() 
# 