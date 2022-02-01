from calendar import c
import numpy as np
import pandas as pd
import cv2

db_data = pd.read_excel('Overall_Data.xlsx')            
df = pd.DataFrame(db_data)

print(len(df))
cam_paras = df['Camera_Matrix'][0].translate({ ord(c): None for c in "[]" }).split(',')
f_x = int(float(cam_paras[0]))
c_x = int(float(cam_paras[2]))
f_y = int(float(cam_paras[4]))
c_y = int(float(cam_paras[5]))
# cameraMatrix = np.array([[f_x,0,c_x],[0,f_y,c_y],[0,0,1]], np.float)
# rVec = np.array(df['Rotation_Vect'][373].translate({ ord(c): None for c in "[]" }).split(','), np.float)
# tVec = np.array(df['Translation'][373].translate({ ord(c): None for c in "[]" }).split(','), np.float)
# loc = df[['loc_x', 'loc_y', 'loc_z']].to_numpy()
# disto = np.array([[0, 0, 0, 0, 0]], np.float)

#image_cod = cv2.projectPoints(loc, rVec, tVec, cameraMatrix, disto)

#res = list(image_cod[0])

final_res = pd.DataFrame(index = range(0, len(df)), columns=['u', 'v', 'LAeq'])

# print(len(res))
# 
# for i in range(0, len(res)):
#     final_res['u'][i] = res[i][0][0]
#     final_res['v'][i] = res[i][0][1]
# 
# final_res['LAeq'] = db_data['LAeq']
# 
# 
# print(new_origin_x, new_origin_y)

#df['loc_x'] = -df['loc_x'][862] + df['loc_x']
#df['loc_y'] = -df['loc_y'][862] + df['loc_y']
#df['loc_z'] = -df['loc_z'][862] + df['loc_z']
#
#print(df['loc_x'][862], df['loc_y'][862], df['loc_z'][862])


final_res['u'] = (df['loc_x']/df['loc_z'])*f_x + c_x
final_res['v'] = (df['loc_y']/df['loc_z'])*f_y + c_y
final_res['LAeq'] = df['LAeq']

print(final_res.columns)

final_res.to_excel('Image.xlsx')
