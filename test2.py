import pandas as pd
import numpy as np
import pyzed.sl as sl

pst_data = pd.read_excel('Position_Data_0_0.xlsx')            
df = pd.DataFrame(pst_data)

selected_rot_mat = df['Trans_Mat'][862].translate({ ord(c): None for c in "[]" }).split(',')
# 4X4 Transformation Matrix
selected = np.array([[float(selected_rot_mat[0]),float(selected_rot_mat[1]),float(selected_rot_mat[2]),float(selected_rot_mat[3])],
                        [float(selected_rot_mat[4]),float(selected_rot_mat[5]),float(selected_rot_mat[6]),float(selected_rot_mat[7])],
                        [float(selected_rot_mat[8]), float(selected_rot_mat[9]), float(selected_rot_mat[10]), float(selected_rot_mat[11])],
                        [float(selected_rot_mat[12]),float(selected_rot_mat[13]),float(selected_rot_mat[14]), float(selected_rot_mat[15])]])
#3X3 Rotation Matrix
#selected = np.array([[selected_rot_mat[0],selected_rot_mat[1],selected_rot_mat[2]],
#                       [selected_rot_mat[4],selected_rot_mat[5],selected_rot_mat[6]],
#                       [selected_rot_mat[8],selected_rot_mat[9],selected_rot_mat[10]]], np.float)

#Transform_pose from ZED SDK tutorial
# def transform_pose(pose, tx, ty, tz) :
#     transform_ = sl.Transform()
#     transform_.set_identity()
#     # Translate the tracking frame by tx along the X axis
#     transform_[0,3] = tx
#     transform_[1,3] = ty
#     transform_[2,3] = tz
#     # Pose(new reference frame) = M.inverse() * pose (camera frame) * M, where M is the transform between the two frames
#     transform_inv = sl.Transform()
#     transform_inv.init_matrix(transform_)
#     transform_inv.inverse()
#     pose = transform_inv * pose * transform_
# 
# transform_ = np.array([[float(1), float(0), float(0), float(selected_rot_mat[3])],
#                      [float(0), float(1), float(0), float(selected_rot_mat[7])],
#                      [float(0), float(0), float(1), float(selected_rot_mat[11])],
#                      [float(0), float(0), float(0), float(1)]])
                    
transformed_x = []
transformed_y = []
transformed_z = []
selected_inv = np.linalg.inv(selected)

for i in range(0,len(df)):
    temp = df['Trans_Mat'][i].translate({ ord(c): None for c in "[]" }).split(',')
    pose = np.array([[float(temp[0]), float(temp[1]), float(temp[2]), float(temp[3])],
                     [float(temp[4]), float(temp[5]), float(temp[6]), float(temp[7])],
                     [float(temp[8]), float(temp[9]), float(temp[10]), float(temp[11])],
                     [float(temp[12]), float(temp[13]), float(temp[14]), float(temp[15])]])
    transformed = pose*selected_inv
    transformed_x = np.append(transformed_x, transformed[0][3])
    transformed_y = np.append(transformed_y, transformed[1][3])
    transformed_z = np.append(transformed_z, transformed[2][3])

df['transformed_x'] = transformed_x
df['transformed_y'] = transformed_y
df['transformed_z'] = transformed_z

df.to_excel('Position_Data_0_0_transformed.xlsx')
