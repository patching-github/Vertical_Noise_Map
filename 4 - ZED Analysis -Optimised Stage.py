#Analyze the camera data from Zed Minis

########################################################################
#
# Copyright (c) 2017, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

#This has been updated for SDK 3.0 from stereolabs.

"""
    Read SVO sample to read the video and the information of the camera. It can pick a frame of the svo and save it as
    a JPEG or PNG file. Depth map and Point Cloud can also be saved into files.


    Future Changes:
    2) Add in textureness_confidence_threshold -> similar to other confidence a lower value means pulling only the higher confidence.
        - This may not be in the python API yet (no documentatoin on it)

    
"""
from timeit import default_timer as timer
import sys
import pyzed.sl as sl
import numpy as np
import pandas as pd
import threading


def grab_run_segment(step, spatial):
    global thread_list
    global stop_signal
    global cam_list
    global filepath
    global save_path
    global area_path
    global load_mesh
    global out_file
    global sheet_name
    global runtime
##    global spatial
    global pymesh
    global filter_level
    global mesh_tex
    global pymesh
    global save_texture
    global tracking
    global area_path

    print(save_texture)
    
#Put this in the camera specific thread Local Variables
    camera_pose = sl.Pose()
    py_translation = sl.Translation()
    rx, ry, rz = [[], []], [[], []], [[], []]
    tx, ty, tz = [[], []], [[], []], [[], []]
    time_val = [[], []]
    rot = [[],[]]
    trans = [[],[]]
    df = [[],[]]
    trans_mat = [[],[]] #This is to add a transform matrix for each location
    rotation_vect = [[],[]]
    translation = [[],[]]
    segment = 0
    

    mapping_state = [[],[]]
    pos_state = [[],[]]

    print("Processing ")

    if len(cam_list) == 1:
        num_frames = cam_list[0].get_svo_number_of_frames()
        calibration_params = cam_list[0].get_camera_information().calibration_parameters
        fX = calibration_params.left_cam.fx
        fY = calibration_params.left_cam.fy
        cX = calibration_params.left_cam.cx
        cY = calibration_params.left_cam.cy
        camMat = np.array([[fX,0,cX],[0,fY,cY],[0,0,1]])

    elif len(cam_list) == 2:
        num_frames = min(cam_list[0].get_svo_number_of_frames(), cam_list[1].get_svo_number_of_frames())
    print('Number of Frames: ', num_frames)

    if len(cam_list) == 1:
        num_frames = cam_list[0].get_svo_number_of_frames()
    elif len(cam_list) == 2:
        num_frames = min(cam_list[0].get_svo_number_of_frames(), cam_list[1].get_svo_number_of_frames())
    print('Number of Frames: ', num_frames)
    # for f_num in range(0, 5):
    for f_num in range(num_frames):
        print(f'Frame Number: {f_num}/{num_frames}')
        for index in range(0, len(cam_list)):
            cam_list[index].grab(runtime)
            mapping_state[index] = np.append(mapping_state[index], cam_list[index].get_spatial_mapping_state())
            save_path[index] = 'Zed_'+str(index)+ '_'+ str(step)+ '_'+str(segment)+'_Mesh'
            
            #Catch Pose Data - Camera 1
            tracking_state = cam_list[index].get_position(camera_pose, sl.REFERENCE_FRAME.WORLD)
            pos_state[index] = np.append(pos_state[index], tracking_state)

            # print("Example of Pose Data")
            # print("Camera Pose") #This should be the transform matrix, which includes rotation matrix and translation to get to this position.
            # print(camera_pose.pose_data().m)
            # print("Orientation") #I think these are the euler angles of the orientation of the camera
            # print(camera_pose.get_orientation().get())
            # print("Rotation vector") #This is something to do with Rodrigues formula but I don't know what it is.
            # print(camera_pose.get_rotation_vector())
            # print("Rotation Matrix") #This is the rotation portion of the matrix from Camera Pose
            # print(camera_pose.get_rotation_matrix())
            # print("Translation") #This is the translation portion of the matrix from Camera Pose
            # print(camera_pose.get_translation().get())

            if tracking_state != sl.POSITIONAL_TRACKING_STATE.FPS_TOO_LOW: # == sl.POSITIONAL_TRACKING_STATE.OK: #Tracking state returns .OK only if a new pose is available.
                rot[index] = camera_pose.get_rotation_vector() #This is the rodrigues Rotation Vector, Orientation is by get_orientation
                rx[index] = np.append(rx[index], rot[index][0])
                ry[index] = np.append(ry[index], rot[index][1])
                rz[index] = np.append(rz[index], rot[index][2])

                trans[index] = camera_pose.get_translation(py_translation).get()
                tx[index] = np.append(tx[index], trans[index][0])
                ty[index] = np.append(ty[index], trans[index][1])
                tz[index] = np.append(tz[index], trans[index][2])

##                time_val2 = sl.Timestamp.get_nanoseconds(camera_pose.timestamp)
##                time_val[index] = np.append(time_val[index], sl.Timestamp.get_nanoseconds(cam_list[index].get_timestamp(sl.TIME_REFERENCE.IMAGE)))
                time_val[index] = np.append(time_val[index], sl.Timestamp.get_nanoseconds(camera_pose.timestamp))
                trans_mat[index].append(np.asmatrix(camera_pose.pose_data().m).tolist()) #trans_mat[index].append(camera_pose.pose_data().m)

                rotation_vect[index].append(np.asmatrix(camera_pose.get_rotation_vector()).tolist())
                translation[index].append(np.asmatrix(camera_pose.get_translation().get()).tolist())

                # print(repr(camera_pose.pose_data().m))
##                print(f'Frame Number: {i}')
##                print(f'Time Value: {time_val[index][i]}')
##                print(f'Pose Time Value: {time_val2}')
            else: #Append error values
                rx[index] = np.append(rx[index], 999)
                ry[index] = np.append(ry[index], 999)
                rz[index] = np.append(rz[index], 999)

                tx[index] = np.append(tx[index], 999)
                ty[index] = np.append(ty[index], 999)
                tz[index] = np.append(tz[index], 999)

                time_val[index] = np.append(time_val[index], sl.Timestamp.get_nanoseconds(camera_pose.timestamp))
                trans_mat[index].append(np.asmatrix(camera_pose.pose_data().m).tolist()) #trans_mat[index].append(camera_pose.pose_data().m)

            if f_num > 0 and f_num % 1000 == 0: #Every 1,000 frames save the mesh
                print(f'In Mod 1000, frame number: {f_num}')
                print(f'Save Texture: {save_texture}')
                cam_list[index].extract_whole_spatial_map(pymesh[index])
                #Disable Camera
                cam_list[index].save_area_map(area_path[index]) #Explicitly saving area map
##                while cam_list[index].get_area_export_state() != sl.ERROR_CODE.SUCCESS:
##                    print(f'Saving Area File Response: {cam_list[index].get_area_export_state()}')
####                    print(cam_list[index].get_area_export_state)
##                print(f'Area Saved Path: {area_path[index]}')
                
##                cam_list[index].disable_positional_tracking(area_path[index])
##                cam_list[index].disable_positional_tracking()
                cam_list[index].disable_spatial_mapping()
                if filter_level:
                    #Filter and Save Pymesh Cam 1
                    print('Filtering Cam ' + str(index))
                    filter_params = sl.MeshFilterParameters()
                    filter_params.set(filter_level)#Was Medium
                    print("Filtering params : {0}.".format(pymesh[index].filter(filter_params)))
                elif filter_level == False:
                    print("Not Filtered")

                if save_texture:
                    apply_texture = pymesh[index].apply_texture(mesh_tex) #Was RGBA
                    print("Applying texture : {0}.".format(apply_texture))
                elif save_texture == False:
                    print("No Texture Applied")
                    
                save_mesh(pymesh[index], save_path[index])
                print(f'Mesh Saved')
##                pymesh[index].clear() #= sl.Mesh() #Reset pymesh
                print(f'Pymesh NOT Reset')

                cam_list[index].enable_spatial_mapping(spatial)

                segment += 1
                save_path[index] = 'Zed_'+str(index)+ '_'+ str(step)+ '_'+str(segment)+'_Mesh'
                print(f'Segment Number: {segment}')

    
    col_camMat = pd.DataFrame(index=np.arange(num_frames), columns=["Camera_Matrix"])
    col_camMat['Camera_Matrix'][0] = np.asmatrix(camMat).tolist()

    #Output the position data of the microphone Camera 1
    for index in range (0, len(cam_list)):
        df[index] = pd.DataFrame()
        df[index]['Time_val'] = time_val[index] #This is measured in nano seconds since last epoch, For some reason the initial two have messed up time stamps, so taking only from 2 onward.
        df[index]['Time'] = (time_val[index]-time_val[index][0])/1000000000 #Subtracting the first measurement and dividing by 10^9 to get it into seconds.
        df[index]['loc_x'] = tx[index]
        df[index]['loc_y'] = ty[index]
        df[index]['loc_z'] = tz[index]
        df[index]['rot_x'] = rx[index]
        df[index]['rot_y'] = ry[index]
        df[index]['rot_z'] = rz[index]
        df[index]['map_state'] = mapping_state[index]
        df[index]['pos_state'] = pos_state[index]
        df[index]['Trans_Mat'] = trans_mat[index]
        df[index]['Camera_Matrix'] = col_camMat
        df[index]['Rotation_Vect'] = rotation_vect[index]
        df[index]['Translation'] = translation[index]
        df[index].to_excel(out_file[index], sheet_name)

        cam_list[index].extract_whole_spatial_map(pymesh[index])

        #Disable Camera
        cam_list[index].disable_positional_tracking(area_path[index])
        cam_list[index].disable_spatial_mapping()

        if filter_level:
            #Filter and Save Pymesh Cam 1
            print('Filtering Cam ' + str(index))
            filter_params = sl.MeshFilterParameters()
            filter_params.set(filter_level)#Was Medium
            print("Filtering params : {0}.".format(pymesh[index].filter(filter_params)))
        elif filter_level == False:
            print("Not Filtered")

        if save_texture:
            apply_texture = pymesh[index].apply_texture(mesh_tex) #Was RGBA
            print("Applying texture : {0}.".format(apply_texture))
        elif save_texture == False:
            print("No Texture Applied")
            
        save_mesh(pymesh[index], save_path[index])
        cam_list[index].close()
        print("\nFINISHED CAM "+ str(index))


def main():
    start = timer()
    global thread_list
    global stop_signal
    global cam_list
    global filepath
    global save_path
    global area_path
    global load_mesh
    global out_file
    global sheet_name
    global runtime
    global spatial
    global pymesh
    global filter_level
    global mesh_tex
    global pymesh
    global save_texture
    global tracking
    global area_path

    for step in range(0,1):
        stop_signal = False
        cam_list = []
        filepath = []
        save_path = []
        area_path = []
        load_mesh = []
        out_file = []
        thread_list = []
        transform = []
        sheet_name = 'Zed Analysis'
        Cam_Rot = []
        Cam_Translation = []
        pymesh = []
        tracking = []



    #Initialize Variables
        Test_Prefix =  'Test_0'#'Comb_Basement_1_720_60_0'
##        step = 1 #This is the sample number you're looking at when it is broken up
        num_cam = 1
        load_mesh_opt = False #Change this to True if you want to use a previous mesh / area
        map_range = 0 #This is recommended to be 0 since SDK 2.6 as it will automatically calculate from resolution_meter #This is how far the map should go out to (range_meter)
        map_resolution = 0.1 #This is the resolution of the map that we are going for (in meters)
        map_type = sl.SPATIAL_MAP_TYPE.MESH# FUSED_POINT_CLOUD #MESH #sl.SPATIAL_MAP_TYPE.FUSED_POINT_CLOUD #Other Option is sl.SPATIAL_MAP_TYPE.MESH
        depth_mode = sl.DEPTH_MODE.ULTRA #ULTRA is the highest, QUALITY, is middle, PERFORMANCE, is speed
        sensing_mode = sl.SENSING_MODE.STANDARD #This changes sensing modes, STANDARD = normal, doesn't distort depth, FILL = smoother map, less accurate depth
        coordinate_units = sl.UNIT.METER #This defines what the mesh is generated in METER is meters,
        filter_level = sl.MESH_FILTER.MEDIUM #False, sl.MESH_FILTER.LOW #MEDIUM, LOW, HIGH are the options, otherwise "False" will result in no filter
        save_texture = True #Set this to True if you want to save a texture, otherwise False
        max_memory = 8192 #default is 2048
        mesh_tex = sl.MESH_TEXTURE_FORMAT.RGBA
        conf_threshold = 100 #Confidence threshold for pixel depth, at 100 all pixels are accepted, at 90 it drops the bottom 10% at 30 it drops the bottom 70%
        tex_threshold = 100 #Confidence threshold for texture, at 100 all pixels are accepted, at 90 it drops the bottom 10% at 30 it drops the bottom 70%
        min_depth = 0.5 #Can go as low as 0.1m for Zed Mini, Default is 0.2m, Maximum is 3m
##        max_depth = map_range #25# 20m is the furthest range for Zed Mini, not necessary to set, has no impact on positional tracking or spatial mapping.
        
    #Define Geometry:
        #For converting the angle of the cameras
        transform.append(sl.Transform())

        #Set the initial rotations and translations.    
        # Cam_Rot.append(sl.Rotation())
        # Cam_Rot[0].set_euler_angles(0,0,0, radian = False) #This is the angles of the sensor head based on Neuroworks Design
        
        # Cam_Translation.append(sl.Translation())
        # Cam_Translation[0].init_vector(0, 1.0, 0) #Assuming it starts at 1.5 m high #Still need to adjust for the specific dimensions of Cameras

        # transform[0].init_rotation_translation(Cam_Rot[0], Cam_Translation[0])


    #Define Running Parameters
        #Combined operation / setup
        runtime = sl.RuntimeParameters()
        runtime.measure3D_reference_frame = sl.REFERENCE_FRAME.WORLD #IF you want to set to floor coordinates.
        runtime.sensing_mode = sensing_mode
        runtime.confidence_threshold = conf_threshold
        runtime.textureness_confidence_threshold = tex_threshold #This may not be able to be set here, if not then set it in the constructor

    #Create mesh Example
        spatial = sl.SpatialMappingParameters(map_type = map_type)
        spatial.save_texture = save_texture
        spatial.range_meter = map_range #Set this to 0 this out to see if we get improved results with Auto Range.
        spatial.set_range(sl.MAPPING_RANGE.AUTO) #AUTO calculates automatically with resolution
    ##    spatial.set_resolution(sl.MAPPING_RESOLUTION.MEDIUM) #This is for lower level of detail, minor geometry will be ignored.
        # spatial.resolution_meter = map_resolution   #This allows to set the specific depth resolution you want 
        # spatial.map_type = map_type
        spatial.max_memory_usage = max_memory


    #Initialize Cameras
        for i in range(num_cam):
            filepath.append(Test_Prefix + '_SVO_0'+ str(i) + str(step) +'.svo')
            out_file.append('Position_Data_' + str(i) + '_'+ str(step) + '.xlsx')
            area_path.append('Zed_'+str(i)+ '_'+ str(step)+'.area')
            save_path.append('Zed_'+str(i)+ '_'+ str(step)+'_Mesh')
            load_mesh.append('Zed_'+str(i)+ '_'+ str(step)+'_Mesh.obj')

            print("Reading SVO file: {0}".format(filepath[i]))
            input_type = sl.InputType()
            input_type.set_from_svo_file(filepath[i])
            init = sl.InitParameters(input_t = input_type,svo_real_time_mode=False)
            init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
            # init.depth_mode = depth_mode
            init.coordinate_units = coordinate_units
            # init.depth_minimum_distance = min_depth
##            init.depth_maximum_distance = max_depth
            cam_list.append(sl.Camera())
            status = cam_list[i].open(init)
            if status != sl.ERROR_CODE.SUCCESS:
                print(repr(status))
                exit(1)
            pymesh.append(sl.Mesh())

            tracking = sl.PositionalTrackingParameters(_init_pos=transform[i])

            if load_mesh_opt: #This is used to load a previous area file and mesh.
                print("load_mesh is True")
                pymesh[i].load(load_mesh[i], update_mesh = True)
                tracking[i].area_file_path = area_path[i] 
            
            err = cam_list[i].enable_positional_tracking(tracking)
            if err != sl.ERROR_CODE.SUCCESS:
                print(repr(err))
                exit(-1)

            err = cam_list[i].enable_spatial_mapping(spatial)
            if err != sl.ERROR_CODE.SUCCESS:
                print(repr(err))
                exit(-1)


        grab_run_segment(step, spatial) #This is the one to use in single thread mode.

        #Code for running multiple threads
    ##    for index in range (0, num_cam):
    ##        thread_list.append(threading.Thread(target = grab_run_thread, args = (index,)))
    ##        print("Starting Thread " + str(index))
    ##        thread_list[index].start()
    ##        
    ##
    ##    #Testing different re-join methods:
    ##    for index in range(0, len(thread_list)):
    ##        print("Joining Thread " + str(index))
    ##        thread_list[index].join()

        print("with Thread Single Thread : ", timer()-start)



def save_area(cam, area_path):
    ##    #Save the Area for future use
    print("In Save_Area")
    cam.save_current_area(area_path)
    print(repr(cam.get_area_export_state()))

# def load_mesh(pymesh, load_mesh):
#     #Loading in a previous mesh
#     pymesh.load(load_mesh)

def print_camera_information(cam):
    while True:
        res = input("Do you want to display camera information? [y/n]: ")
        if res == "y":
            print()
            print(repr((cam.get_self_calibration_state())))
            print("Distorsion factor of the right cam before calibration: {0}.".format(
                cam.get_camera_information().calibration_parameters_raw.right_cam.disto))
            print("Distorsion factor of the right cam after calibration: {0}.\n".format(
                cam.get_camera_information().calibration_parameters.right_cam.disto))

            print("Confidence threshold: {0}".format(cam.get_confidence_threshold()))
            print("Depth min and max range values: {0}, {1}".format(cam.get_depth_min_range_value(),
                                                                    cam.get_depth_max_range_value()))

            print("Resolution: {0}, {1}.".format(round(cam.get_resolution().width, 2), cam.get_resolution().height))
            print("Camera FPS: {0}".format(cam.get_camera_fps()))
            print("Frame count: {0}.\n".format(cam.get_svo_number_of_frames()))
            break
        elif res == "n":
            print("Camera information not displayed.\n")
            break
        else:
            print("Error, please enter [y/n].\n")


def saving_image(key, mat):
    if key == 115:
        img = sl.ERROR_CODE.ERROR_CODE_FAILURE
        while img != sl.ERROR_CODE.SUCCESS:
            filepath = input("Enter filepath name: ")
            img = mat.write(filepath)
            print("Saving image : {0}".format(repr(img)))
            if img == sl.ERROR_CODE.SUCCESS:
                break
            else:
                print("Help: you must enter the filepath + filename + PNG extension.")


def saving_depth(cam):
    while True:
        res = input("Do you want to save the depth map? [y/n]: ")
        if res == "y":
            save_depth = 0
            while not save_depth:
                filepath = input("Enter filepath name: ")
                save_depth = sl.save_camera_depth_as(cam, sl.DEPTH_FORMAT.DEPTH_FORMAT_PNG, filepath)
                if save_depth:
                    print("Depth saved.")
                    break
                else:
                    print("Help: you must enter the filepath + filename without extension.")
            break
        elif res == "n":
            print("Depth will not be saved.")
            break
        else:
            print("Error, please enter [y/n].")


def saving_point_cloud(cam):
    while True:
        res = input("Do you want to save the point cloud? [y/n]: ")
        if res == "y":
            save_point_cloud = 0
            while not save_point_cloud:
                filepath = input("Enter filepath name: ")
                save_point_cloud = sl.save_camera_point_cloud_as(cam,
                                                                   sl.POINT_CLOUD_FORMAT.POINT_CLOUD_FORMAT_PLY_ASCII, #Was as pcd format
                                                                   filepath, True)
                if save_point_cloud:
                    print("Point cloud saved.")
                    break
                else:
                    print("Help: you must enter the filepath + filename without extension.")
            break
        elif res == "n":
            print("Point cloud will not be saved.")
            break
        else:
            print("Error, please enter [y/n].")

def run(cam, runtime, camera_pose, viewer, py_translation):
    while True:
        if cam.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            tracking_state = cam.get_position(camera_pose)
            text_translation = ""
            text_rotation = ""
            if tracking_state == sl.TRACKING_STATE.TRACKING_STATE_OK:
                rotation = camera_pose.get_rotation_vector()
                rx = round(rotation[0], 2)
                ry = round(rotation[1], 2)
                rz = round(rotation[2], 2)

                translation = camera_pose.get_translation(py_translation)
                tx = round(translation.get()[0], 2)
                ty = round(translation.get()[1], 2)
                tz = round(translation.get()[2], 2)

                text_translation = str((tx, ty, tz))
                text_rotation = str((rx, ry, rz))
                pose_data = camera_pose.pose_data(sl.Transform())
                viewer.update_zed_position(pose_data)

            viewer.update_text(text_translation, text_rotation, tracking_state)
        else:
            sl.c_sleep_ms(1)

def start_zed(cam, runtime, camera_pose, viewer, py_translation):
    zed_callback = threading.Thread(target=run, args=(cam, runtime, camera_pose, viewer, py_translation))
    zed_callback.start()

def print_mesh_information(pymesh, apply_texture):
    while True:
        res = input("Do you want to display mesh information? [y/n]: ")
        if res == "y":
            if apply_texture:
                print("Vertices : \n{0} \n".format(pymesh.vertices))
                print("Uv : \n{0} \n".format(pymesh.uv))
                print("Normals : \n{0} \n".format(pymesh.normals))
                print("Triangles : \n{0} \n".format(pymesh.triangles))
                break
            else:
                print("Cannot display information of the sl.")
                break
        if res == "n":
            print("Mesh information will not be displayed.")
            break
        else:
            print("Error, please enter [y/n].")


def save_filter(filter_params):
    while True:
        res = input("Do you want to save the mesh filter parameters? [y/n]: ")
        if res == "y":
            params = sl.ERROR_CODE.FAILURE
            while params != sl.ERROR_CODE.SUCCESS:
                filepath = input("Enter filepath name : ")
                params = filter_params.save(filepath)
                print("Saving mesh filter parameters: {0}".format(repr(params)))
                if params:
                    break
                else:
                    print("Help : you must enter the filepath + filename without extension.")
            break
        elif res == "n":
            print("Mesh filter parameters will not be saved.")
            break
        else:
            print("Error, please enter [y/n].")


def save_mesh(pymesh, filepath):
    while True:
        res = 'y' #input("Do you want to save the mesh? [y/n]: ")
        if res == "y":
            msh = sl.ERROR_CODE.FAILURE
            while msh != sl.ERROR_CODE.SUCCESS:
##                filepath = input("Enter filepath name: ")
                msh = pymesh.save(filepath)
                print("Saving mesh: {0}".format(repr(msh)))
                if msh:
                    break
                else:
                    print("Help : you must enter the filepath + filename without extension.")
            break
        elif res == "n":
            print("Mesh will not be saved.")
            break
        else:
            print("Error, please enter [y/n].")

def grab_run():
    global thread_list
    global stop_signal
    global cam_list
    global filepath
    global save_path
    global area_path
    global load_mesh
    global out_file
    global sheet_name
    global runtime
    global spatial
    global pymesh
    global filter_level
    global mesh_tex
    global pymesh
    global save_texture

    print(save_texture)
    
#Put this in the camera specific thread Local Variables
    camera_pose = sl.Pose()
    py_translation = sl.Translation()
    rx, ry, rz = [[], []], [[], []], [[], []]
    tx, ty, tz = [[], []], [[], []], [[], []]
    time_val = [[], []]
    rot = [[],[]]
    trans = [[],[]]
    df = [[],[]]

    mapping_state = [[],[]]
    pos_state = [[],[]]

    print("Processing ")

    if len(cam_list) == 1:
        num_frames = cam_list[0].get_svo_number_of_frames()
    elif len(cam_list) == 2:
        num_frames = min(cam_list[0].get_svo_number_of_frames(), cam_list[1].get_svo_number_of_frames())
    print('Number of Frames: ', num_frames)
##    for i in range(0, 10):
    for i in range(num_frames):
##        print(f'Frame Number: {i}/{num_frames}')
        for index in range(0, len(cam_list)):
            cam_list[index].grab(runtime)
            mapping_state[index] = np.append(mapping_state[index], cam_list[index].get_spatial_mapping_state())
            
            #Catch Pose Data - Camera 1
            tracking_state = cam_list[index].get_position(camera_pose, sl.REFERENCE_FRAME.WORLD)
            pos_state[index] = np.append(pos_state[index], tracking_state)
            if tracking_state != sl.POSITIONAL_TRACKING_STATE.FPS_TOO_LOW: # == sl.POSITIONAL_TRACKING_STATE.OK: #Tracking state returns .OK only if a new pose is available.
                rot[index] = camera_pose.get_rotation_vector()
                rx[index] = np.append(rx[index], rot[index][0])
                ry[index] = np.append(ry[index], rot[index][1])
                rz[index] = np.append(rz[index], rot[index][2])

                trans[index] = camera_pose.get_translation(py_translation).get()
                tx[index] = np.append(tx[index], trans[index][0])
                ty[index] = np.append(ty[index], trans[index][1])
                tz[index] = np.append(tz[index], trans[index][2])

##                time_val2 = sl.Timestamp.get_nanoseconds(camera_pose.timestamp)
##                time_val[index] = np.append(time_val[index], sl.Timestamp.get_nanoseconds(cam_list[index].get_timestamp(sl.TIME_REFERENCE.IMAGE)))
                time_val[index] = np.append(time_val[index], sl.Timestamp.get_nanoseconds(camera_pose.timestamp))

##                print(f'Frame Number: {i}')
##                print(f'Time Value: {time_val[index][i]}')
##                print(f'Pose Time Value: {time_val2}')
            else: #Append error values
                rx[index] = np.append(rx[index], 999)
                ry[index] = np.append(ry[index], 999)
                rz[index] = np.append(rz[index], 999)

                tx[index] = np.append(tx[index], 999)
                ty[index] = np.append(ty[index], 999)
                tz[index] = np.append(tz[index], 999)

                time_val[index] = np.append(time_val[index], sl.Timestamp.get_nanoseconds(camera_pose.timestamp))

    #Output the position data of the microphone Camera 1
    for index in range (0, len(cam_list)):
        df[index] = pd.DataFrame()
        df[index]['Time_val'] = time_val[index] #This is measured in nano seconds since last epoch, For some reason the initial two have messed up time stamps, so taking only from 2 onward.
        df[index]['Time'] = (time_val[index]-time_val[index][0])/1000000000 #Subtracting the first measurement and dividing by 10^9 to get it into seconds.
        df[index]['loc_x'] = tx[index]
        df[index]['loc_y'] = ty[index]
        df[index]['loc_z'] = tz[index]
        df[index]['rot_x'] = rx[index]
        df[index]['rot_y'] = ry[index]
        df[index]['rot_z'] = rz[index]
        df[index]['map_state'] = mapping_state[index]
        df[index]['pos_state'] = pos_state[index]
        df[index].to_excel(out_file[index], sheet_name)

        cam_list[index].extract_whole_spatial_map(pymesh[index])

        #Disable Camera
        cam_list[index].disable_positional_tracking(area_path[index])
        cam_list[index].disable_spatial_mapping()

        if filter_level:
            #Filter and Save Pymesh Cam 1
            print('Filtering Cam ' + str(index))
            filter_params = sl.MeshFilterParameters()
            filter_params.set(filter_level)#Was Medium
            print("Filtering params : {0}.".format(pymesh[index].filter(filter_params)))
        elif filter_level == False:
            print("Not Filtered")

        if save_texture:
            apply_texture = pymesh[index].apply_texture(mesh_tex) #Was RGBA
            print("Applying texture : {0}.".format(apply_texture))
        elif save_texture == False:
            print("No Texture Applied")
            
        save_mesh(pymesh[index], save_path[index])
        cam_list[index].close()
        print("\nFINISHED CAM "+ str(index))

def grab_run_thread(index):
    global thread_list
    global stop_signal
    global cam_list
    global filepath
    global save_path
    global area_path
    global load_mesh
    global out_file
    global sheet_name
    global runtime
    global spatial
    global pymesh
    global filter_level
    global mesh_tex
    global pymesh
    
#Put this in the camera specific thread Local Variables
    camera_pose = sl.Pose()
    py_translation = sl.Translation()
    rx, ry, rz = [], [], []
    tx, ty, tz = [], [], []
    time_val = []

    print("Processing Camera " + str(index))
##    for i in range(0, 5000):
    print('Number of Frames: ', cam_list[index].get_svo_number_of_frames())
    for i in range(cam_list[index].get_svo_number_of_frames()):
        cam_list[index].grab(runtime)

        #Catch Pose Data - Camera 1
        tracking_state = cam_list[index].get_position(camera_pose, sl.REFERENCE_FRAME.WORLD)
        if tracking_state != sl.POSITIONAL_TRACKING_STATE.FPS_TOO_LOW: # == sl.POSITIONAL_TRACKING_STATE.OK: #Tracking state returns .OK only if a new pose is available.
            rot = camera_pose.get_rotation_vector()
            rx = np.append(rx, rot[0])
            ry = np.append(ry, rot[1])
            rz = np.append(rz, rot[2])

            trans = camera_pose.get_translation(py_translation).get()
            tx = np.append(tx, trans[0])
            ty = np.append(ty, trans[1])
            tz = np.append(tz, trans[2])

            time_val = np.append(time_val, sl.Timestamp.get_nanoseconds(camera_pose.timestamp))
        else: #Append error values
            rx = np.append(rx, 999)
            ry = np.append(ry, 999)
            rz = np.append(rz, 999)

            tx = np.append(tx, 999)
            ty = np.append(ty, 999)
            tz = np.append(tz, 999)

            time_val = np.append(time_val, sl.Timestamp.get_nanoseconds(camera_pose.timestamp))

    #Output the position data of the microphone Camera 1        
    df = pd.DataFrame()
    df['Time_val'] = time_val #This is measured in nano seconds since last epoch
    df['Time'] = (time_val-time_val[0])/1000000000 #Subtracting the first measurement and dividing by 10^9 to get it into seconds.
    df['loc_x'] = tx
    df['loc_y'] = ty
    df['loc_z'] = tz
    df['rot_x'] = rx
    df['rot_y'] = ry
    df['rot_z'] = rz
    df.to_excel(out_file[index], sheet_name)

    cam_list[index].extract_whole_spatial_map(pymesh[index])

    #Disable Camera
    cam_list[index].disable_positional_tracking(area_path[index])
    cam_list[index].disable_spatial_mapping()

    #Filter and Save Pymesh Cam 1
    print('Filtering Cam ' + str(index))
    filter_params = sl.MeshFilterParameters()
    filter_params.set(filter_level)#Was Medium
    print("Filtering params : {0}.".format(pymesh[index].filter(filter_params)))

    apply_texture = pymesh[index].apply_texture(mesh_tex) #Was RGBA
    print("Applying texture : {0}.".format(apply_texture))
    save_mesh(pymesh[index], save_path[index])

    cam_list[index].close()
    print("\nFINISHED CAM "+ str(index))

if __name__ == "__main__":
    main()
