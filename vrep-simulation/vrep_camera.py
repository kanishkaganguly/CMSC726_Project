#!/usr/bin/python

import vrep
import array
import numpy as np
import cv2
from PIL import Image

def init_cam(clientID):
    res, frontCamHandle = vrep.simxGetObjectHandle(clientID, 'FrontCam', vrep.simx_opmode_oneshot_wait)
    vrep.simxGetVisionSensorImage(clientID, frontCamHandle, 0, vrep.simx_opmode_streaming)
    return frontCamHandle

def get_cam(clientID,frontCamHandle):
    err, resolution, image = vrep.simxGetVisionSensorImage(clientID, frontCamHandle, 0, vrep.simx_opmode_buffer)
    if err == vrep.simx_return_ok:
        image_byte_array = array.array('b', image)
        image_buffer = Image.frombuffer("RGB", (resolution[0], resolution[1]), image_byte_array, "raw", "BGR", 0, 1)
        img_out = np.asarray(image_buffer)
        img_out = cv2.flip(img_out, 0)
        return 1,img_out
    elif err == vrep.simx_return_novalue_flag:
        return 0,None
        pass
    else:
        return err,None