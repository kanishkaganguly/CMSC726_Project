#!/usr/bin/python

import vrep
import vrep_helper
import numpy as np
import cv2

near_clipping_plane = 0.5
far_clipping_plane = 3.0
perspective_angle = 1.0472
resolution_x = 640
resolution_y = 480


def init_cam(clientID):
    res, depthHandle = vrep.simxGetObjectHandle(clientID, 'kinect_depth', vrep.simx_opmode_oneshot_wait)
    vrep.simxSetObjectFloatParameter(clientID, depthHandle, vrep.sim_visionfloatparam_near_clipping,
                                     near_clipping_plane,
                                     vrep.simx_opmode_oneshot_wait)
    vrep.simxSetObjectFloatParameter(clientID, depthHandle, vrep.sim_visionfloatparam_far_clipping, far_clipping_plane,
                                     vrep.simx_opmode_oneshot_wait)
    vrep.simxSetObjectFloatParameter(clientID, depthHandle, vrep.sim_visionfloatparam_perspective_angle,
                                     perspective_angle,
                                     vrep.simx_opmode_oneshot_wait)
    vrep.simxSetObjectIntParameter(clientID, depthHandle, vrep.sim_visionintparam_resolution_x, resolution_x,
                                   vrep.simx_opmode_oneshot_wait)
    vrep.simxSetObjectIntParameter(clientID, depthHandle, vrep.sim_visionintparam_resolution_y, resolution_y,
                                   vrep.simx_opmode_oneshot_wait)
    vrep.simxSetObjectIntParameter(clientID, depthHandle, vrep.sim_visionintparam_render_mode, 3,
                                   vrep.simx_opmode_oneshot_wait)
    vrep.simxGetVisionSensorDepthBuffer(clientID, depthHandle, vrep.simx_opmode_streaming)
    return depthHandle


def get_depth(clientID, depthHandle):
    err, resolution, buffer = vrep.simxGetVisionSensorDepthBuffer(clientID, depthHandle, vrep.simx_opmode_buffer)
    if err == vrep.simx_return_ok:
        buff = np.array(buffer)
        buff.resize([resolution[1], resolution[0]])
        return buff
    elif err == vrep.simx_return_novalue_flag:
        return 0, None
        pass
    else:
        return err, None


def main():
    helper = None
    try:
        vrep.simxFinish(-1)
        clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)

        if clientID != -1:
            helper = vrep_helper.Helper(clientID)
            print('Main Script Started')

            depth_handle = init_cam(clientID)
            helper.start_sim()
            while vrep.simxGetConnectionId(clientID) != -1:
                depth_out = get_depth(clientID, depth_handle)

                # # depth_array_disp = (2.0 * near_clipping_plane * far_clipping_plane) / (
                # #     far_clipping_plane + near_clipping_plane - (2 * depth_out - 1) * (
                # #         far_clipping_plane - near_clipping_plane))
                # # depth_array_disp = np.flipud(depth_array_disp)
                # # max_val = np.max(np.max(depth_array_disp))
                # #
                # # depth_array_disp[depth_array_disp >= far_clipping_plane] = 0
                # # depth_array_save = depth_array_disp * 1000
                # # depth_array_save = depth_array_save.astype(np.uint16)
                # # cv2.imwrite('save_test.png', depth_array_save)
                #
                # cv2.imshow('DEPTH', depth_array_disp / max_val)
                cv2.imshow('DEPTH', depth_out)
                cv2.waitKey(1)
        else:
            print "Failed to connect to remote API Server"
            helper.stop_sim()
            vrep.simxFinish(clientID)
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        helper.stop_sim()
        vrep.simxFinish(clientID)


if __name__ == '__main__':
    main()
