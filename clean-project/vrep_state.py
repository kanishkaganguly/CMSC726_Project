#!/usr/bin/python3

import vrep


def init_state(clientID, quadHandle):
    vrep.simxGetObjectPosition(clientID, quadHandle, -1, vrep.simx_opmode_streaming)
    vrep.simxGetObjectOrientation(clientID, quadHandle, -1, vrep.simx_opmode_streaming)


def get_quad_pos(clientID, quadHandle):
    err, pos = vrep.simxGetObjectPosition(clientID, quadHandle, -1, vrep.simx_opmode_buffer)
    return pos


def get_quad_orientation(clientID, quadHandle):
    err, euler = vrep.simxGetObjectOrientation(clientID, quadHandle, -1, vrep.simx_opmode_buffer)
    return euler


def set_quad_position(clientID, quadHandle, rootHandle, pos):
    vrep.simxSetObjectPosition(clientID, quadHandle, rootHandle, pos, vrep.simx_opmode_oneshot_wait)
    return


def set_quad_euler(clientID, quadHandle, rootHandle, euler):
    vrep.simxSetObjectPosition(clientID, quadHandle, rootHandle, euler, vrep.simx_opmode_oneshot_wait)
    return


def get_target_pos(clientID, targetHandle):
    err, pos = vrep.simxGetObjectPosition(clientID, targetHandle, -1, vrep.simx_opmode_buffer)
    return pos


def get_target_orientation(clientID, targetHandle):
    err, euler = vrep.simxGetObjectOrientation(clientID, targetHandle, -1, vrep.simx_opmode_buffer)
    return euler
