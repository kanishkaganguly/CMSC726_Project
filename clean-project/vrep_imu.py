#!/usr/bin/python3

import vrep


def init_imu(clientID, quadHandle):
    vrep.simxGetFloatSignal(clientID, 'gyroX', vrep.simx_opmode_streaming)
    vrep.simxGetFloatSignal(clientID, 'gyroY', vrep.simx_opmode_streaming)
    vrep.simxGetFloatSignal(clientID, 'gyroZ', vrep.simx_opmode_streaming)

    vrep.simxGetFloatSignal(clientID, 'accelX', vrep.simx_opmode_streaming)
    vrep.simxGetFloatSignal(clientID, 'accelY', vrep.simx_opmode_streaming)
    vrep.simxGetFloatSignal(clientID, 'accelZ', vrep.simx_opmode_streaming)

    vrep.simxGetObjectPosition(clientID, quadHandle, -1, vrep.simx_opmode_streaming)
    vrep.simxGetObjectOrientation(clientID, quadHandle, -1, vrep.simx_opmode_streaming)


def get_imu(clientID):
    err, gX = vrep.simxGetFloatSignal(clientID, 'gyroX', vrep.simx_opmode_buffer)
    err, gY = vrep.simxGetFloatSignal(clientID, 'gyroY', vrep.simx_opmode_buffer)
    err, gZ = vrep.simxGetFloatSignal(clientID, 'gyroZ', vrep.simx_opmode_buffer)

    err, aX = vrep.simxGetFloatSignal(clientID, 'accelX', vrep.simx_opmode_buffer)
    err, aY = vrep.simxGetFloatSignal(clientID, 'accelY', vrep.simx_opmode_buffer)
    err, aZ = vrep.simxGetFloatSignal(clientID, 'accelZ', vrep.simx_opmode_buffer)

    # print('%.2f,%.2f,%.2f,%.2f,%.2f,%.2f') % (gX, gY, gZ, aX, aY, aZ)
    return [gX, gY, gZ, aX, aY, aZ]


def get_pos(clientID, quadHandle):
    err, pos = vrep.simxGetObjectPosition(clientID, quadHandle, -1, vrep.simx_opmode_buffer)
    return pos


def get_orientation(clientID, quadHandle):
    err, euler = vrep.simxGetObjectOrientation(clientID, quadHandle, -1, vrep.simx_opmode_buffer)
    return euler
