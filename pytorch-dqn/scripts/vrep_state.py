#!/usr/bin/python3

import numpy as np

import vrep


class StateHelper(object):
    def __init__(self, clientID, handleList):
        for eachHandle in handleList:
            vrep.simxGetObjectPosition(clientID, eachHandle, -1, vrep.simx_opmode_streaming)
            vrep.simxGetObjectOrientation(clientID, eachHandle, -1, vrep.simx_opmode_streaming)

    def get_state(self, clientID, objHandle):
        err, pos = vrep.simxGetObjectPosition(clientID, objHandle, -1, vrep.simx_opmode_buffer)
        err, euler = vrep.simxGetObjectOrientation(clientID, objHandle, -1, vrep.simx_opmode_buffer)
        state = np.array([pos[0], pos[1], pos[2], euler[2]])
        return state

    def set_state(self, clientID, objHandle, state):
        pos = [state[0], state[1], state[2]]
        euler = [0.0, 0.0, state[3]]
        vrep.simxSetObjectPosition(clientID, objHandle, -1, pos, vrep.simx_opmode_oneshot)
        vrep.simxSetObjectOrientation(clientID, objHandle, -1, euler, vrep.simx_opmode_oneshot)
