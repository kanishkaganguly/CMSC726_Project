#!/usr/bin/python3

import vrep
import vrep_helper

clientID = -1
sim_functions = None


def get_quad_target(clientID):
    err, quadHandle = vrep.simxGetObjectHandle(clientID, 'Quadricopter_target', vrep.simx_opmode_blocking)
    return quadHandle


def get_final_target(clientID):
    err, gotoHandle = vrep.simxGetObjectHandle(clientID, 'Quadricopter_target', vrep.simx_opmode_blocking)
    return gotoHandle


def init_quad(clientID, quadHandle):
    vrep.simxGetObjectPosition(clientID, quadHandle, quadHandle, vrep.simx_opmode_streaming)
    vrep.simxGetObjectOrientation(clientID, quadHandle, quadHandle, vrep.simx_opmode_streaming)


def set_quad_position(clientID, quadHandle, pos):
    vrep.simxSetObjectPosition(clientID, quadHandle, quadHandle, pos, vrep.simx_opmode_oneshot_wait)
    return


def get_quad_position(clientID, quadHandle):
    err, pos = vrep.simxGetObjectPosition(clientID, quadHandle, quadHandle, vrep.simx_opmode_buffer)
    return pos


def main():
    try:
        vrep.simxFinish(-1)
        clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
        sim_functions = vrep_helper.Helper(clientID)

        if clientID != -1:
            sim_functions.start_sim()
            print('Simulator Started')

            sim_functions = vrep_helper.Helper(clientID)
            quad_target = get_quad_target(clientID)
            init_quad(clientID, quad_target)
            print('Initialized Quadrotor')

            while vrep.simxGetConnectionId(clientID) != -1:
                curr_pos = get_quad_position(clientID, quad_target)
                curr_pos[2] += 0.001
                new_pos = curr_pos
                set_quad_position(clientID, quad_target, new_pos)

        else:
            print("Failed to connect to remote API Server")
            sim_functions.exit_sim()
    except KeyboardInterrupt:
        sim_functions.exit_sim()
    finally:
        sim_functions.exit_sim()


if __name__ == '__main__':
    main()
