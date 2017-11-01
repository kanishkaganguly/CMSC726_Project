#!/usr/bin/python3

import vrep

# propellers = ['propeller1Vel', 'propeller2Vel', 'propeller3Vel', 'propeller4Vel']
propellers = ['rotor1_thrust', 'rotor2_thrust', 'rotor3_thrust', 'rotor4_thrust']


def init_rotors(clientID):
    # Clear all signals
    for i in range(len(propellers)):
        vrep.simxClearFloatSignal(clientID, propellers[i], vrep.simx_opmode_oneshot)

    # Set all propellers to zero
    for i in range(len(propellers)):
        vrep.simxSetFloatSignal(clientID, propellers[i], 0.0, vrep.simx_opmode_oneshot)


def move_rotors(clientID, propeller_vels):
    thrust1_ret = vrep.simxSetFloatSignal(clientID, propellers[0], propeller_vels[0], vrep.simx_opmode_oneshot)
    thrust2_ret = vrep.simxSetFloatSignal(clientID, propellers[1], propeller_vels[1], vrep.simx_opmode_oneshot)
    thrust3_ret = vrep.simxSetFloatSignal(clientID, propellers[2], propeller_vels[2], vrep.simx_opmode_oneshot)
    thrust4_ret = vrep.simxSetFloatSignal(clientID, propellers[3], propeller_vels[3], vrep.simx_opmode_oneshot)

    # print(thrust1_ret, thrust2_ret, thrust3_ret, thrust4_ret)
