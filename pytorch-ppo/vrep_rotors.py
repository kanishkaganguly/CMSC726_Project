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
        vrep.simxSetFloatSignal(clientID, propellers[i], 1e-8, vrep.simx_opmode_oneshot)


def set_rotors(clientID, propeller_vels):
    [vrep.simxSetFloatSignal(clientID, prop, vels, vrep.simx_opmode_oneshot) for prop, vels in zip(propellers,
                                                                                                   propeller_vels)]
    return
