#!/usr/bin/python3

import vrep
import vrep_helper
import vrep_rotors

clientID = -1
sim_functions = None


def main():
    try:
        vrep.simxFinish(-1)
        clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
        sim_functions = vrep_helper.Helper(clientID)

        if clientID != -1:
            sim_functions.start_sim()
            print('Simulator Started')

            sim_functions = vrep_helper.Helper(clientID)
            vrep_rotors.init_rotors(clientID)
            rotor_thrusts = [0.0, 0.0, 0.0, 0.0]
            while vrep.simxGetConnectionId(clientID) != -1:
                rotor_thrusts[0] += 1e-5
                rotor_thrusts[1] += 1e-5
                rotor_thrusts[2] += 1e-5
                rotor_thrusts[3] += 1e-5
                vrep_rotors.move_rotors(clientID, rotor_thrusts)

        else:
            print("Failed to connect to remote API Server")
            sim_functions.exit_sim()
    except KeyboardInterrupt:
        sim_functions.exit_sim()
    finally:
        sim_functions.exit_sim()


if __name__ == '__main__':
    main()
