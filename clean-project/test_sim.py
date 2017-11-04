#!/usr/bin/python3

import quad_helper
import vrep
import vrep_helper

clientID = -1
sim_functions = None
quad_functions = None


def main():
    try:
        vrep.simxFinish(-1)
        clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
        vrep.simxSynchronous(clientID, True)
        sim_functions = vrep_helper.Helper(clientID)
        quadHandle = sim_functions.get_handle("Quadricopter")
        targetHandle = sim_functions.get_handle("Quadricopter_target")
        quad_functions = quad_helper.QuadHelper(clientID, quadHandle, targetHandle)

        if clientID != -1:
            curr_rotor_thrusts = [0.001, 0.001, 0.001, 0.001]
            new_rotor_thrusts = curr_rotor_thrusts
            sim_functions.start_sim()
            print('Simulator Started')
            quad_functions.init_quad()
            print('Quadrotor Initialized')
            while vrep.simxGetConnectionId(clientID) != -1:
                curr_pos, curr_euler = quad_functions.fetch_quad_state()
                print(curr_pos, curr_euler)
                delta_thrust = [0.0000, 0.0000, 0.0000, 0.0000]
                new_rotor_thrusts[0] = curr_rotor_thrusts[0] + delta_thrust[0]
                new_rotor_thrusts[1] = curr_rotor_thrusts[1] + delta_thrust[1]
                new_rotor_thrusts[2] = curr_rotor_thrusts[2] + delta_thrust[2]
                new_rotor_thrusts[3] = curr_rotor_thrusts[3] + delta_thrust[3]
                quad_functions.apply_rotor_thrust(new_rotor_thrusts)
                vrep.simxSynchronousTrigger(clientID)

    except KeyboardInterrupt:
        sim_functions.exit_sim()
    finally:
        sim_functions.exit_sim()


if __name__ == '__main__':
    main()
