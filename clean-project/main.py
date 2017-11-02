#!/usr/bin/python3

import numpy as np

import pytorch_helper
import quad_helper
import vrep
import vrep_helper

clientID = -1
sim_functions = None
quad_functions = None
nn_functions = None


def main():
    try:
        vrep.simxFinish(-1)
        clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
        sim_functions = vrep_helper.Helper(clientID)
        quadHandle = sim_functions.get_handle("Quadricopter")
        targetHandle = sim_functions.get_handle("Quadricopter_target")
        quad_functions = quad_helper.RL(clientID, quadHandle, targetHandle)

        if clientID != -1:
            # Initialize Quad Variables
            curr_rotor_thrusts = [0.001, 0.001, 0.001, 0.001]
            target_pos, target_euler = quad_functions.fetch_target_state()

            # Initialize Network Variables
            nn_functions = pytorch_helper.NN()
            output_vector = nn_functions.generate_output_combos()
            nn_functions.D_in = 10
            nn_functions.D_out = len(output_vector)
            input_var, output_var = nn_functions.create_in_out_vars()
            batch_size = 10000
            epoch = 500000
            nn_functions.create_model()
            print('Initialized Network')

            # Initialize Simulator and Quadrotor
            sim_functions.start_sim()
            print('Simulator Started')
            quad_functions.init_quad()
            print('Quadrotor Initialized')

            while vrep.simxGetConnectionId(clientID) != -1:
                for _ in range(epoch):
                    for _ in range(batch_size):
                        curr_pos, curr_euler = quad_functions.fetch_quad_state()
                        curr_state = np.array(curr_pos + curr_euler +
                                              curr_rotor_thrusts)
                        input_var = nn_functions.np_to_torch(curr_state)

                        output_var = nn_functions.get_predicted_data(input_var)
                        q_vals = nn_functions.torch_to_np(output_var)
                        print(len(q_vals))
                    quad_functions.reset_quad()
                break
        else:
            print("Failed to connect to remote API Server")
            sim_functions.exit_sim()
    except KeyboardInterrupt:
        sim_functions.exit_sim()
    finally:
        sim_functions.exit_sim()


if __name__ == '__main__':
    main()
