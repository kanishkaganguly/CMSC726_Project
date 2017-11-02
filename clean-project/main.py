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
        rootHandle = sim_functions.get_handle("Quadricopter_root")
        quad_functions = quad_helper.QuadHelper(clientID, quadHandle, targetHandle, rootHandle)

        if clientID != -1:
            # Initialize Quad Variables
            curr_rotor_thrusts = [0.001, 0.001, 0.001, 0.001]
            new_rotor_thrusts = curr_rotor_thrusts
            target_pos, target_euler = quad_functions.fetch_target_state()

            # Initialize Network Variables
            nn_functions = pytorch_helper.NN()
            output_vector = nn_functions.generate_output_combos()
            nn_functions.D_in = 10
            nn_functions.D_out = len(output_vector)
            nn_functions.create_in_out_vars()
            batch_size = 1000
            epoch = 10000
            nn_functions.create_model()
            print('Initialized Network')

            # Initialize Simulator and Quadrotor
            sim_functions.start_sim()
            print('Simulator Started')
            quad_functions.init_quad()
            print('Quadrotor Initialized')

            while vrep.simxGetConnectionId(clientID) != -1:
                for i in range(epoch):
                    print("EPOCH %d" % i)
                    # INPUT CURRENT STATE
                    curr_pos, curr_euler = quad_functions.fetch_quad_state()
                    sim_functions.pause_sim()
                    curr_state = np.array(curr_pos + curr_euler +
                                          curr_rotor_thrusts, dtype=np.float32)
                    nn_functions.input_var.data = nn_functions.np_to_torch(curr_state)

                    # GET Q VALUES
                    output_var = nn_functions.get_predicted_data(nn_functions.input_var)
                    q_vals = nn_functions.torch_to_np(output_var.data)

                    # GET MAX Q VALUES
                    max_qval_idx = np.argmax(q_vals)
                    delta_thrust = output_vector[max_qval_idx]
                    new_rotor_thrusts[0] = curr_rotor_thrusts[0] + delta_thrust[0]
                    new_rotor_thrusts[1] = curr_rotor_thrusts[1] + delta_thrust[1]
                    new_rotor_thrusts[2] = curr_rotor_thrusts[2] + delta_thrust[2]
                    new_rotor_thrusts[3] = curr_rotor_thrusts[3] + delta_thrust[3]
                    sim_functions.start_sim()

                    # DO MAX Q VALUE ACTION
                    quad_functions.apply_rotor_thrust(new_rotor_thrusts)
                    for _ in range(batch_size):
                        pass
                    sim_functions.pause_sim()

                    # GET NEW STATE
                    next_pos, next_euler = quad_functions.fetch_quad_state()
                    # GET REWARD
                    reward = quad_functions.get_reward(next_pos, next_euler, target_pos, target_euler)

                    # GET ONE HOT ERROR
                    onehot_err = nn_functions.onehot_from_reward(reward, len(q_vals), max_qval_idx)

                    # DO BACKPROP
                    nn_functions.error_var.data = nn_functions.np_to_torch(onehot_err)
                    nn_functions.get_loss(nn_functions.output_var, nn_functions.error_var)
                    nn_functions.do_backprop()
                    print("Loss: %f" % nn_functions.loss.data[0])
                    print("\n")
                    quad_functions.reset_quad()
                    sim_functions.start_sim()
        else:
            print("Failed to connect to remote API Server")
            sim_functions.exit_sim()
    except KeyboardInterrupt:
        sim_functions.exit_sim()
    finally:
        sim_functions.exit_sim()


if __name__ == '__main__':
    main()
