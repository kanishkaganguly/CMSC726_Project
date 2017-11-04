#!/usr/bin/python3

import time

import numpy as np

import pytorch_helper
import quad_helper
import vrep
import vrep_helper
from torch.autograd import Variable

clientID = -1
sim_functions = None
quad_functions = None
nn_functions = None
dt = 0.05


def main():
    try:
        vrep.simxFinish(-1)
        clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
        sim_functions = vrep_helper.Helper(clientID)
        quadHandle = sim_functions.get_handle("Quadricopter")
        targetHandle = sim_functions.get_handle("Quadricopter_target")
        quad_functions = quad_helper.QuadHelper(clientID, quadHandle, targetHandle)

        if clientID != -1:
            # Initialize Quad Variables
            curr_rotor_thrusts = [0.001, 0.001, 0.001, 0.001]
            new_rotor_thrusts = curr_rotor_thrusts
            target_pos, target_euler = quad_functions.fetch_target_state()

            # Initialize Network Variables
            nn_functions = pytorch_helper.NNBase()
            output_vector = nn_functions.generate_output_combos()
            nn_functions.model = pytorch_helper.MLP(10, [128, 256, 128], len(output_vector))

            epoch = 50000
            batch_time = 5
            time_count = 0
            epochs_per_episode = 10

            nn_functions.create_model()
            finished_epochs = nn_functions.load_wts("save.pth.tar")
            print('Initialized Network')

            # Initialize Simulator and Quadrotor
            sim_functions.start_sim()
            print('Simulator Started')
            quad_functions.init_quad()
            print('Quadrotor Initialized')

            while vrep.simxGetConnectionId(clientID) != -1:
                for i in range(epoch):
                    start_time = time.time()
                    print("EPOCH %d" % i)
                    # INPUT CURRENT STATE
                    curr_pos, curr_euler = quad_functions.fetch_quad_state()
                    curr_state = np.array(curr_pos + curr_euler +
                                          curr_rotor_thrusts, dtype=np.float32)

                    # GET Q VALUES
                    output_var = nn_functions.get_predicted_data(nn_functions.np_to_torch(curr_state))
                    q_vals = nn_functions.torch_to_np(output_var.data)

                    # GET MAX Q VALUES
                    max_qval_idx = np.argmax(q_vals)
                    delta_thrust = output_vector[max_qval_idx]
                    new_rotor_thrusts[0] = curr_rotor_thrusts[0] + delta_thrust[0]
                    new_rotor_thrusts[1] = curr_rotor_thrusts[1] + delta_thrust[1]
                    new_rotor_thrusts[2] = curr_rotor_thrusts[2] + delta_thrust[2]
                    new_rotor_thrusts[3] = curr_rotor_thrusts[3] + delta_thrust[3]

                    # DO MAX Q VALUE ACTION
                    quad_functions.apply_rotor_thrust(new_rotor_thrusts)
                    vrep.simxSynchronousTrigger(clientID)
                    while time_count < batch_time:
                        vrep.simxSynchronousTrigger(clientID)
                        time_count += dt
                    time_count = 0

                    # GET NEW STATE
                    next_pos, next_euler = quad_functions.fetch_quad_state()

                    # GET REWARD
                    reward = quad_functions.get_reward(new_rotor_thrusts, next_pos, next_euler, target_pos,
                                                       target_euler)

                    # GET ONE HOT ERROR
                    onehot_err = nn_functions.onehot_from_reward(reward, len(q_vals), max_qval_idx)

                    # DO BACKPROP
                    error_var = nn_functions.np_to_torch(onehot_err)
                    nn_functions.get_loss(nn_functions.output_var, Variable(error_var))
                    nn_functions.do_backprop()

                    print("Loss: %f" % nn_functions.loss.data[0])
                    print(("Time: %f") % (time.time() - start_time))
                    print("\n")

                    # RESET QUAD PER EPISODE
                    if i % epochs_per_episode == 0:
                        print("Episode Finished. Resetting Quad.")
                        print("\n")
                        sim_functions.stop_sim()
                        nn_functions.save_wts("save.pth.tar", i+finished_epochs)
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
