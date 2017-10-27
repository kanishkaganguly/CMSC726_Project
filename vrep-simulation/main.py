#!/usr/bin/python3

import pytorch_helper
import rl_helper
import vrep
import vrep_helper

rl_functions = None
sim_functions = None
nn_functions = None
clientID = -1
try:
    vrep.simxFinish(-1)
    clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)

    if clientID != -1:
        print('Simulator Started')

        print('Initialized Quadrotor')
        sim_functions = vrep_helper.Helper(clientID)
        rl_functions = rl_helper.RL(clientID)
        rl_functions.init_sensors()
        rl_functions.reset_quad()
        sim_functions.start_sim()
        rl_functions.target_location = [0.0, 0.0, 4.0]
        keep_training = True

        print('Initialized Neural Network')
        nn_functions = pytorch_helper.NN()

        while vrep.simxGetConnectionId(clientID) != -1:
            for i in range(1000):
                print('Iteration #' + str(i))
                rl_functions.rotor_data = [1.0 + (i / 10000), 1.0 + (i / 10000), 1.0 + (i / 10000), 1.0 + (i / 10000)]
                rl_functions.do_action()
                # curr_state = np.array(rl_functions.get_state(), dtype=np.float).reshape((1, 6))
                # curr_state_input = nn_functions.get_variable_from_array(curr_state)
                # curr_rotor_output = nn_functions.get_predicted_rotor(curr_state_input)
                # rl_functions.rotor_data += curr_rotor_output.data.numpy()[0]
                # rl_functions.do_action()
                #
                # sim_functions.pause_sim()
                #
                # next_state = np.array(rl_functions.get_state(), dtype=np.float).reshape((1, 6))
                # next_state_input = nn_functions.get_variable_from_array(next_state)
                # next_rotor_output = nn_functions.get_predicted_rotor(next_state_input)
                #
                # loss = nn_functions.get_loss(next_rotor_output, curr_rotor_output)
                # nn_functions.do_backprop(loss)
                #
                # sim_functions.start_sim()

                if i >= 999:
                    rl_functions.reset_quad()

        print("Done Training")
        sim_functions.exit_sim()
    else:
        print("Failed to connect to remote API Server")
        sim_functions.exit_sim()
except KeyboardInterrupt:
    sim_functions.exit_sim()
finally:
    sim_functions.exit_sim()
