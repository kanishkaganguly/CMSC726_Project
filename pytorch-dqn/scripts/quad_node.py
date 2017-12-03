#! /usr/bin/env python3
import numpy as np

from pytorch_helper import QuadDQN
from quad_helper import QuadHelper


def main():
    control_quad = QuadHelper()
    dqn_quad = QuadDQN()

    epoch = 0
    epoch_size = 10000

    while epoch < epoch_size:
        for i in range(dqn_quad.episode_size):
            print("Epoch: %d Episode %d" % (epoch, i))
            # Get current state
            print("Getting current state")
            curr_state = np.array(control_quad.get_state(), dtype=np.float32)
            print(curr_state)
            # Get action q_values
            print("Getting predicted q_values")
            pred_q = dqn_quad.predict_action(curr_state)
            # Get action with max q_value
            print("Getting best action")
            max_q_idx = np.argmax(pred_q)
            max_q = np.amax(pred_q)
            # Do action
            print("Moving quadcopter")
            control_quad.move_quad(dqn_quad.do_action(max_q_idx))
            # Get new state
            print("Getting new state")
            new_state = control_quad.get_state()
            print(new_state)
            # Get reward
            print("Getting reward")
            reward = dqn_quad.get_reward(curr_state, control_quad.target_state)
            # Set target q_values for backprop
            print("Setting target values")
            target_q = pred_q
            target_q[max_q_idx] = reward + dqn_quad.gamma * max_q
            print(target_q)
            print("Computing loss")
            dqn_quad.get_loss(target_q, pred_q)
            # Do backprop
            print("Backpropagation")
            dqn_quad.backprop()
            print('\n')

            test_state = control_quad.get_state()
            if any(abs(test_state[0:2])) > 10.0:
                print("Quadcopter out of bounds")
                break

        print("Epoch reset")
        epoch += 1
        control_quad.sim_quad.reset()
        print('\n')


if __name__ == '__main__':
    main()
