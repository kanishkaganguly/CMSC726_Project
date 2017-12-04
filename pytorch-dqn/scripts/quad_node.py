#! /usr/bin/env python3
import argparse

import numpy as np

from pytorch_helper import QuadDQN
from quad_helper import QuadHelper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch_size", type=int, default=10000, help="Total training epochs")
    parser.add_argument("--episode_size", type=int, default=100000, help="Training episodes per epoch")
    parser.add_argument("--epsilon", type=float, default=0.01, help="Greedy Epsilon starting value")
    parser.add_argument("--gamma", type=float, default=0.6, help="DQN gamma starting value")
    parser.add_argument("--load_model", action='store_true', default=False, help="Load saved model")

    args = parser.parse_args()

    control_quad = QuadHelper()
    dqn_quad = QuadDQN()

    # Argument parsing
    epoch_size = args.epoch_size
    dqn_quad.episode_size = args.episode_size
    dqn_quad.eps = args.epsilon
    dqn_quad.gamma = args.gamma
    if args.load_model:
        dqn_quad.load_wts('dqn_quad.pth')

    epoch = 0
    while epoch < epoch_size:
        for i in range(dqn_quad.episode_size):
            print("Epoch: %d Episode %d" % (epoch, i))
            print("Epsilon Greedy: %f" % dqn_quad.eps)
            print("DQN Discounted Reward: %f" % dqn_quad.gamma)
            # Get current state
            print("Getting current state")
            curr_state = np.array(control_quad.get_state(), dtype=np.float32)
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

            # Test out of bounds
            test_state = control_quad.get_state()
            if abs(test_state[0]) > 10.0 or abs(test_state[1]) > 10.0 or test_state[2] > 5.0 or test_state[2] < 0.0:
                print("Quadcopter out of bounds")
                # Get reward
                print("Getting reward")
                reward = -50
                # Set target q_values for backprop
                print("Setting target values")
                target_q = np.copy(pred_q)
                target_q[max_q_idx] = reward + dqn_quad.gamma * max_q
                print("Computing loss")
                dqn_quad.get_loss(target_q, pred_q)
                # Do backprop
                print("Backpropagation")
                dqn_quad.backprop()
                print('\n')
                break
            else:
                # Get reward
                print("Getting reward")
                reward = dqn_quad.get_reward(new_state, control_quad.target_state)
                # Set target q_values for backprop
                print("Setting target values")
                target_q = np.copy(pred_q)
                target_q[max_q_idx] = reward + dqn_quad.gamma * max_q
                print("Computing loss")
                dqn_quad.get_loss(target_q, pred_q)
                # Do backprop
                print("Backpropagation")
                dqn_quad.backprop()
                print('\n')
            if i % 100 == 0:
                with open('dqn_outputs.txt', 'a') as the_file:
                    the_file.write('Epoch: %d Episode: %d\n' % (epoch, i))
                    the_file.write('Epsilon Greedy: %f\n' % dqn_quad.eps)
                    the_file.write('Reward: %f\n' % reward)
                    the_file.write('Loss: %f\n' % float(dqn_quad.loss.data[0]))
                    the_file.write('\n')
        print("Epoch reset")
        epoch += 1
        if epoch % 100 == 0:
            dqn_quad.save_wts('dqn_quad.pth', epoch)
        if epoch % 50 == 0:
            dqn_quad.eps += 0.001
            control_quad.reset(rand_target=True)
        else:
            control_quad.reset()
        print('\n')


if __name__ == '__main__':
    main()
