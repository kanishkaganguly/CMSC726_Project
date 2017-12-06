#! /usr/bin/env python3
import argparse
import datetime

import numpy as np

import vrep
from pytorch_helper import QuadDQN
from quad_helper import QuadHelper


class Quad(object):
    def __init__(self, dqn_quad, control_quad):
        self.mode = "train"
        self.dqn_quad = dqn_quad
        self.control_quad = control_quad

    def write_data(self, epoch, reward, iteration):
        with open('dqn_outputs.txt', 'a') as the_file:
            the_file.write('Epoch: %d Episode: %d\n' % (epoch, iteration))
            the_file.write('Epsilon Greedy: %f\n' % self.dqn_quad.eps)
            the_file.write('Reward: %f\n' % reward)
            the_file.write('Loss: %f\n' % float(self.dqn_quad.loss.data[0]))
            the_file.write('Learning Rate: %f\n' % float(self.dqn_quad.scheduler.get_lr()[0]))
            the_file.write('\n')

    def run_one_episode(self, epoch_id, ep_id, mode):
        print("Epoch: %d Episode %d" % (epoch_id, ep_id))
        print("Epsilon Greedy: %f" % self.dqn_quad.eps)
        print("DQN Discount Factor: %f" % self.dqn_quad.gamma)

        # Get current state
        print("Getting current state")
        curr_state = np.array(self.control_quad.get_quad_state(), dtype=np.float32)

        # Get action q_values
        print("Getting predicted q_values")
        pred_q = self.dqn_quad.predict_action(curr_state)

        # Get action with max q_value
        print("Getting best action")
        max_q_idx = np.argmax(pred_q)
        max_q = np.amax(pred_q)

        # Do action
        print("Moving quadcopter")
        self.control_quad.move_quad(self.dqn_quad.do_action(max_q_idx))

        # Get new state
        print("Getting new state")
        new_state = self.control_quad.get_quad_state()

        # Test out of bounds
        test_state = self.control_quad.get_quad_state()
        if abs(test_state[0]) > 10.0 or abs(test_state[1]) > 10.0 or test_state[2] > 5.0 or test_state[2] < 0.0:
            print("Quadcopter out of bounds")
            if mode == 'test':
                return 'Failure'
            elif mode == 'train':
                # Get reward
                print("Getting reward")
                reward = -50
                # Set target q_values for backprop
                print("Setting target values")
                target_q = np.copy(pred_q)
                target_q[max_q_idx] = reward + self.dqn_quad.gamma * max_q
                print("Computing loss")
                self.dqn_quad.get_loss(target_q, pred_q)
                # Do backprop
                print("Backpropagation")
                self.dqn_quad.backprop()
                print('\n')
                return 'break'
            else:
                print("Running episode in invalid mode.")
        elif mode == 'train':
            # Get reward
            print("Getting reward")
            reward = self.dqn_quad.get_reward(new_state, self.control_quad.get_target_state())
            # Set target q_values for backprop
            print("Setting target values")
            target_q = np.copy(pred_q)
            target_q[max_q_idx] = reward + self.dqn_quad.gamma * max_q
            print("Computing loss")
            self.dqn_quad.get_loss(target_q, pred_q)

            # Do backprop
            print("Backpropagation")
            self.dqn_quad.backprop()
            print('\n')

            if ep_id % 100 == 0:
                self.write_data(self.dqn_quad, epoch_id, reward, ep_id)

        return 'continue'

    def run_one_epoch(self, epoch_id):
        for i in range(self.dqn_quad.episode_size):
            res = self.run_one_episode(self.dqn_quad, self.control_quad, epoch_id, i, mode='train')
            if res == 'break':
                break

    def test_quad(self):
        self.control_quad.reset()
        self.control_quad.SetTarget([2, 1, 1, 0])
        while not self.control_quad.check_target_reached():
            res = self.run_one_episode(self.dqn_quad, self.control_quad, 100, 100, mode='test')
            if res == 'Failure':
                print('Our quadrotor failed test.')
                break
            else:
                print('Continuing test.')
        if res != 'Failure':
            print("Our quadrotor has reached the test target.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch_size", type=int, default=10000, help="Total training epochs")
    parser.add_argument("--episode_size", type=int, default=100000, help="Training episodes per epoch")
    parser.add_argument("--epsilon", type=float, default=0.01, help="Greedy Epsilon starting value")
    parser.add_argument("--gamma", type=float, default=0.1, help="DQN gamma starting value")
    parser.add_argument("--load_model", action='store_true', default=False, help="Load saved model")
    parser.add_argument("--test", action='store_true', default=False, help="Testing phase")
    args = parser.parse_args()

    control_quad = QuadHelper()
    dqn_quad = QuadDQN()
    main_quad = Quad(dqn_quad=dqn_quad, control_quad=control_quad)

    # Argument parsing
    epoch_size = args.epoch_size
    dqn_quad.episode_size = args.episode_size
    dqn_quad.eps = args.epsilon
    dqn_quad.gamma = args.gamma
    if args.test:
        main_quad.mode = "test"

    if args.load_model == True:
        dqn_quad.load_wts('dqn_quad.pth')

    while (vrep.simxGetConnectionId(control_quad.sim_quad.clientID) != -1):
        with open('dqn_outputs.txt', 'a') as the_file:
            log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            the_file.write('\n************* SAVE FILE %s *************\n' % log_time)

        epoch = 0
        while epoch < epoch_size:
            main_quad.run_one_epoch(dqn_quad, control_quad, epoch)
            print("Epoch reset")
            epoch += 1
            if epoch % 10 == 0:
                dqn_quad.save_wts('dqn_quad.pth', epoch)
            if epoch % 5 == 0:
                dqn_quad.eps += (1. / (1. + dqn_quad.eps_decay * epoch_size))
                dqn_quad.gamma += (1. / (1. + dqn_quad.gamma_decay * epoch_size))
                control_quad.reset(rand_target=True)
            else:
                control_quad.reset()
            print('\n')

        # Test our trained quadrotor
        main_quad.test_quad(dqn_quad, control_quad)

        print("V-REP Exited...")


if __name__ == '__main__':
    main()
