#! /usr/bin/env python3
import argparse
import datetime

import vrep
from pytorch_helper import QuadDQN
from quad import Quad
from quad_helper import QuadHelper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch_size", type=int, default=10000, help="Total training epochs")
    parser.add_argument("--episode_size", type=int, default=100000, help="Training episodes per epoch")
    parser.add_argument("--epsilon", type=float, default=0.01, help="Greedy Epsilon starting value")
    parser.add_argument("--gamma", type=float, default=0.01, help="DQN gamma starting value")
    parser.add_argument("--load_model", action='store_true', default=False, help="Load saved model")
    parser.add_argument("--test", action='store_true', default=False, help="Testing phase")
    args = parser.parse_args()

    # Initialize classes
    control_quad = QuadHelper()
    dqn_quad = QuadDQN()
    main_quad = Quad(dqn_quad=dqn_quad, control_quad=control_quad)

    # Argument parsing
    main_quad.epoch_size = args.epoch_size
    dqn_quad.episode_size = args.episode_size
    dqn_quad.eps = args.epsilon
    dqn_quad.gamma = args.gamma
    if args.load_model == True:
        dqn_quad.load_wts('dqn_quad.pth')

    while (vrep.simxGetConnectionId(control_quad.sim_quad.clientID) != -1):
        with open('dqn_outputs.txt', 'a') as the_file:
            log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            the_file.write('\n************* SAVE FILE %s *************\n' % log_time)

        if args.test:
            print("Testing Quadrotor")
            # Test our trained quadrotor
            main_quad.mode = 'test'
            dqn_quad.load_wts('dqn_quad.pth')
            main_quad.test_quad()
        else:
            # Train quadcopter
            epoch = 0
            while epoch < main_quad.epoch_size:
                main_quad.run_one_epoch(epoch)
                print("Epoch reset")
                epoch += 1
                main_quad.task_every_n_epochs(epoch)
                print('\n')
            print("Finished training")

            # Test quadrotor
            main_quad.mode = "test"
            main_quad.test_quad(dqn_quad, control_quad)

    control_quad.sim_quad.exit_sim()
    print("V-REP Exited...")


if __name__ == '__main__':
    main()
