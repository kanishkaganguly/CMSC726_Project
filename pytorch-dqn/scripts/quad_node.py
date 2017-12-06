#! /usr/bin/env python3
import argparse
import datetime

import numpy as np

import vrep
from pytorch_helper import QuadDQN
from quad_helper import QuadHelper

def write_data(dqn_quad, epoch, reward, iteration):
    with open('dqn_outputs.txt', 'a') as the_file:
        the_file.write('Epoch: %d Episode: %d\n' % (epoch, iteration))
        the_file.write('Epsilon Greedy: %f\n' % dqn_quad.eps)
        the_file.write('Reward: %f\n' % reward)
        the_file.write('Loss: %f\n' % float(dqn_quad.loss.data[0]))
        the_file.write('Learning Rate: %f\n' % float(dqn_quad.scheduler.get_lr()[0]))
        the_file.write('\n')

def run_one_episode(dqn_quad, control_quad, epoch_id, ep_id, mode):
    print("Epoch: %d Episode %d" % (epoch_id, ep_id))
    print("Epsilon Greedy: %f" % dqn_quad.eps)
    print("DQN Discount Factor: %f" % dqn_quad.gamma)

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
        if mode == 'test':
            return 'Failure'
        elif mode == 'train':
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
            return 'break'
        else:
            print("Running episode in invalid mode.")
    elif mode == 'train':
        # Get reward
        print("Getting reward")
        reward = dqn_quad.get_reward(new_state, control_quad.GetTargetState())
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

        if ep_id % 100 == 0:
            write_data(dqn_quad, epoch_id, reward, ep_id)

    return 'continue'

def run_one_epoch(dqn_quad, control_quad, epoch_id):
    for i in range(dqn_quad.episode_size):
        res = run_one_episode(dqn_quad, control_quad, epoch_id, i, mode='train')
        if res == 'break':
            break

def test_quad(dqn_quad, control_quad):
    control_quad.reset()
    control_quad.SetTarget([2,1,1,0])
    while(control_quad.ReachedTarget() == False):
        res = run_one_episode(dqn_quad, control_quad, 100, 100, mode='test')
        if res == 'Failure':
            print('Our quadroptor failed test.')
            break
        else:
            print('Continuing test.')
    if res != 'Failure':
        print("Our quadroptor has reached the test target.")

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

    if args.load_model == True:
        dqn_quad.load_wts('dqn_quad.pth')

    while (vrep.simxGetConnectionId(control_quad.sim_quad.clientID) != -1):
        with open('dqn_outputs.txt', 'a') as the_file:
            log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            the_file.write('\n************* SAVE FILE %s *************\n' % log_time)

        epoch = 0
        while epoch < epoch_size:
            run_one_epoch(dqn_quad, control_quad, epoch)
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
        test_quad(dqn_quad, control_quad)

        print("V-REP Exited...")

if __name__ == '__main__':
    main()
