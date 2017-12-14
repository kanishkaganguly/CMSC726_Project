#! /usr/bin/env python3
import math
import os

import numpy as np
import torch
from torch.autograd import Variable


class QuadDQN(object):
    def __init__(self, cuda, epoch_size, episode_size):
        self.cuda = cuda
        self.epoch_size = epoch_size
        self.episode_size = episode_size
        self.input = 4
        self.action = 8
        self.hidden = 16
        if self.cuda:
            self.x = Variable(torch.randn(1, self.input)).cuda()
            self.y = Variable(torch.randn(1, self.action), requires_grad=False).cuda()
            self.model = torch.nn.Sequential(torch.nn.Linear(self.input, self.hidden), torch.nn.ReLU(),
                                             torch.nn.Linear(self.hidden, self.action)).cuda()
            self.loss_fn = torch.nn.MSELoss(size_average=False).cuda()
        else:
            self.x = Variable(torch.randn(1, self.input))
            self.y = Variable(torch.randn(1, self.action), requires_grad=False)
            self.model = torch.nn.Sequential(torch.nn.Linear(self.input, self.hidden), torch.nn.ReLU(),
                                             torch.nn.Linear(self.hidden, self.action))
            self.loss_fn = torch.nn.MSELoss(size_average=False)

        self.learning_rate = 0.1
        self.eps = 0.1
        self.eps_list = np.linspace(self.eps, 1.0, self.epoch_size, endpoint=True)
        self.gamma = 0.1
        self.gamma_list = np.linspace(self.gamma, 0.85, self.epoch_size, endpoint=True)

        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss = 0.0

        self.replay_buffer = []
        self.replay_state = {}
        self.buffer_size = 100
        self.curr_buffer_pointer = 0

    # Replay Buffer
    def push_to_buffer(self, target, pred, max_q_idx, reward):
        self.replay_state['target'] = target
        self.replay_state['pred'] = pred
        self.replay_state['max_q_idx'] = max_q_idx
        self.replay_state['reward'] = reward
        self.replay_buffer.insert(self.curr_buffer_pointer % self.buffer_size, self.replay_state)

    def pop_from_buffer(self):
        val = self.replay_buffer.pop()
        return val['target'], val['pred'], val['reward'], val['max_q_idx']

    # Predict next action
    def predict_action(self, state):
        if self.cuda:
            self.x = torch.from_numpy(state).cuda()
            self.y = self.model(Variable(self.x)).cuda()
            return self.y.data.cpu().numpy()
        else:
            self.x = torch.from_numpy(state)
            self.y = self.model(Variable(self.x))
            return self.y.data.numpy()

    # Get distance loss after action
    def get_loss(self, target, predicted):
        if self.cuda:
            self.y_pred = Variable(torch.from_numpy(predicted), requires_grad=True).cuda()
            self.y_tgt = Variable(torch.from_numpy(target), requires_grad=False).cuda()
        else:
            self.y_pred = Variable(torch.from_numpy(predicted), requires_grad=True)
            self.y_tgt = Variable(torch.from_numpy(target), requires_grad=False)
        self.loss = self.loss_fn(self.y_pred, self.y_tgt)
        print("Loss: %f" % self.loss.data[0])
        return

    # Do backprop
    def backprop(self):
        print("Learning Rate: %f" % self.learning_rate)

        self.optim.zero_grad()
        self.loss.backward()
        self.optim.step()

    # Get reward
    def get_reward(self, new_state, curr_state, target_state):
        prev_deviation_x = np.linalg.norm(curr_state[0] - target_state[0])
        prev_deviation_y = np.linalg.norm(curr_state[1] - target_state[1])
        prev_deviation_z = np.linalg.norm(curr_state[2] - target_state[2])
        prev_deviation_yaw = np.linalg.norm(curr_state[3] - target_state[3])

        curr_deviation_x = np.linalg.norm(new_state[0] - target_state[0])
        curr_deviation_y = np.linalg.norm(new_state[1] - target_state[1])
        curr_deviation_z = np.linalg.norm(new_state[2] - target_state[2])
        curr_deviation_yaw = np.linalg.norm(new_state[3] - target_state[3])

        if curr_deviation_x <= prev_deviation_x:
            reward_x = 1.0
        else:
            reward_x = -1.0
        if curr_deviation_y <= prev_deviation_y:
            reward_y = 1.0
        else:
            reward_y = -1.0
        if curr_deviation_z <= prev_deviation_z:
            reward_z = 1.0
        else:
            reward_z = -1.0
        if curr_deviation_yaw <= prev_deviation_yaw:
            reward_yaw = 1.0
        else:
            reward_yaw = -1.0

        reward = np.tanh(0.9 * reward_x + 0.9 * reward_y + 0.6 * reward_z + 0.1 * reward_yaw)

        print("Reward: %f" % reward)
        return reward

    # Sigmoid
    def sigmoid(self, val):
        return math.exp(val) / (math.exp(val) + 1)

    # Value to action converter
    def convert_action(self, action):
        converter = {
            0: 'FWD',
            1: 'BCK',
            2: 'LFT',
            3: 'RGT',
            4: 'UP',
            5: 'DWN',
            6: 'ROT_CW',
            7: 'ROT_CCW'
        }
        return converter.get(action)

    # Do action
    def do_action(self, action_val):
        if self.eps < np.random.rand():
            print("Picking random action")
            return self.convert_action(np.random.randint(0, self.action))
        else:
            return self.convert_action(action_val)

    # Save weights
    def save_wts(self, savefile, epoch):
        saveme = {
            'state_dict': self.model.state_dict(),
            'optimizer' : self.optim.state_dict(),
            'epoch'     : epoch,
            'epsilon'   : self.eps,
            'gamma'     : self.gamma
        }
        torch.save(saveme, savefile)

    # Load weights
    def load_wts(self, savefile):
        print("Loading saved model: %s\n" % savefile)
        if os.path.isfile(savefile):
            checkpoint = torch.load(savefile)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optim.load_state_dict(checkpoint['optimizer'])
            self.eps = checkpoint['epsilon']
            self.gamma = checkpoint['gamma']
        else:
            return 0
