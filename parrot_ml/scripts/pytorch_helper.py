#! /usr/bin/env python
import torch
from torch.autograd import Variable
import numpy as np
import math


class QuadDQN(object):
    def __init__(self):
        self.N = 1
        self.input = 4
        self.action = 8
        self.hidden = 16
        self.model = torch.nn.Sequential(torch.nn.Linear(self.input, self.hidden), torch.nn.ReLU(),
                                         torch.nn.Linear(self.hidden, self.action))
        self.loss_fn = torch.nn.MSELoss(size_average=False)
        self.learning_rate = 1e-4
        self.eps = 0.2
        self.gamma = 0.9
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss = 0.0

    # Predict next action
    def predict_action(self, state):
        self.x = torch.from_numpy(state)
        self.y = self.model(Variable(self.x))
        return self.y.data.numpy()

    # Get distance loss after action
    def get_loss(self, target, predicted):
        self.loss = self.loss_fn(Variable(torch.from_numpy(predicted), requires_grad=True),
                                 Variable(torch.from_numpy(target), requires_grad=False))
        print("Loss: %f" % self.loss.data[0])
        return

    # Do backprop
    def backprop(self):
        self.optim.zero_grad()
        self.loss.backward()
        self.optim.step()

    # Get reward
    def get_reward(self, curr_state, target_state):
        deviation_x = np.linalg.norm(curr_state[0] - target_state[0])
        deviation_y = np.linalg.norm(curr_state[1] - target_state[1])
        deviation_z = np.linalg.norm(curr_state[2] - target_state[2])
        deviation_yaw = np.linalg.norm(curr_state[3] - target_state[3])

        sigma_x = 0.1
        sigma_y = 0.1
        sigma_z = 0.01
        sigma_yaw = 0.1
        reward_x = math.exp(-deviation_x ** 2 / (2 * sigma_x))
        reward_y = math.exp(-deviation_y ** 2 / (2 * sigma_y))
        reward_z = math.exp(-deviation_z ** 2 / (2 * sigma_z))
        reward_yaw = math.exp(-deviation_yaw ** 2 / (2 * sigma_yaw))

        reward = self.sigmoid(reward_x + reward_y + reward_z + reward_yaw)
        print("Position Reward: %f" % reward)
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
        if self.eps > np.random.rand():
            print("Picking random action")
            return self.convert_action(np.random.randint(0, self.action))
        else:
            return self.convert_action(action_val)
