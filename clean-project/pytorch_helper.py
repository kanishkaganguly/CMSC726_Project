#!/usr/bin/python3

from itertools import combinations_with_replacement

import numpy as np
import torch
from torch.autograd import Variable


class NN():
    def __init__(self):
        # Minibatch size
        self.N = 1
        # Input dimension (Position + Orientation)
        self.D_in = 0
        # Output dimension (Rotor Thrusts)
        self.D_out = 0
        # Hidden Layer 1 dimension
        self.H1 = 128
        # Hidden Layer 2 dimension
        self.H2 = 256
        # Hidden Layer 3 dimension
        self.H3 = 128

        # Reinforcement Learning Parameters
        self.learning_rate = 1e-4
        self.discount_factor = 1e-4
        self.reward = 0.0

    def create_model(self):
        # Create model
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.D_in, self.H1),
            torch.nn.Sigmoid(),
            torch.nn.ReLU(),
            torch.nn.Linear(self.H3, self.D_out),
        )
        self.model.cuda()

        # Set optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Loss function
        self.loss_fn = torch.nn.MSELoss(size_average=False)
        return

    def create_in_out_vars(self):
        self.input_var = Variable(torch.zeros(self.D_in, 1).cuda())
        self.output_var = Variable(torch.zeros(self.D_out, 1).cuda(), requires_grad=False)
        self.error_var = Variable(torch.zeros(self.D_out, 1).cuda(), requires_grad=False)

        return

    def generate_output_combos(self):
        delta_thrusts = np.linspace(-1, +1, 95, dtype=np.float32)
        rotor_combi = list(combinations_with_replacement(delta_thrusts, 4))
        return rotor_combi

    def np_to_torch(self, data):
        return torch.from_numpy(data).cuda()

    def torch_to_np(self, data):
        return data.cpu().numpy()

    def get_predicted_data(self, state_data):
        self.output_var = self.model(self.input_var)
        return self.output_var

    def get_loss(self, pred, real):
        self.loss = self.loss_fn(pred, real)
        return

    def do_backprop(self):
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        return

    def save_model(self):
        self.model.save_state_dict("quad_rl_model.pt")

    def load_model(self):
        self.model.load_state_dict("quad_rl_model.pt")

    '''
    Convert reward to one hot vector for backprop
    '''

    def onehot_from_reward(self, reward, vector_length, one_hot_idx):
        vector = np.zeros((vector_length, 1), dtype=np.float32)
        vector[one_hot_idx] = reward
        return vector
