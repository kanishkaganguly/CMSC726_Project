#!/usr/bin/python3

import torch


class NN():
    def __init__(self):
        # Tensor type
        self.dtype = torch.FloatTensor
        # Minibatch size
        self.N = 1
        # Input dimension (Position + Orientation)
        self.D_in = 6
        # Output dimension (Rotor Thrusts)
        self.D_out = 4
        # Hidden Layer 1 dimension
        self.H1 = 12
        # Hidden Layer 2 dimension
        self.H2 = 12

        # Reinforcement Learning Parameters
        self.learning_rate = 1e-4
        self.discount_factor = 1e-4
        self.reward = 0.0

        # Create model
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.D_in, self.H1),
            torch.nn.ReLU(),
            torch.nn.Linear(self.H2, self.D_out),
        )

        # Set optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Loss function
        self.loss_fn = torch.nn.MSELoss(size_average=False)

    def get_variable_from_array(self, array, requires_grad=False):
        d = torch.from_numpy(array).type(self.dtype)
        x = torch.autograd.Variable(d)
        return x

    def get_predicted_rotor(self, state_data):
        pred = self.model(state_data)
        return pred

    def get_loss(self, pred_rotor, curr_rotor):
        # loss = self.loss_fn(pred_rotor, curr_rotor)
        loss = (pred_rotor - curr_rotor).pow(2).sum()
        return loss

    def do_backprop(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return
