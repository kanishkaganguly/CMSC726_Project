#!/usr/bin/python3

from itertools import combinations_with_replacement

import numpy as np
import torch
from torch.autograd import Variable


class NN():
    def __init__(self, cuda=False):
        # Tensor type
        self.dtype = torch.FloatTensor
        # Minibatch size
        self.N = 1
        # Input dimension (Position + Orientation)
        self.D_in = 0
        # Output dimension (Rotor Thrusts)
        self.D_out = 0
        # Hidden Layer 1 dimension
        self.H1 = 12
        # Hidden Layer 2 dimension
        self.H2 = 40
        # Hidden Layer 3 dimension
        self.H3 = 12
        self.cuda = cuda

        # Reinforcement Learning Parameters
        self.learning_rate = 1e-4
        self.discount_factor = 1e-4
        self.reward = 0.0

    def load_wts(self, modelfile):
        if os.path.isfile(modelfile):
            checkpoint = torch.load(modelfile)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

    def save_wts(self, savefile):
        saveme = {  #TODO save other stuff too, like epoch etc
            'state_dict': self.model.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
        }
        torch.save(saveme, savefile)

    def create_model(self):
        # Create model
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.D_in, self.H1),
            torch.nn.ReLU(),
            torch.nn.ReLU(),
            torch.nn.Linear(self.H3, self.D_out),
        )
        if self.cuda:
            self.model = torch.nn.DataParallel(self.model).cuda()
        # Set optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Loss function
        self.loss_fn = torch.nn.MSELoss(size_average=False)

    def create_in_out_vars(self):  #TODO: are these needed? no need to init input_var and output_var
        self.input_var = Variable(torch.zeros(self.D_in, 1))
        self.output_var = Variable(torch.zeros(self.D_out, 1), requires_grad=False)  #TODO, check why require_grads = False
        if self.cuda:
            self.input_var = self.input_var.cuda()
            self.output_var = self.output_var.cuda()

    def generate_output_combos(self):
        delta_thrusts = np.linspace(-2, +2, 200, dtype=np.float32)
        rotor_combi = list(combinations_with_replacement(delta_thrusts, 4))
        return rotor_combi

    def np_to_torch(self, data):
        torchtensor = torch.from_numpy(data)
        return torchtensor.cuda() if self.cuda else torchtensor

    def torch_to_np(self, data):
        return data.cpu().numpy() if self.cuda else data.numpy()

    def get_predicted_data(self, state_data):
        self.output_var = self.model(self.input_var) #TODO: state_data is input but never used. model should predict on state_data right. self.input_var and self.output_var probably not needed?
        return self.output_var

    def get_loss(self, pred, curr):
        return self.loss_fn(pred, curr)

    def do_backprop(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
