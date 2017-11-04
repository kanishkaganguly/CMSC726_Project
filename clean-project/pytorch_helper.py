#!/usr/bin/python3

from itertools import combinations_with_replacement

import numpy as np
import torch
from torch.autograd import Variable
from torch import nn

class NNBase(object):
    def __init__(self, model, cuda=True):
        self.cuda = cuda
        self.model = model

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
        if self.cuda:
            self.model = self.model.cuda() # torch.nn.DataParallel(self.model).cuda()  #TODO dataparallel not working
        # Set optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Loss function
        self.loss_fn = torch.nn.MSELoss(size_average=False)

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
        self.output_var = self.model(Variable(state_data))
        return self.output_var

    def get_loss(self, pred, curr):
        self.loss = self.loss_fn(pred, curr)

    def do_backprop(self):
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    '''
    Convert reward to one hot vector for backprop
    '''

    def onehot_from_reward(self, reward, vector_length, one_hot_idx):
        vector = np.zeros((vector_length, 1), dtype=np.float32)
        vector[one_hot_idx] = reward
        return vector

class MLP(nn.Module):  #multi layer perceptron
    def __init__(self, n_inp, n_feature_list, n_class):
        super(MLP, self).__init__()
        self.layerlist = nn.ModuleList([])
        for idx, num_hidden_units in enumerate(n_feature_list):
            inp = n_inp if idx==0 else n_feature_list[idx-1]
            out = n_class if idx==len(n_feature_list)-1 else num_hidden_units
            self.layerlist += [nn.Linear(inp, out), nn.ReLU()]
        self.m = nn.Sequential(*self.layerlist)

    def forward(self, x):
        return self.m(x)


