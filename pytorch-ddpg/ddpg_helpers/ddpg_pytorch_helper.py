#!/usr/bin/python3

from itertools import combinations_with_replacement
from collections import deque
import numpy as np
import torch, os
from torch.autograd import Variable
from torch import nn

from ddpg_memory import SequentialMemory
from ddpg_random_process import OrnsteinUhlenbeckProcess
from ddpg_utils import *
from ddpg_model import Actor, Critic


class DDPG(object):
    def __init__(self,  nb_states, nb_actions, args):
        # self.cuda = USE_CUDA #args.cuda
        self.cuda = args.cuda

        self.nb_states = nb_states
        self.nb_actions = nb_actions
        
        #Init models
        #actor_kwargs = {'n_inp':self.nb_states, 'n_feature_list':[args.hidden1,args.hidden2], 'n_class':self.nb_actions}
        #self.actor = MLP(**actor_kwargs)
        #self.actor_target = MLP(**actor_kwargs)
        #self.critic = MLP(**actor_kwargs)  #TODO: actor and critic has same structure for now.
        #self.critic_target = MLP(**actor_kwargs)

        net_cfg = {
            'hidden1':args.hidden1,
            'hidden2':args.hidden2,
            'init_w':args.init_w
        }
        self.actor = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_target = Actor(self.nb_states, self.nb_actions, **net_cfg)

        self.critic = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_target = Critic(self.nb_states, self.nb_actions, **net_cfg)

        self.criterion = nn.MSELoss()
        if self.cuda:
            self.actor = self.actor.cuda() # torch.nn.DataParallel(self.model).cuda()  #TODO dataparallel not working
            self.critic = self.critic.cuda()
            self.actor_target = self.actor_target.cuda()
            self.critic_target = self.critic_target.cuda()
            self.criterion = self.criterion.cuda()

        # Set optimizer  
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=args.prate)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=args.rate)
        # Loss function
        self.loss_fn = torch.nn.MSELoss(size_average=False)

        hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)
        
        self.memory = SequentialMemory(limit=args.rmsize, window_length=args.window_length)
        self.random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=args.ou_theta, mu=args.ou_mu, sigma=args.ou_sigma)

        # Hyper-parameters
        self.batch_size = args.bsize
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon

        self.epsilon = 1.0
        self.s_t = None # Most recent state
        self.a_t = None # Most recent action
        self.is_training = True

    def update_policy(self):
        # Sample batch
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)

        # Prepare for the target q batch
        next_q_values = self.critic_target([
            to_tensor(next_state_batch, volatile=True),
            self.actor_target(to_tensor(next_state_batch, volatile=True)),
        ])
        next_q_values.volatile=False

        target_q_batch = to_tensor(reward_batch) + \
            self.discount*to_tensor(terminal_batch.astype(np.float))*next_q_values

        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic([ to_tensor(state_batch), to_tensor(action_batch) ])
        
        value_loss = self.criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()

        policy_loss = -self.critic([
            to_tensor(state_batch),
            self.actor(to_tensor(state_batch))
        ])

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()
        
    def random_action(self):
        action = np.random.uniform(-1.,1.,self.nb_actions)
        self.a_t = action
        return action
        
    def observe(self, r_t, s_t1, done):
        if self.is_training:
            self.memory.append(self.s_t, self.a_t, r_t, done)
            self.s_t = s_t1
        
    def select_action(self, s_t, decay_epsilon=True):
        action = to_numpy(
            self.actor(to_tensor(np.array([s_t])))
        ).squeeze(0)
        action += self.is_training*max(self.epsilon, 0)*self.random_process.sample()
        action = np.clip(action, -1., 1.)

        if decay_epsilon:
            self.epsilon -= self.depsilon
        
        self.a_t = action
        return action

    def reset(self, obs):
        self.s_t = obs
        self.random_process.reset_states()

    def load_wts(self, modelfile):
        if os.path.isfile(modelfile + 'model.pth.tar'):
            checkpoint = torch.load(modelfile)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.actor_optim.load_state_dict(checkpoint['actor_optim'])
            self.critic_optim.load_state_dict(checkpoint['critic_optim'])
            return checkpoint['step']
        else:
            return  0

    def save_wts(self, savefile, step):
        saveme = {  #TODO save other stuff too, like epoch etc
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optim' : self.actor_optim.state_dict(),
            'critic_optim' : self.critic_optim.state_dict(),
            'step' : step
        }
        torch.save(saveme, savefile + 'model.pth.tar')

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


